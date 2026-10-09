"""
AESW Next-Hop Decision Engine
=============================
Implements candidate neighbor scoring, numerically stable softmax action selection,
and explainable next-hop decisions for Adaptive Evidence-Sharing Walkers (AESW).

Mathematical Formulation:
    Candidate Score:
        S(u) = a * P(u) + b * N(u) - g * R(u) - d * D(u)

    where:
        P(u) = Positive evidence strength in [0.0, 1.0] (from M8 EvidenceCache with M9 churn decay)
        N(u) = Local novelty / new-information value in (0.0, 1.0] (1 / (1 + visit_count))
        R(u) = Local revisit penalty >= 0.0 (visit_count)
        D(u) = Locally observable traversal delay / movement cost >= 1.0
        a, b, g, d = Configurable non-negative weights

    Softmax Selection Probability:
        P_select(u) = exp((S(u) - max_v S(v)) / T) / sum_w exp((S(w) - max_v S(v)) / T)
        where T > 0 is the exploration temperature.

Architectural Guarantees:
1. Strict Epistemic Firewall: Operates exclusively on locally observed candidate links.
   Zero access to ground-truth graph, p_on/p_off, or true target location.
2. Numerical Stability: Stabilized softmax with max-subtraction preventing overflow.
3. Decoupled Responsibilities: Does NOT execute movement or simulation updates.
4. Explainable Output: Exposes individual score components and selection probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np

from aesw.aesw.mode import SearchMode
from aesw.baselines.actions import SearchAction
from aesw.baselines.types import ActionType
from aesw.environment.observation import Observation
from aesw.memory.cache import EvidenceCache


@dataclass(frozen=True)
class CandidateScore:
    """Explainable multi-criteria score breakdown for a single candidate node.

    Attributes:
        node_id: Identifier of the candidate node u.
        positive_evidence: P(u) in [0.0, 1.0], derived from decayed positive evidence.
        novelty: N(u) in (0.0, 1.0], inversely proportional to local visit count.
        revisit_penalty: R(u) >= 0.0, local visit frequency penalty.
        delay: D(u) >= 1.0, locally observable traversal duration.
        total_score: S(u) = a*P + b*N - g*R - d*D.
    """
    node_id: Union[int, str]
    positive_evidence: float
    novelty: float
    revisit_penalty: float
    delay: float
    total_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert score breakdown to a dictionary."""
        return {
            "node_id": self.node_id,
            "positive_evidence": self.positive_evidence,
            "novelty": self.novelty,
            "revisit_penalty": self.revisit_penalty,
            "delay": self.delay,
            "total_score": self.total_score,
        }


@dataclass(frozen=True)
class NextHopDecision:
    """Structured, explainable decision result emitted by the NextHopDecisionEngine.

    Attributes:
        mode: SearchMode under which the decision was evaluated (LOCAL or LONG_JUMP).
        selected_node: Identifier of the chosen candidate, or None if no candidate exists.
        candidates: Tuple of all evaluated legal candidate node IDs.
        scores: Mapping from candidate node ID to its CandidateScore breakdown.
        probabilities: Mapping from candidate node ID to its softmax selection probability.
        selected_probability: Softmax probability of the chosen candidate.
        action: Formatted SearchAction ready for downstream execution.
        lambda_hat: Churn rate estimate applied during evidence evaluation.
        metadata: Diagnostic audit details.
    """
    mode: SearchMode
    selected_node: Optional[Union[int, str]]
    candidates: tuple[Union[int, str], ...]
    scores: Mapping[Union[int, str], CandidateScore]
    probabilities: Mapping[Union[int, str], float]
    selected_probability: float
    action: SearchAction
    lambda_hat: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert decision details to an exportable dictionary for diagnostics and UI."""
        return {
            "mode": self.mode.value,
            "selected_node": self.selected_node,
            "candidates": list(self.candidates),
            "scores": {str(k): v.to_dict() for k, v in self.scores.items()},
            "probabilities": {str(k): v for k, v in self.probabilities.items()},
            "selected_probability": self.selected_probability,
            "action_type": self.action.action_type.value,
            "action_destination": self.action.destination,
            "action": {
                "action_type": self.action.action_type.value,
                "destination": self.action.destination,
                "metadata": dict(self.action.metadata or {}),
            },
            "lambda_hat": self.lambda_hat,
            "metadata": dict(self.metadata),
        }


class CandidateScorer:
    """Computes multi-factor scores S(u) = a*P(u) + b*N(u) - g*R(u) - d*D(u).

    Attributes:
        weight_positive: Weight a for positive evidence term P(u).
        weight_novelty: Weight b for novelty term N(u).
        weight_revisit: Weight g for revisit penalty term R(u).
        weight_delay: Weight d for traversal delay term D(u).
    """

    def __init__(
        self,
        weight_positive: float = 1.0,
        weight_novelty: float = 1.0,
        weight_revisit: float = 1.0,
        weight_delay: float = 1.0,
    ) -> None:
        """Initialize candidate scoring weights.

        Args:
            weight_positive: Weight a >= 0.0 for positive evidence. Defaults to 1.0.
            weight_novelty: Weight b >= 0.0 for novelty. Defaults to 1.0.
            weight_revisit: Weight g >= 0.0 for revisit penalty. Defaults to 1.0.
            weight_delay: Weight d >= 0.0 for traversal delay. Defaults to 1.0.

        Raises:
            ValueError: If any weight is negative.
        """
        if weight_positive < 0.0:
            raise ValueError(f"weight_positive must be non-negative, got {weight_positive}")
        if weight_novelty < 0.0:
            raise ValueError(f"weight_novelty must be non-negative, got {weight_novelty}")
        if weight_revisit < 0.0:
            raise ValueError(f"weight_revisit must be non-negative, got {weight_revisit}")
        if weight_delay < 0.0:
            raise ValueError(f"weight_delay must be non-negative, got {weight_delay}")

        self._weight_positive = float(weight_positive)
        self._weight_novelty = float(weight_novelty)
        self._weight_revisit = float(weight_revisit)
        self._weight_delay = float(weight_delay)

    @property
    def weight_positive(self) -> float:
        return self._weight_positive

    @property
    def weight_novelty(self) -> float:
        return self._weight_novelty

    @property
    def weight_revisit(self) -> float:
        return self._weight_revisit

    @property
    def weight_delay(self) -> float:
        return self._weight_delay

    def compute_positive_evidence(
        self,
        node_id: Union[int, str],
        evidence_cache: Optional[EvidenceCache],
        current_time: int,
        lambda_hat: float = 0.0,
    ) -> float:
        """Compute positive evidence term P(u) in [0.0, 1.0].

        Returns time-decayed effective confidence if positive evidence exists for node_id,
        otherwise returns 0.0.
        """
        if evidence_cache is None:
            return 0.0

        weighted = evidence_cache.get(node_id=node_id, current_time=current_time, lambda_hat=lambda_hat)
        if weighted is None or not weighted.is_positive:
            return 0.0

        return max(0.0, min(1.0, float(weighted.effective_confidence)))

    @classmethod
    def compute_novelty(
        cls,
        node_id: Union[int, str, float] = 0,
        visit_counts: Optional[Mapping[Union[int, str], int]] = None,
    ) -> float:
        """Compute novelty term N(u) in (0.0, 1.0].

        Accepts either (visit_count: int) directly or (node_id, visit_counts).
        N(u) = 1.0 / (1.0 + visit_count). Unvisited nodes receive 1.0; frequently visited nodes approach 0.
        """
        if visit_counts is not None and isinstance(node_id, (str, int)) and node_id in visit_counts:
            v_count = int(visit_counts[node_id])
        elif isinstance(node_id, (int, float)) and visit_counts is None:
            v_count = int(node_id)
        else:
            v_count = 0

        if v_count < 0:
            v_count = 0

        return float(1.0 / (1.0 + float(v_count)))

    @classmethod
    def compute_revisit_penalty(
        cls,
        node_id: Union[int, str, float] = 0,
        visit_counts: Optional[Mapping[Union[int, str], int]] = None,
    ) -> float:
        """Compute revisit penalty term R(u) >= 0.0.

        Accepts either (visit_count: int) directly or (node_id, visit_counts).
        Distinguishes never visited (0.0), visited once (1.0), and multiple visits (> 1.0).
        """
        if visit_counts is not None and isinstance(node_id, (str, int)) and node_id in visit_counts:
            v_count = int(visit_counts[node_id])
        elif isinstance(node_id, (int, float)) and visit_counts is None:
            v_count = int(node_id)
        else:
            v_count = 0

        return float(max(0, v_count))

    @classmethod
    def compute_delay(
        cls,
        node_id: Union[int, str],
        traversal_delays: Optional[Mapping[Union[int, str], float]] = None,
        default_delay: float = 1.0,
        delays: Optional[Mapping[Union[int, str], float]] = None,
    ) -> float:
        """Compute traversal delay term D(u) >= 1.0.

        Consumes locally available delay if provided, otherwise defaults to standard unit step delay.
        """
        mapping = delays if delays is not None else traversal_delays
        if mapping is not None and node_id in mapping:
            return float(max(1.0, mapping[node_id]))
        return float(max(1.0, default_delay))

    def score_candidate(
        self,
        node_id: Union[int, str],
        evidence_cache: Optional[EvidenceCache] = None,
        current_time: int = 0,
        lambda_hat: float = 0.0,
        visit_counts: Optional[Mapping[Union[int, str], int]] = None,
        traversal_delays: Optional[Mapping[Union[int, str], float]] = None,
        visit_count: Optional[int] = None,
        traversal_delay: Optional[float] = None,
        delays: Optional[Mapping[Union[int, str], float]] = None,
    ) -> CandidateScore:
        """Score an individual candidate node u across all four objective criteria."""
        effective_visit_counts = visit_counts
        if visit_count is not None:
            effective_visit_counts = {node_id: visit_count}

        effective_delays = delays if delays is not None else traversal_delays
        if traversal_delay is not None:
            effective_delays = {node_id: traversal_delay}

        p_val = self.compute_positive_evidence(
            node_id=node_id,
            evidence_cache=evidence_cache,
            current_time=current_time,
            lambda_hat=lambda_hat,
        )
        n_val = self.compute_novelty(node_id=node_id, visit_counts=effective_visit_counts)
        r_val = self.compute_revisit_penalty(node_id=node_id, visit_counts=effective_visit_counts)
        d_val = self.compute_delay(node_id=node_id, traversal_delays=effective_delays)

        total = (
            self._weight_positive * p_val
            + self._weight_novelty * n_val
            - self._weight_revisit * r_val
            - self._weight_delay * d_val
        )

        return CandidateScore(
            node_id=node_id,
            positive_evidence=p_val,
            novelty=n_val,
            revisit_penalty=r_val,
            delay=d_val,
            total_score=total,
        )

    def score_all(
        self,
        candidates: Sequence[Union[int, str]],
        evidence_cache: Optional[EvidenceCache] = None,
        current_time: int = 0,
        lambda_hat: float = 0.0,
        visit_counts: Optional[Mapping[Union[int, str], int]] = None,
        traversal_delays: Optional[Mapping[Union[int, str], float]] = None,
    ) -> dict[Union[int, str], CandidateScore]:
        """Score all candidate nodes, returning a dictionary mapping node_id -> CandidateScore."""
        return {
            u: self.score_candidate(
                node_id=u,
                evidence_cache=evidence_cache,
                current_time=current_time,
                lambda_hat=lambda_hat,
                visit_counts=visit_counts,
                traversal_delays=traversal_delays,
            )
            for u in candidates
        }


class SoftmaxSelector:
    """Computes numerically stabilized softmax probabilities and performs stochastic selection."""

    @staticmethod
    def compute_probabilities(
        scores: Mapping[Union[int, str], CandidateScore],
        temperature: float = 1.0,
    ) -> dict[Union[int, str], float]:
        """Compute numerically stable softmax probabilities P_select(u).

        Formulation:
            z_u = S(u) / T
            z_max = max_v z_v
            exp_u = exp(z_u - z_max)
            P(u) = exp_u / sum_v exp_v

        Edge cases:
            - Exactly 1 candidate: probability = 1.0 deterministically.
            - 0 candidates: returns empty dictionary.

        Args:
            scores: Mapping from candidate node ID to CandidateScore.
            temperature: Exploration temperature T > 0.

        Returns:
            Mapping from candidate node ID to normalized probability P(u) in [0.0, 1.0].

        Raises:
            ValueError: If temperature <= 0.0.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature T must be strictly positive (> 0), got {temperature}")

        if not scores:
            return {}

        candidates = list(scores.keys())
        if len(candidates) == 1:
            return {candidates[0]: 1.0}

        raw_scores = [scores[c].total_score for c in candidates]
        z_vals = [s / temperature for s in raw_scores]
        z_max = max(z_vals)

        # Numerically stabilized exponentials
        exp_vals = [math.exp(z - z_max) for z in z_vals]
        total_exp = sum(exp_vals)

        if total_exp <= 0.0 or not math.isfinite(total_exp):
            # Uniform fallback in extreme numerical degradation
            uniform_p = 1.0 / len(candidates)
            return {c: uniform_p for c in candidates}

        probabilities = {}
        for c, exp_v in zip(candidates, exp_vals):
            p = max(0.0, min(1.0, exp_v / total_exp))
            probabilities[c] = p

        # Renormalize to ensure exact sum == 1.0
        prob_sum = sum(probabilities.values())
        if prob_sum > 0.0:
            for c in probabilities:
                probabilities[c] /= prob_sum

        return probabilities

    @staticmethod
    def select_candidate(
        probabilities: Mapping[Union[int, str], float],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Optional[Union[int, str]], float]:
        """Select a candidate node probabilistically according to softmax distribution.

        Args:
            probabilities: Mapping from candidate node ID to probability P(u).
            rng: Optional isolated NumPy random Generator.

        Returns:
            Tuple of (selected_node_id, selected_probability). If empty, returns (None, 0.0).
        """
        if not probabilities:
            return None, 0.0

        candidates = list(probabilities.keys())
        if len(candidates) == 1:
            return candidates[0], 1.0

        probs = np.array([probabilities[c] for c in candidates], dtype=np.float64)
        # Re-normalize to guarantee sum == 1.0 for np.random.Generator
        prob_sum = probs.sum()
        if prob_sum > 0.0:
            probs = probs / prob_sum
        else:
            probs = np.ones(len(candidates), dtype=np.float64) / len(candidates)

        active_rng = rng if rng is not None else np.random.default_rng()
        chosen_idx = int(active_rng.choice(len(candidates), p=probs))
        chosen_node = candidates[chosen_idx]
        return chosen_node, float(probabilities[chosen_node])


class NextHopDecisionEngine:
    """AESW Next-Hop Decision Engine.

    Coordinates multi-criteria scoring, softmax probability evaluation, and action selection
    for an individual searcher based strictly on local observation, memory, and search mode.

    Attributes:
        scorer: CandidateScorer instance.
        temperature: Exploration temperature T > 0.
    """

    DEFAULT_TEMPERATURE: float = 1.0

    def __init__(
        self,
        weight_positive: float = 1.0,
        weight_novelty: float = 1.0,
        weight_revisit: float = 1.0,
        weight_delay: float = 1.0,
        temperature: float = DEFAULT_TEMPERATURE,
        mode_weight_scaling: bool = False,
    ) -> None:
        """Initialize next-hop decision engine.

        Args:
            weight_positive: Weight a >= 0.0 for positive evidence P(u).
            weight_novelty: Weight b >= 0.0 for novelty N(u).
            weight_revisit: Weight g >= 0.0 for revisit penalty R(u).
            weight_delay: Weight d >= 0.0 for traversal delay D(u).
            temperature: Softmax exploration temperature T > 0. Defaults to 1.0.
            mode_weight_scaling: If True, dynamically scales exploration weights under LONG_JUMP mode.

        Raises:
            ValueError: If any weight is negative or temperature <= 0.0.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature T must be strictly positive (> 0), got {temperature}")

        self._scorer = CandidateScorer(
            weight_positive=weight_positive,
            weight_novelty=weight_novelty,
            weight_revisit=weight_revisit,
            weight_delay=weight_delay,
        )
        self._temperature = float(temperature)
        self._mode_weight_scaling = bool(mode_weight_scaling)

    @property
    def scorer(self) -> CandidateScorer:
        """Candidate scoring instance."""
        return self._scorer

    @property
    def temperature(self) -> float:
        """Exploration temperature T."""
        return self._temperature

    @property
    def mode_weight_scaling(self) -> bool:
        """Whether mode-dependent weight scaling is enabled."""
        return self._mode_weight_scaling

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"Temperature T must be strictly positive (> 0), got {value}")
        self._temperature = float(value)

    @staticmethod
    def extract_legal_candidates(observation: Observation) -> list[Union[int, str]]:
        """Extract legal active neighbor candidates directly from an Observation snapshot.

        Filters checked_neighbors to those whose edge state is confirmed active (ON).
        """
        candidates = [
            nbr for nbr in observation.checked_neighbors
            if observation.is_edge_known_active(nbr)
        ]
        return sorted(candidates, key=str)

    def decide(
        self,
        observation: Optional[Observation] = None,
        evidence_cache: Optional[EvidenceCache] = None,
        current_time: int = 0,
        mode: Union[str, SearchMode] = SearchMode.LOCAL,
        candidates: Optional[Sequence[Union[int, str]]] = None,
        visit_counts: Optional[Mapping[Union[int, str], int]] = None,
        traversal_delays: Optional[Mapping[Union[int, str], float]] = None,
        lambda_hat: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> NextHopDecision:
        """Evaluate legal candidates, compute softmax distribution, and decide next action.

        Args:
            observation: Local Observation snapshot from which candidates and current node are extracted.
            evidence_cache: Walker's private local EvidenceCache.
            current_time: Current discrete simulation timestamp t.
            mode: SearchMode from M11 ModeController (LOCAL or LONG_JUMP).
            candidates: Optional explicit candidate pool override. If omitted, extracted from observation.
            visit_counts: Local mapping from vertex ID to visit count.
            traversal_delays: Optional mapping of local edge traversal durations.
            lambda_hat: Current churn estimate from M9 ChurnEstimator for evidence decay.
            rng: Optional isolated NumPy random Generator.

        Returns:
            NextHopDecision containing selected candidate, full scores, probabilities, and SearchAction.
        """
        resolved_mode = SearchMode(str(mode).upper()) if not isinstance(mode, SearchMode) else mode

        # Determine legal candidate set strictly bounded by partial visibility
        if candidates is not None:
            candidate_list = list(candidates)
        elif observation is not None:
            candidate_list = self.extract_legal_candidates(observation)
        else:
            candidate_list = []

        current_node = observation.current_node if observation is not None else None

        # Case 1: Empty candidate set (e.g. isolated node or all incident links OFF)
        if not candidate_list:
            action = SearchAction(
                action_type=ActionType.STAY,
                destination=current_node,
                metadata={
                    "mode": resolved_mode.value,
                    "reason": "no_available_candidates",
                },
            )
            return NextHopDecision(
                mode=resolved_mode,
                selected_node=None,
                candidates=(),
                scores={},
                probabilities={},
                selected_probability=0.0,
                action=action,
                lambda_hat=lambda_hat,
                metadata={"reason": "no_available_candidates"},
            )

        # Case 2: Score all legal candidates
        scorer = self._scorer
        if self._mode_weight_scaling and resolved_mode == SearchMode.LONG_JUMP:
            scorer = CandidateScorer(
                weight_positive=self._scorer.weight_positive * 0.5,
                weight_novelty=self._scorer.weight_novelty * 1.5,
                weight_revisit=self._scorer.weight_revisit * 0.5,
                weight_delay=self._scorer.weight_delay,
            )

        scores = scorer.score_all(
            candidates=candidate_list,
            evidence_cache=evidence_cache,
            current_time=current_time,
            lambda_hat=lambda_hat,
            visit_counts=visit_counts,
            traversal_delays=traversal_delays,
        )

        # Case 3: Compute numerically stabilized softmax probabilities
        probabilities = SoftmaxSelector.compute_probabilities(
            scores=scores,
            temperature=self._temperature,
        )

        # Case 4: Stochastic selection using supplied isolated RNG
        selected_node, selected_prob = SoftmaxSelector.select_candidate(
            probabilities=probabilities,
            rng=rng,
        )

        # Build formatted SearchAction
        action_metadata = {
            "mode": resolved_mode.value,
            "score": scores[selected_node].total_score if selected_node is not None else 0.0,
            "probability": selected_prob,
        }
        action = SearchAction(
            action_type=ActionType.MOVE,
            destination=selected_node,
            metadata=action_metadata,
        )

        return NextHopDecision(
            mode=resolved_mode,
            selected_node=selected_node,
            candidates=tuple(candidate_list),
            scores=scores,
            probabilities=probabilities,
            selected_probability=selected_prob,
            action=action,
            lambda_hat=lambda_hat,
            metadata={
                "temperature": self._temperature,
                "candidate_count": len(candidate_list),
            },
        )
