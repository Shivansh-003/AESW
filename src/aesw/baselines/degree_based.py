"""
Degree-Based Walk Policy (Baseline 4)
=====================================
Search policy preferring locally observed neighbors with higher observed degree.
Strictly adheres to partial observability: uses ONLY locally observed degrees,
never querying global graph degrees or hidden topology.
"""

from typing import Optional
import numpy as np

from aesw.utils.reproducibility import create_rng
from aesw.baselines.types import BaselineType, ActionType
from aesw.baselines.actions import SearchAction
from aesw.baselines.base import BaselinePolicy, BaselineMetadata
from aesw.environment.observation import Observation


class DegreeBasedWalkPolicy(BaselinePolicy):
    """Degree-biased walk policy under partial visibility.

    Information Constraint:
    The walker has no access to global graph topology or true unobserved vertex degrees.
    Locally Observable Degree Model:
    - When the walker occupies node u, it legitimately observes the number of active incident links:
      deg_obs(u) = |N_obs(u)|. This is stored in local policy memory.
    - For an unvisited neighbor v, the policy assigns a conservative prior degree deg_prior = 1.0
      (the observed incident edge from u).
    - Candidate score:
      score(v) = deg_obs(v) + epsilon
    - Transition probability:
      P(v) = score(v) / sum_{w} score(w)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        epsilon: float = 1e-3,
        default_prior_degree: float = 1.0,
    ) -> None:
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        if default_prior_degree <= 0.0:
            raise ValueError(f"default_prior_degree must be strictly positive, got {default_prior_degree}")

        self._seed = seed
        self._epsilon = float(epsilon)
        self._default_prior = float(default_prior_degree)
        self._rng: np.random.Generator = create_rng(seed)

        # Local cache of legitimately observed vertex degrees: node_id -> observed active degree
        self._observed_degrees: dict[int | str, float] = {}

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.DEGREE_BASED

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def observed_degrees(self) -> dict[int | str, float]:
        """Read-only view of legitimately observed vertex degrees."""
        return dict(self._observed_degrees)

    @property
    def metadata(self) -> BaselineMetadata:
        return BaselineMetadata(
            name="Degree-Based Walk",
            baseline_type=BaselineType.DEGREE_BASED,
            description="Topology-biased walk preferring neighbors with higher locally observed degree.",
            uses_memory=True,
            uses_detection_signals=False,
            uses_communication=False,
            is_multi_walker=False,
            parameters={"epsilon": self._epsilon, "default_prior_degree": self._default_prior},
        )

    def decide(self, observation: Observation) -> SearchAction:
        """Select next hop biased by locally observable degree."""
        candidates = self.get_available_candidates(observation)

        # Update degree knowledge for the currently occupied node from observation
        current_node = observation.current_node
        self._observed_degrees[current_node] = float(len(candidates))

        if not candidates:
            action = SearchAction(
                action_type=ActionType.STAY,
                destination=current_node,
                metadata={"reason": "no_available_neighbors"},
            )
            return self._validate_action(action, observation)

        # Compute locally observable scores
        scores = np.array([
            self._observed_degrees.get(v, self._default_prior) + self._epsilon
            for v in candidates
        ], dtype=np.float64)

        total_score = np.sum(scores)
        if total_score <= 0.0 or not np.isfinite(total_score):
            probabilities = np.ones(len(candidates), dtype=np.float64) / len(candidates)
        else:
            probabilities = scores / total_score

        # Sample candidate according to degree-biased probability distribution
        chosen_idx = int(self._rng.choice(len(candidates), p=probabilities))
        chosen_destination = candidates[chosen_idx]

        action = SearchAction(
            action_type=ActionType.MOVE,
            destination=chosen_destination,
            metadata={
                "candidate_count": len(candidates),
                "chosen_observed_degree": self._observed_degrees.get(chosen_destination, self._default_prior),
                "candidate_scores": dict(zip(candidates, scores.tolist())),
            },
        )
        return self._validate_action(action, observation)

    def reset(self) -> None:
        """Clear local observed degree cache and re-seed PRNG."""
        self._observed_degrees.clear()
        self._rng = create_rng(self._seed)
