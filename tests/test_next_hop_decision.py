"""
Unit and Integration Tests for AESW Next-Hop Decision Engine
============================================================
Tests candidate scoring, numerically stable softmax selection,
multi-criteria tradeoff (evidence, novelty, revisit penalty, delay),
epistemic observation firewall, mode integration, and RNG isolation.
"""

from __future__ import annotations

import math
import random
import numpy as np
import pytest

from aesw.aesw.churn import ChurnEstimator
from aesw.aesw.communication import CommunicationMode, NodeMediatedExchange
from aesw.aesw.mode import AdaptiveModeController, SearchMode
from aesw.aesw.next_hop import (
    CandidateScore,
    CandidateScorer,
    NextHopDecision,
    NextHopDecisionEngine,
    SoftmaxSelector,
)
from aesw.baselines.actions import SearchAction
from aesw.baselines.types import ActionType
from aesw.environment.observation import (
    Observation,
    ObservedEdgeInfo,
    TargetSignalObservation,
)
from aesw.environment.types import EdgeState
from aesw.memory.cache import EvidenceCache
from aesw.memory.models import NodeEvidence
from aesw.memory.types import EvidencePolarity, EvidenceSource


def _create_mock_observation(
    walker_id: int | str = 0,
    current_node: int | str = 1,
    time: int = 10,
    neighbors: tuple[int | str, ...] = (2, 3, 4),
    active_neighbors: tuple[int | str, ...] = (2, 3),
) -> Observation:
    """Helper to synthesize a local observation snapshot with controllable link states."""
    observed_edges = {}
    for nbr in neighbors:
        state = EdgeState.ON if nbr in active_neighbors else EdgeState.OFF
        observed_edges[nbr] = ObservedEdgeInfo(
            neighbor_id=nbr,
            state=state,
            observed_at_time=time,
        )

    return Observation(
        walker_id=walker_id,
        current_node=current_node,
        time=time,
        checked_neighbors=neighbors,
        observed_edges=observed_edges,
        target_signal=TargetSignalObservation(
            detected=False,
            signal_strength=0.0,
            timestamp=time,
        ),
        neighbor_budget_used=len(neighbors),
    )


def _make_evidence(
    node_id: int | str,
    polarity: EvidencePolarity,
    confidence: float,
    timestamp: int,
    walker_id: int | str = 0,
) -> NodeEvidence:
    """Helper to synthesize valid NodeEvidence records."""
    is_pos = (polarity == EvidencePolarity.POSITIVE)
    return NodeEvidence(
        node_id=node_id,
        visited=True,
        target_found=is_pos,
        signal_strength=confidence if is_pos else 0.0,
        timestamp=timestamp,
        walker_id=walker_id,
        confidence=confidence,
        polarity=polarity,
        source=EvidenceSource.DIRECT_VISIT,
    )


# ==============================================================================
# 1. CANDIDATE SCORER WEIGHTS AND INITIALIZATION
# ==============================================================================


class TestCandidateScorerWeights:
    """Verifies weight validation and configuration on CandidateScorer."""

    def test_default_weights(self) -> None:
        scorer = CandidateScorer()
        assert scorer.weight_positive == 1.0
        assert scorer.weight_novelty == 1.0
        assert scorer.weight_revisit == 1.0
        assert scorer.weight_delay == 1.0

    def test_custom_positive_weights(self) -> None:
        scorer = CandidateScorer(
            weight_positive=2.5,
            weight_novelty=1.8,
            weight_revisit=0.5,
            weight_delay=3.0,
        )
        assert scorer.weight_positive == 2.5
        assert scorer.weight_novelty == 1.8
        assert scorer.weight_revisit == 0.5
        assert scorer.weight_delay == 3.0

    def test_zero_weights_allowed(self) -> None:
        scorer = CandidateScorer(
            weight_positive=0.0,
            weight_novelty=0.0,
            weight_revisit=0.0,
            weight_delay=0.0,
        )
        assert scorer.weight_positive == 0.0
        assert scorer.weight_novelty == 0.0

    def test_negative_weight_positive_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_positive must be non-negative"):
            CandidateScorer(weight_positive=-0.1)

    def test_negative_weight_novelty_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_novelty must be non-negative"):
            CandidateScorer(weight_novelty=-1.0)

    def test_negative_weight_revisit_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_revisit must be non-negative"):
            CandidateScorer(weight_revisit=-0.5)

    def test_negative_weight_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_delay must be non-negative"):
            CandidateScorer(weight_delay=-2.0)


# ==============================================================================
# 2. EVIDENCE TERM EVALUATION P(u)
# ==============================================================================


class TestPositiveEvidenceTerm:
    """Verifies P(u) extraction and decay under EvidenceCache and churn estimation."""

    def test_evidence_term_without_cache(self) -> None:
        scorer = CandidateScorer()
        p = scorer.compute_positive_evidence(node_id=2, evidence_cache=None, current_time=5, lambda_hat=0.1)
        assert p == 0.0

    def test_evidence_term_missing_node(self) -> None:
        scorer = CandidateScorer()
        cache = EvidenceCache()
        p = scorer.compute_positive_evidence(node_id=99, evidence_cache=cache, current_time=5, lambda_hat=0.1)
        assert p == 0.0

    def test_evidence_term_negative_evidence_returns_zero(self) -> None:
        """Cleared/negative evidence must return 0.0 for positive evidence P(u)."""
        scorer = CandidateScorer()
        cache = EvidenceCache()
        neg_ev = _make_evidence(
            node_id=2,
            polarity=EvidencePolarity.NEGATIVE,
            confidence=0.9,
            timestamp=5,
        )
        cache.store(neg_ev)

        p = scorer.compute_positive_evidence(node_id=2, evidence_cache=cache, current_time=5, lambda_hat=0.1)
        assert p == 0.0

    def test_evidence_term_fresh_positive_evidence(self) -> None:
        scorer = CandidateScorer()
        cache = EvidenceCache()
        pos_ev = _make_evidence(
            node_id=2,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.85,
            timestamp=10,
        )
        cache.store(pos_ev)

        # At time 10, age = 0 -> weight = 1.0 -> effective confidence = 0.85
        p = scorer.compute_positive_evidence(node_id=2, evidence_cache=cache, current_time=10, lambda_hat=0.2)
        assert math.isclose(p, 0.85, rel_tol=1e-5)

    def test_evidence_term_decay_with_age_and_churn(self) -> None:
        scorer = CandidateScorer()
        cache = EvidenceCache()
        pos_ev = _make_evidence(
            node_id=2,
            polarity=EvidencePolarity.POSITIVE,
            confidence=1.0,
            timestamp=0,
        )
        cache.store(pos_ev)

        # age = 5, lambda_hat = 0.2 -> exp(-0.2 * 5) = exp(-1.0) ~= 0.367879
        p = scorer.compute_positive_evidence(node_id=2, evidence_cache=cache, current_time=5, lambda_hat=0.2)
        expected = math.exp(-1.0)
        assert math.isclose(p, expected, rel_tol=1e-5)

    def test_evidence_term_zero_churn_preserves_confidence(self) -> None:
        scorer = CandidateScorer()
        cache = EvidenceCache()
        pos_ev = _make_evidence(
            node_id=3,
            polarity=EvidencePolarity.POSITIVE,
            confidence=0.75,
            timestamp=0,
        )
        cache.store(pos_ev)

        p = scorer.compute_positive_evidence(node_id=3, evidence_cache=cache, current_time=100, lambda_hat=0.0)
        assert math.isclose(p, 0.75, rel_tol=1e-5)


# ==============================================================================
# 3. NOVELTY AND REVISIT TERMS N(u) AND R(u)
# ==============================================================================


class TestNoveltyAndRevisitTerms:
    """Verifies novelty N(u) = 1/(1+v) and revisit penalty R(u) = v."""

    def test_novelty_unvisited_node(self) -> None:
        assert CandidateScorer.compute_novelty(0) == 1.0

    def test_novelty_visited_once(self) -> None:
        assert CandidateScorer.compute_novelty(1) == 0.5

    def test_novelty_monotonic_decrease(self) -> None:
        counts = [0, 1, 2, 5, 10, 50, 100]
        novelties = [CandidateScorer.compute_novelty(v) for v in counts]
        for i in range(len(novelties) - 1):
            assert novelties[i] > novelties[i + 1]
            assert 0.0 < novelties[i] <= 1.0

    def test_revisit_penalty_unvisited_node(self) -> None:
        assert CandidateScorer.compute_revisit_penalty(0) == 0.0

    def test_revisit_penalty_visited_nodes(self) -> None:
        assert CandidateScorer.compute_revisit_penalty(1) == 1.0
        assert CandidateScorer.compute_revisit_penalty(7) == 7.0

    def test_revisit_monotonic_increase(self) -> None:
        counts = [0, 1, 2, 5, 10, 50]
        penalties = [CandidateScorer.compute_revisit_penalty(v) for v in counts]
        for i in range(len(penalties) - 1):
            assert penalties[i] < penalties[i + 1]

    def test_negative_visit_count_handling(self) -> None:
        # Negative visit count is clamped to 0
        assert CandidateScorer.compute_novelty(-1) == 1.0
        assert CandidateScorer.compute_revisit_penalty(-1) == 0.0


# ==============================================================================
# 4. DELAY TERM D(u)
# ==============================================================================


class TestDelayTerm:
    """Verifies D(u) traversal delay calculation and validation."""

    def test_default_delay_is_one(self) -> None:
        assert CandidateScorer.compute_delay(2, delays=None) == 1.0
        assert CandidateScorer.compute_delay(2, delays={3: 2.0}) == 1.0

    def test_custom_delay_mapping(self) -> None:
        delays = {2: 2.5, 3: 4.0}
        assert CandidateScorer.compute_delay(2, delays=delays) == 2.5
        assert CandidateScorer.compute_delay(3, delays=delays) == 4.0

    def test_delay_below_one_clamped_or_enforced(self) -> None:
        # Delay specification enforces >= 1.0
        delays = {2: 0.5, 3: -1.0}
        assert CandidateScorer.compute_delay(2, delays=delays) == 1.0
        assert CandidateScorer.compute_delay(3, delays=delays) == 1.0


# ==============================================================================
# 5. TOTAL SCORE ARITHMETIC AND CANDIDATE SCORE DATACLASS
# ==============================================================================


class TestTotalScoreCalculation:
    """Verifies S(u) = a*P + b*N - g*R - d*D exact arithmetic and dataclass."""

    def test_exact_score_arithmetic(self) -> None:
        scorer = CandidateScorer(
            weight_positive=2.0,
            weight_novelty=3.0,
            weight_revisit=1.5,
            weight_delay=0.5,
        )
        cache = EvidenceCache()
        cache.store(
            _make_evidence(
                node_id=2,
                polarity=EvidencePolarity.POSITIVE,
                confidence=0.8,
                timestamp=0,
            )
        )
        # current_time=0, lambda_hat=0.0 -> P = 0.8
        # visit_count=2 -> N = 1/3, R = 2.0
        # delay=2.0 -> D = 2.0
        # S = 2.0 * 0.8 + 3.0 * (1/3) - 1.5 * 2.0 - 0.5 * 2.0
        # S = 1.6 + 1.0 - 3.0 - 1.0 = -1.4
        cand_score = scorer.score_candidate(
            node_id=2,
            evidence_cache=cache,
            current_time=0,
            lambda_hat=0.0,
            visit_count=2,
            traversal_delay=2.0,
        )

        assert isinstance(cand_score, CandidateScore)
        assert cand_score.node_id == 2
        assert math.isclose(cand_score.positive_evidence, 0.8, rel_tol=1e-5)
        assert math.isclose(cand_score.novelty, 1.0 / 3.0, rel_tol=1e-5)
        assert math.isclose(cand_score.revisit_penalty, 2.0, rel_tol=1e-5)
        assert math.isclose(cand_score.delay, 2.0, rel_tol=1e-5)
        assert math.isclose(cand_score.total_score, -1.4, rel_tol=1e-5)

    def test_candidate_score_to_dict(self) -> None:
        score = CandidateScore(
            node_id="A",
            positive_evidence=0.9,
            novelty=0.5,
            revisit_penalty=1.0,
            delay=1.0,
            total_score=0.4,
        )
        d = score.to_dict()
        assert d["node_id"] == "A"
        assert d["positive_evidence"] == 0.9
        assert d["total_score"] == 0.4

    def test_score_all_batch(self) -> None:
        scorer = CandidateScorer()
        scores = scorer.score_all(
            candidates=[1, 2, 3],
            evidence_cache=None,
            current_time=0,
            lambda_hat=0.0,
            visit_counts={1: 0, 2: 1, 3: 5},
        )
        assert set(scores.keys()) == {1, 2, 3}
        assert scores[1].novelty == 1.0
        assert scores[2].novelty == 0.5
        assert math.isclose(scores[3].novelty, 1.0 / 6.0)


# ==============================================================================
# 6. SOFTMAX SELECTOR NUMERICAL STABILITY & PROPERTIES
# ==============================================================================


class TestSoftmaxSelector:
    """Verifies stabilized softmax probabilities and temperature behavior."""

    def test_empty_scores_returns_empty_dict(self) -> None:
        probs = SoftmaxSelector.compute_probabilities({}, temperature=1.0)
        assert probs == {}

    def test_single_candidate_probability_is_one(self) -> None:
        score = CandidateScore(1, 0.0, 1.0, 0.0, 1.0, 0.0)
        probs = SoftmaxSelector.compute_probabilities({1: score}, temperature=1.0)
        assert probs == {1: 1.0}

    def test_equal_scores_yield_uniform_probabilities(self) -> None:
        scores = {
            i: CandidateScore(i, 0.0, 0.5, 1.0, 1.0, 2.5) for i in range(4)
        }
        probs = SoftmaxSelector.compute_probabilities(scores, temperature=1.0)
        for i in range(4):
            assert math.isclose(probs[i], 0.25, rel_tol=1e-5)

    def test_probabilities_sum_to_one(self) -> None:
        scores = {
            1: CandidateScore(1, 0.0, 1.0, 0.0, 1.0, 1.2),
            2: CandidateScore(2, 0.5, 0.5, 1.0, 1.0, -0.4),
            3: CandidateScore(3, 0.9, 0.2, 3.0, 1.0, 2.8),
        }
        probs = SoftmaxSelector.compute_probabilities(scores, temperature=1.5)
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)

    def test_higher_score_receives_higher_probability(self) -> None:
        scores = {
            1: CandidateScore(1, 0.0, 1.0, 0.0, 1.0, 1.0),
            2: CandidateScore(2, 0.0, 1.0, 0.0, 1.0, 3.0),
        }
        probs = SoftmaxSelector.compute_probabilities(scores, temperature=1.0)
        assert probs[2] > probs[1]

    def test_numerical_stability_extreme_positive_scores(self) -> None:
        """Massive scores (e.g. 10000.0) must not overflow with stabilized softmax."""
        scores = {
            1: CandidateScore(1, 0.0, 1.0, 0.0, 1.0, 10000.0),
            2: CandidateScore(2, 0.0, 1.0, 0.0, 1.0, 10001.0),
            3: CandidateScore(3, 0.0, 1.0, 0.0, 1.0, 10002.0),
        }
        probs = SoftmaxSelector.compute_probabilities(scores, temperature=1.0)
        assert all(not math.isnan(p) and not math.isinf(p) for p in probs.values())
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)
        assert probs[3] > probs[2] > probs[1]

    def test_numerical_stability_extreme_negative_scores(self) -> None:
        """Massive negative scores must not underflow into NaNs."""
        scores = {
            1: CandidateScore(1, 0.0, 1.0, 0.0, 1.0, -5000.0),
            2: CandidateScore(2, 0.0, 1.0, 0.0, 1.0, -4999.0),
        }
        probs = SoftmaxSelector.compute_probabilities(scores, temperature=1.0)
        assert all(not math.isnan(p) for p in probs.values())
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6)
        assert probs[2] > probs[1]

    def test_temperature_validation_rejects_non_positive(self) -> None:
        scores = {1: CandidateScore(1, 0.0, 1.0, 0.0, 1.0, 1.0)}
        with pytest.raises(ValueError, match="strictly positive"):
            SoftmaxSelector.compute_probabilities(scores, temperature=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            SoftmaxSelector.compute_probabilities(scores, temperature=-0.5)

    def test_high_temperature_approaches_uniform(self) -> None:
        scores = {
            1: CandidateScore(1, 0.0, 1.0, 0.0, 1.0, 1.0),
            2: CandidateScore(2, 0.0, 1.0, 0.0, 1.0, 5.0),
        }
        probs = SoftmaxSelector.compute_probabilities(scores, temperature=100.0)
        assert math.isclose(probs[1], 0.5, abs_tol=0.05)
        assert math.isclose(probs[2], 0.5, abs_tol=0.05)

    def test_low_temperature_approaches_greedy(self) -> None:
        scores = {
            1: CandidateScore(1, 0.0, 1.0, 0.0, 1.0, 1.0),
            2: CandidateScore(2, 0.0, 1.0, 0.0, 1.0, 2.0),
        }
        probs = SoftmaxSelector.compute_probabilities(scores, temperature=0.01)
        assert probs[2] > 0.999
        assert probs[1] < 0.001


# ==============================================================================
# 7. STOCHASTIC SAMPLING & RNG ISOLATION
# ==============================================================================


class TestStochasticSamplingAndRNG:
    """Verifies reproducible selection, distribution convergence, and RNG isolation."""

    def test_empty_probabilities_returns_none(self) -> None:
        chosen, prob = SoftmaxSelector.select_candidate({})
        assert chosen is None
        assert prob == 0.0

    def test_single_candidate_deterministic_selection(self) -> None:
        chosen, prob = SoftmaxSelector.select_candidate({42: 1.0})
        assert chosen == 42
        assert prob == 1.0

    def test_deterministic_reproducibility_with_seed(self) -> None:
        probs = {1: 0.2, 2: 0.5, 3: 0.3}
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)

        seq1 = [SoftmaxSelector.select_candidate(probs, rng=rng1)[0] for _ in range(50)]
        seq2 = [SoftmaxSelector.select_candidate(probs, rng=rng2)[0] for _ in range(50)]

        assert seq1 == seq2

    def test_rng_isolation_does_not_affect_python_random_or_global_numpy(self) -> None:
        """Supplied isolated generator must not alter python random or global np.random state."""
        random.seed(42)
        r_before = random.random()

        np.random.seed(42)
        np_before = np.random.rand()

        # Reset states
        random.seed(42)
        np.random.seed(42)

        # Run selection using isolated generator
        isolated_rng = np.random.default_rng(9999)
        SoftmaxSelector.select_candidate({1: 0.5, 2: 0.5}, rng=isolated_rng)

        r_after = random.random()
        np_after = np.random.rand()

        assert r_before == r_after
        assert np_before == np_after

    def test_empirical_frequency_converges_to_softmax_probabilities(self) -> None:
        probs = {1: 0.7, 2: 0.3}
        rng = np.random.default_rng(42)
        n_trials = 2000

        counts = {1: 0, 2: 0}
        for _ in range(n_trials):
            chosen, _ = SoftmaxSelector.select_candidate(probs, rng=rng)
            counts[chosen] += 1

        freq1 = counts[1] / n_trials
        assert math.isclose(freq1, 0.7, abs_tol=0.03)


# ==============================================================================
# 8. NEXT-HOP DECISION ENGINE & OBSERVATION INTEGRATION
# ==============================================================================


class TestNextHopDecisionEngine:
    """Verifies candidate extraction from observation and structured decision output."""

    def test_engine_initialization_defaults(self) -> None:
        engine = NextHopDecisionEngine()
        assert engine.temperature == 1.0
        assert engine.scorer.weight_positive == 1.0

    def test_engine_temperature_validation(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            NextHopDecisionEngine(temperature=-1.0)

    def test_extract_legal_candidates_filters_off_edges(self) -> None:
        """Legal candidates must strictly include checked neighbors whose link is ON."""
        obs = _create_mock_observation(
            current_node=1,
            neighbors=(2, 3, 4),
            active_neighbors=(2, 4),  # 3 is OFF
        )
        engine = NextHopDecisionEngine()
        candidates = engine.extract_legal_candidates(obs)
        assert candidates == [2, 4]
        assert 3 not in candidates

    def test_empty_candidates_returns_stay_action(self) -> None:
        obs = _create_mock_observation(
            current_node=1,
            neighbors=(2, 3),
            active_neighbors=(),  # all OFF
        )
        engine = NextHopDecisionEngine()
        decision = engine.decide(
            mode=SearchMode.LOCAL,
            observation=obs,
        )

        assert decision.selected_node is None
        assert decision.candidates == ()
        assert decision.scores == {}
        assert decision.probabilities == {}
        assert decision.selected_probability == 0.0
        assert decision.action.action_type == ActionType.STAY
        assert decision.action.destination == 1

    def test_single_legal_candidate_selects_deterministically(self) -> None:
        obs = _create_mock_observation(
            current_node=1,
            neighbors=(2, 3),
            active_neighbors=(2,),  # only 2 is ON
        )
        engine = NextHopDecisionEngine()
        decision = engine.decide(
            mode=SearchMode.LOCAL,
            observation=obs,
        )

        assert decision.selected_node == 2
        assert decision.candidates == (2,)
        assert decision.selected_probability == 1.0
        assert decision.action.action_type == ActionType.MOVE
        assert decision.action.destination == 2

    def test_mode_recorded_in_decision_and_action_metadata(self) -> None:
        engine = NextHopDecisionEngine()
        dec_local = engine.decide(
            mode=SearchMode.LOCAL,
            candidates=[2, 3],
        )
        assert dec_local.mode == SearchMode.LOCAL
        assert dec_local.action.metadata["mode"] == "LOCAL"

        dec_jump = engine.decide(
            mode=SearchMode.LONG_JUMP,
            candidates=[2, 3],
        )
        assert dec_jump.mode == SearchMode.LONG_JUMP
        assert dec_jump.action.metadata["mode"] == "LONG_JUMP"

    def test_decision_to_dict_structure(self) -> None:
        engine = NextHopDecisionEngine()
        decision = engine.decide(
            mode=SearchMode.LOCAL,
            candidates=[2, 3],
            rng=np.random.default_rng(42),
        )
        d = decision.to_dict()
        assert d["mode"] == "LOCAL"
        assert d["selected_node"] in [2, 3]
        assert "scores" in d
        assert "probabilities" in d
        assert "action" in d
        assert d["action"]["action_type"].upper() == "MOVE"


# ==============================================================================
# 9. SEARCH MODE WEIGHT SCALING INTEGRATION
# ==============================================================================


class TestModeWeightScaling:
    """Verifies that LONG_JUMP mode adjusts weights toward exploration when enabled."""

    def test_long_jump_weight_scaling_boosts_novelty_and_reduces_revisit(self) -> None:
        engine = NextHopDecisionEngine(
            weight_positive=1.0,
            weight_novelty=1.0,
            weight_revisit=1.0,
            mode_weight_scaling=True,
        )
        # In LOCAL mode:
        scores_local = engine.decide(
            mode=SearchMode.LOCAL,
            candidates=[1, 2],
            visit_counts={1: 0, 2: 5},
        ).scores

        # In LONG_JUMP mode: novelty boosted (x1.5), revisit penalty dampened (x0.5)
        scores_jump = engine.decide(
            mode=SearchMode.LONG_JUMP,
            candidates=[1, 2],
            visit_counts={1: 0, 2: 5},
        ).scores

        # Candidate 1 is unvisited: N=1.0, R=0
        # In LOCAL: total = 1.0 * 1.0 - 1.0 = 0.0 (D=1)
        # In JUMP: total = 1.5 * 1.0 - 1.0 = 0.5
        assert scores_jump[1].total_score > scores_local[1].total_score


# ==============================================================================
# 10. SCIENTIFIC BEHAVIOR SCENARIOS
# ==============================================================================


class TestScientificBehaviorScenarios:
    """Validates the core research hypotheses for next-hop selection."""

    def test_scenario_a_fresh_positive_evidence_attracts_walker(self) -> None:
        """Scenario A: A neighbor with fresh positive target evidence is prioritized

        over an unvisited neighbor when evidence weight is dominant.
        """
        engine = NextHopDecisionEngine(
            weight_positive=3.0,
            weight_novelty=1.0,
            weight_revisit=1.0,
            weight_delay=1.0,
            temperature=0.5,
        )
        cache = EvidenceCache()
        # Neighbor 2 has strong fresh positive target clue
        cache.store(
            _make_evidence(
                node_id=2,
                polarity=EvidencePolarity.POSITIVE,
                confidence=0.9,
                timestamp=10,
            )
        )
        # Candidate 2: visited once (N=0.5, R=1.0), but has fresh clue (P=0.9)
        # Candidate 3: unvisited (N=1.0, R=0.0), no clue (P=0.0)
        decision = engine.decide(
            mode=SearchMode.LOCAL,
            candidates=[2, 3],
            evidence_cache=cache,
            current_time=10,
            lambda_hat=0.05,
            visit_counts={2: 1, 3: 0},
        )

        score_2 = decision.scores[2].total_score
        score_3 = decision.scores[3].total_score

        # Score 2: 3.0*0.9 + 1.0*0.5 - 1.0*1.0 - 1.0*1.0 = 2.7 + 0.5 - 2.0 = 1.2
        # Score 3: 3.0*0.0 + 1.0*1.0 - 1.0*0.0 - 1.0*1.0 = 1.0 - 1.0 = 0.0
        assert score_2 > score_3
        assert decision.probabilities[2] > decision.probabilities[3]

    def test_scenario_b_high_churn_decays_stale_clue_allowing_novelty_to_win(self) -> None:
        """Scenario B: In a high-churn volatile environment, an old clue decays heavily.

        The walker abandons the stale clue in favor of fresh novelty.
        """
        engine = NextHopDecisionEngine(
            weight_positive=2.0,
            weight_novelty=2.0,
            weight_revisit=1.0,
            weight_delay=1.0,
            temperature=0.5,
        )
        cache = EvidenceCache()
        # Clue recorded at t=0
        cache.store(
            _make_evidence(
                node_id=2,
                polarity=EvidencePolarity.POSITIVE,
                confidence=0.8,
                timestamp=0,
            )
        )

        # Under high churn (lambda_hat = 1.0) and age = 10:
        # P(2) = 0.8 * exp(-1.0 * 10) ~= 0.8 * 4.5e-5 ~= 0.0
        # Candidate 2: visited 2 times (N = 1/3, R = 2.0)
        # Candidate 3: unvisited (N = 1.0, R = 0.0)
        decision = engine.decide(
            mode=SearchMode.LOCAL,
            candidates=[2, 3],
            evidence_cache=cache,
            current_time=10,
            lambda_hat=1.0,
            visit_counts={2: 2, 3: 0},
        )

        score_2 = decision.scores[2].total_score
        score_3 = decision.scores[3].total_score

        # The stale clue has decayed to ~0, so unvisited candidate 3 easily wins
        assert score_3 > score_2
        assert decision.probabilities[3] > 0.8

    def test_scenario_c_revisit_penalty_avoids_backtracking(self) -> None:
        """Scenario C: When two nodes have equal zero evidence and equal delays,

        a frequently visited node (e.g. previous node) is strongly avoided.
        """
        engine = NextHopDecisionEngine(
            weight_positive=1.0,
            weight_novelty=1.0,
            weight_revisit=2.0,
            weight_delay=1.0,
            temperature=1.0,
        )

        # Candidate 1: previous node visited 4 times
        # Candidate 2: unvisited node (0 visits)
        decision = engine.decide(
            mode=SearchMode.LOCAL,
            candidates=[1, 2],
            visit_counts={1: 4, 2: 0},
        )

        score_1 = decision.scores[1].total_score
        score_2 = decision.scores[2].total_score

        # Candidate 1: N=0.2, R=4.0 -> Score = 0.2 - 2.0*4.0 - 1.0 = -8.8
        # Candidate 2: N=1.0, R=0.0 -> Score = 1.0 - 0.0 - 1.0 = 0.0
        assert score_2 > score_1
        assert decision.probabilities[2] > 0.999


# ==============================================================================
# 11. END-TO-END SUBSYSTEM INTEGRATION
# ==============================================================================


class TestSubsystemIntegration:
    """Verifies seamless data flow across Evidence Memory, Churn Estimator,
    Node-Mediated Sharing, Mode Controller, and Next-Hop Decision Engine.
    """

    def test_full_chain_execution(self) -> None:
        # 1. Churn Estimator tracks local observations
        churn_estimator = ChurnEstimator(eta=0.3)
        obs_t0 = _create_mock_observation(current_node=1, time=0, neighbors=(2, 3), active_neighbors=(2, 3))
        obs_t1 = _create_mock_observation(current_node=1, time=1, neighbors=(2, 3), active_neighbors=(2,))
        churn_estimator.update_from_observation(obs_t0)
        churn_estimator.update_from_observation(obs_t1)
        lambda_hat = churn_estimator.lambda_hat

        # 2. Evidence Cache stores local clue
        local_cache = EvidenceCache()
        local_cache.store(
            _make_evidence(
                node_id=2,
                polarity=EvidencePolarity.POSITIVE,
                confidence=0.8,
                timestamp=1,
            )
        )

        # 3. Node-Mediated Exchange exchanges clues with shared cache
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        shared_cache = exchange.get_node_cache(node_id=1)
        shared_cache.store(
            _make_evidence(
                node_id=3,
                polarity=EvidencePolarity.NEGATIVE,
                confidence=0.9,
                timestamp=1,
                walker_id="walker_99",
            )
        )
        exchange.pull(
            walker_id=0,
            at_node=1,
            timestamp=1,
            target_cache=local_cache,
        )

        # 4. Mode Controller determines search mode
        mode_controller = AdaptiveModeController(window_size=3)
        mode = mode_controller.update(timestamp=1, evidence=obs_t1)
        assert mode == SearchMode.LOCAL

        # 5. Next-Hop Decision Engine selects action
        decision_engine = NextHopDecisionEngine(temperature=1.0)
        decision = decision_engine.decide(
            mode=mode,
            observation=obs_t1,
            evidence_cache=local_cache,
            current_time=2,
            lambda_hat=lambda_hat,
            visit_counts={2: 0},
            rng=np.random.default_rng(100),
        )

        assert decision.mode == SearchMode.LOCAL
        assert decision.selected_node == 2
        assert decision.action.action_type == ActionType.MOVE
        assert decision.action.destination == 2
        assert decision.lambda_hat == lambda_hat
