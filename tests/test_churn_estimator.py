"""
Unit and Integration Tests for AESW Churn Estimator
===================================================
Tests online estimation of dynamic graph volatility (lambda_hat) from locally
observed neighborhood overlap under strict partial observability constraints.
"""

from __future__ import annotations

import math
import pytest

from aesw.aesw.churn import ChurnEstimator, DEFAULT_ETA, DEFAULT_EPSILON
from aesw.environment.observation import Observation, ObservedEdgeInfo, TargetSignalObservation
from aesw.environment.types import EdgeState
from aesw.memory.cache import EvidenceCache
from aesw.memory.decay import exponential_decay
from aesw.memory.models import NodeEvidence


def _create_test_observation(
    walker_id: int | str,
    current_node: int | str,
    time: int,
    checked_neighbors: tuple[int | str, ...],
    active_neighbors: set[int | str],
) -> Observation:
    """Helper to synthesize a valid local Observation snapshot."""
    observed_edges = {
        nbr: ObservedEdgeInfo(
            neighbor_id=nbr,
            state=EdgeState.ON if nbr in active_neighbors else EdgeState.OFF,
            observed_at_time=time,
        )
        for nbr in checked_neighbors
    }
    return Observation(
        walker_id=walker_id,
        current_node=current_node,
        time=time,
        checked_neighbors=checked_neighbors,
        observed_edges=observed_edges,
        target_signal=TargetSignalObservation(detected=False, signal_strength=0.0),
        neighbor_budget_used=len(checked_neighbors),
    )


class TestChurnEstimatorCore:
    """Core mathematical and state validation tests for ChurnEstimator."""

    def test_01_first_observation_initializes_state(self) -> None:
        """Test 1: First observation initializes state without manufacturing churn."""
        estimator = ChurnEstimator(initial_lambda=0.0)

        # Pre-update state
        assert estimator.update_count == 0
        assert estimator.current_neighborhood is None
        assert estimator.previous_neighborhood is None
        assert estimator.current_timestamp is None
        assert estimator.previous_timestamp is None
        assert estimator.overlap is None
        assert estimator.raw_churn_rate is None
        assert estimator.estimated_lambda == 0.0

        # First update
        result = estimator.update({1, 2, 3}, timestamp=0)

        assert estimator.update_count == 1
        assert estimator.current_neighborhood == frozenset({1, 2, 3})
        assert estimator.previous_neighborhood is None
        assert estimator.current_timestamp == 0.0
        assert estimator.previous_timestamp is None
        assert estimator.overlap is None
        assert estimator.raw_churn_rate is None
        assert estimator.estimated_lambda == 0.0
        assert result == 0.0

    def test_02_identical_neighborhoods_produce_zero_raw_churn(self) -> None:
        """Test 2: Identical neighborhoods produce overlap=1 and raw churn=0."""
        estimator = ChurnEstimator(eta=0.5, initial_lambda=0.5)

        estimator.update({10, 20}, timestamp=1)
        res = estimator.update({10, 20}, timestamp=2)

        assert estimator.overlap == 1.0
        assert estimator.raw_churn_rate == 0.0
        assert estimator.previous_neighborhood == frozenset({10, 20})
        assert estimator.current_neighborhood == frozenset({10, 20})
        assert estimator.previous_timestamp == 1.0
        assert estimator.current_timestamp == 2.0
        # (1 - 0.5) * 0.5 + 0.5 * 0.0 = 0.25
        assert math.isclose(res, 0.25, abs_tol=1e-9)

    def test_03_stable_observations_drive_lambda_toward_zero(self) -> None:
        """Test 3: Stable repeated observations drive estimated lambda toward zero."""
        estimator = ChurnEstimator(eta=0.2, initial_lambda=1.0)
        estimator.update({"A", "B"}, timestamp=0)

        lambdas = []
        for t in range(1, 25):
            val = estimator.update({"A", "B"}, timestamp=t)
            lambdas.append(val)

        # Monotonically non-increasing and converging toward 0
        for i in range(len(lambdas) - 1):
            assert lambdas[i] >= lambdas[i + 1]

        assert lambdas[-1] < 0.01
        assert estimator.estimated_lambda >= 0.0

    def test_04_partial_neighborhood_changes_produce_positive_churn(self) -> None:
        """Test 4: Partial neighborhood changes produce 0 < overlap < 1 and positive raw churn."""
        estimator = ChurnEstimator(eta=0.5)
        estimator.update({1, 2, 3, 4}, timestamp=0)

        # 2 elements in common (3, 4), 2 dropped (1, 2), 2 added (5, 6)
        # Intersection = {3, 4} (size 2), Union = {1, 2, 3, 4, 5, 6} (size 6)
        # Jaccard overlap = 2/6 = 1/3
        estimator.update({3, 4, 5, 6}, timestamp=2)

        assert math.isclose(estimator.overlap, 1.0 / 3.0, abs_tol=1e-9)
        expected_raw = -math.log(1.0 / 3.0) / 2.0
        assert math.isclose(estimator.raw_churn_rate, expected_raw, abs_tol=1e-9)
        assert estimator.raw_churn_rate > 0.0
        assert estimator.estimated_lambda > 0.0

    def test_05_completely_different_neighborhoods_remain_finite(self) -> None:
        """Test 5: Completely disjoint neighborhoods produce overlap=0, finite, non-negative estimate."""
        epsilon = 1e-4
        estimator = ChurnEstimator(eta=1.0, epsilon=epsilon)
        estimator.update({"A", "B"}, timestamp=1)

        estimator.update({"C", "D"}, timestamp=2)

        assert estimator.overlap == 0.0
        expected_raw = -math.log(epsilon) / 1.0
        assert math.isclose(estimator.raw_churn_rate, expected_raw, abs_tol=1e-9)
        assert math.isfinite(estimator.estimated_lambda)
        assert estimator.estimated_lambda >= 0.0

    def test_06_empty_to_empty_produces_zero_churn(self) -> None:
        """Test 6: Empty -> empty produces overlap=1 and raw churn=0."""
        estimator = ChurnEstimator(eta=0.5, initial_lambda=0.4)
        estimator.update([], timestamp=1)
        estimator.update([], timestamp=3)

        assert estimator.overlap == 1.0
        assert estimator.raw_churn_rate == 0.0
        # (1 - 0.5) * 0.4 + 0.5 * 0.0 = 0.2
        assert math.isclose(estimator.estimated_lambda, 0.2, abs_tol=1e-9)

    def test_07_empty_to_non_empty_produces_finite_positive_churn(self) -> None:
        """Test 7: Empty -> non-empty produces finite positive churn."""
        estimator = ChurnEstimator(eta=0.5, epsilon=1e-3)
        estimator.update([], timestamp=0)
        estimator.update([1, 2], timestamp=1)

        assert estimator.overlap == 0.0
        assert estimator.raw_churn_rate > 0.0
        assert math.isfinite(estimator.raw_churn_rate)
        assert estimator.estimated_lambda > 0.0
        assert math.isfinite(estimator.estimated_lambda)

    def test_08_non_empty_to_empty_produces_finite_positive_churn(self) -> None:
        """Test 8: Non-empty -> empty produces finite positive churn."""
        estimator = ChurnEstimator(eta=0.5, epsilon=1e-3)
        estimator.update([1, 2], timestamp=0)
        estimator.update([], timestamp=1)

        assert estimator.overlap == 0.0
        assert estimator.raw_churn_rate > 0.0
        assert math.isfinite(estimator.raw_churn_rate)
        assert estimator.estimated_lambda > 0.0
        assert math.isfinite(estimator.estimated_lambda)

    def test_09_delta_t_affects_estimate_inversely(self) -> None:
        """Test 9: Delta_t inversely scales the raw churn rate."""
        # Case 1: Delta_t = 1
        est1 = ChurnEstimator(eta=1.0)
        est1.update([1, 2], timestamp=0)
        est1.update([2, 3], timestamp=1)  # overlap = 1/3

        # Case 2: Delta_t = 2
        est2 = ChurnEstimator(eta=1.0)
        est2.update([1, 2], timestamp=0)
        est2.update([2, 3], timestamp=2)  # overlap = 1/3

        assert math.isclose(est1.overlap, est2.overlap, abs_tol=1e-9)
        assert math.isclose(est1.raw_churn_rate, 2.0 * est2.raw_churn_rate, abs_tol=1e-9)
        assert math.isclose(est1.estimated_lambda, 2.0 * est2.estimated_lambda, abs_tol=1e-9)

    def test_10_eta_one_completely_trusts_latest_raw(self) -> None:
        """Test 10: eta=1 sets estimated lambda directly equal to raw churn."""
        estimator = ChurnEstimator(eta=1.0, initial_lambda=99.0)
        estimator.update([1, 2], timestamp=0)
        estimator.update([2, 3], timestamp=1)

        assert estimator.estimated_lambda == estimator.raw_churn_rate
        assert not math.isclose(estimator.estimated_lambda, 99.0)

    def test_11_eta_zero_preserves_previous_lambda(self) -> None:
        """Test 11: eta=0 preserves previous lambda regardless of observations."""
        estimator = ChurnEstimator(eta=0.0, initial_lambda=0.75)
        estimator.update([1, 2], timestamp=0)
        estimator.update([3, 4], timestamp=1)  # complete churn!

        assert estimator.raw_churn_rate > 0.0
        assert estimator.estimated_lambda == 0.75

    def test_12_invalid_eta_is_rejected(self) -> None:
        """Test 12: Invalid eta outside [0, 1] is rejected with ValueError."""
        with pytest.raises(ValueError, match="eta"):
            ChurnEstimator(eta=-0.01)
        with pytest.raises(ValueError, match="eta"):
            ChurnEstimator(eta=1.01)

    def test_13_invalid_timestamps_rejected(self) -> None:
        """Test 13: Invalid or non-increasing timestamps are rejected."""
        estimator = ChurnEstimator()

        # Negative timestamp on first update
        with pytest.raises(ValueError, match="non-negative"):
            estimator.update([1, 2], timestamp=-1)

        estimator.update([1, 2], timestamp=5)

        # Non-increasing timestamp: equal (Delta_t = 0)
        with pytest.raises(ValueError, match="Delta_t must be positive"):
            estimator.update([1, 2], timestamp=5)

        # Decreasing timestamp (Delta_t < 0)
        with pytest.raises(ValueError, match="Delta_t must be positive"):
            estimator.update([1, 2], timestamp=4)

        # Missing timestamp when passing raw iterable
        with pytest.raises(ValueError, match="timestamp must be provided"):
            estimator.update([1, 2], timestamp=None)

    def test_14_estimated_lambda_always_non_negative(self) -> None:
        """Test 14: Estimated lambda is guaranteed >= 0 under all valid updates."""
        estimator = ChurnEstimator(eta=0.5, initial_lambda=0.0)
        sequences = [
            ([1, 2, 3], 1),
            ([1, 2, 3], 2),
            ([1, 2], 3),
            ([2, 3, 4, 5], 4),
            ([], 5),
            ([], 6),
            ([10], 7),
        ]
        for nbrs, t in sequences:
            val = estimator.update(nbrs, timestamp=t)
            assert val >= 0.0
            assert estimator.estimated_lambda >= 0.0

    def test_15_estimated_lambda_always_finite(self) -> None:
        """Test 15: Estimated lambda is guaranteed finite under disjoint, empty, or large shifts."""
        estimator = ChurnEstimator(eta=0.9, epsilon=1e-6)
        disjoint_seq = [
            ([1], 1),
            ([2], 2),
            ([3], 3),
            ([], 4),
            ([100, 200], 5),
        ]
        for nbrs, t in disjoint_seq:
            val = estimator.update(nbrs, timestamp=t)
            assert math.isfinite(val)
            assert math.isfinite(estimator.estimated_lambda)

    def test_16_deterministic_behavior(self) -> None:
        """Test 16: Identical observations at identical timestamps produce identical lambda sequences."""
        est_a = ChurnEstimator(eta=0.3, initial_lambda=0.1)
        est_b = ChurnEstimator(eta=0.3, initial_lambda=0.1)

        updates = [
            ([1, 2], 0),
            ([2, 3], 1),
            ([2, 3], 3),
            ([3, 4, 5], 6),
            ([], 10),
        ]
        results_a = [est_a.update(n, t) for n, t in updates]
        results_b = [est_b.update(n, t) for n, t in updates]

        assert results_a == results_b
        assert est_a.to_dict() == est_b.to_dict()

    def test_17_multiple_estimators_do_not_share_state(self) -> None:
        """Test 17: Independent estimator instances (for multiple walkers) maintain private state."""
        walker1_estimator = ChurnEstimator(eta=0.5, initial_lambda=0.0)
        walker2_estimator = ChurnEstimator(eta=0.5, initial_lambda=0.0)

        # Walker 1 observes a stable static environment
        walker1_estimator.update({1, 2}, timestamp=0)
        walker1_estimator.update({1, 2}, timestamp=1)
        walker1_estimator.update({1, 2}, timestamp=2)

        # Walker 2 observes high churn
        walker2_estimator.update({10, 20}, timestamp=0)
        walker2_estimator.update({30, 40}, timestamp=1)
        walker2_estimator.update({50, 60}, timestamp=2)

        assert walker1_estimator.estimated_lambda == 0.0
        assert walker2_estimator.estimated_lambda > 0.0
        assert walker1_estimator.current_neighborhood != walker2_estimator.current_neighborhood
        assert walker1_estimator.update_count == walker2_estimator.update_count == 3

    def test_18_information_boundary_verification(self) -> None:
        """Test 18: Strict epistemic firewall: estimator has no access to hidden dynamics or true graph."""
        estimator = ChurnEstimator()

        forbidden_attributes = [
            "p_on",
            "p_off",
            "true_graph",
            "graph",
            "ground_truth",
            "target_state",
            "future_state",
            "active_graph_view",
        ]
        for attr in forbidden_attributes:
            assert not hasattr(estimator, attr), f"Estimator leaks forbidden attribute {attr}"

        # Ensure estimator update strictly accepts local inputs
        obs = _create_test_observation(
            walker_id="w1",
            current_node=1,
            time=0,
            checked_neighbors=(2, 3),
            active_neighbors={2},
        )
        rate = estimator.update(obs)
        assert rate >= 0.0
        assert estimator.current_neighborhood == frozenset({2})

    def test_19_m8_evidence_memory_integration(self) -> None:
        """Test 19: Estimator output directly parameterizes M8 EvidenceCache decay evaluations."""
        estimator = ChurnEstimator(eta=0.5)
        cache = EvidenceCache()

        # Gather evidence at t=0
        evidence = NodeEvidence(
            node_id=42,
            visited=True,
            target_found=True,
            signal_strength=1.0,
            timestamp=0,
            walker_id="w1",
            confidence=1.0,
        )
        cache.store(evidence)

        # Walker observes high volatility between t=0 and t=5
        estimator.update({1, 2}, timestamp=0)
        estimator.update({3, 4}, timestamp=5)

        estimated_lambda = estimator.estimated_lambda
        assert estimated_lambda > 0.0

        # Query cache passing estimator float
        decayed_1 = cache.get(node_id=42, current_time=10, lambda_hat=estimated_lambda)
        assert decayed_1 is not None
        assert 0.0 < decayed_1.weight < 1.0

        # Query cache passing ChurnEstimator directly
        decayed_2 = cache.get(node_id=42, current_time=10, lambda_hat=estimator)
        assert decayed_2 is not None
        assert math.isclose(decayed_1.weight, decayed_2.weight, abs_tol=1e-9)

        # Decay matches formal mathematical decay model
        expected_weight = exponential_decay(age=10, lambda_hat=estimated_lambda)
        assert math.isclose(decayed_1.weight, expected_weight, abs_tol=1e-9)


class TestScientificBehavior:
    """Scientific validation linking M9 volatility estimation to M8 adaptive memory decay."""

    def test_scientific_volatility_to_memory_decay_coupling(self) -> None:
        """Verify: Low churn -> slow decay; High churn -> fast decay.

        Scientific Hypothesis:
        Under a stable topology, observed neighborhood overlap is high, resulting
        in a low churn estimate lambda_hat. Consequently, evidence in memory decays
        slowly (w ≈ 1.0). Under a volatile topology, observed overlap is low,
        driving lambda_hat up, which rapidly attenuates evidence weights.
        """
        stable_estimator = ChurnEstimator(eta=0.5, initial_lambda=0.0)
        volatile_estimator = ChurnEstimator(eta=0.5, initial_lambda=0.0)

        cache_stable = EvidenceCache()
        cache_volatile = EvidenceCache()

        # Both record evidence at t=0
        ev_stable = NodeEvidence(
            node_id="target_cell",
            visited=True,
            target_found=True,
            signal_strength=1.0,
            timestamp=0,
            walker_id="w_stable",
            confidence=1.0,
        )
        ev_volatile = NodeEvidence(
            node_id="target_cell",
            visited=True,
            target_found=True,
            signal_strength=1.0,
            timestamp=0,
            walker_id="w_volatile",
            confidence=1.0,
        )
        cache_stable.store(ev_stable)
        cache_volatile.store(ev_volatile)

        # Environment evolution from t=0 to t=10
        # Stable walker: observes the same neighborhood repeatedly
        stable_estimator.update({1, 2, 3, 4}, timestamp=0)
        for t in range(1, 11):
            stable_estimator.update({1, 2, 3, 4}, timestamp=t)

        # Volatile walker: observes frequent topological disruption
        volatile_estimator.update({1, 2, 3, 4}, timestamp=0)
        volatile_estimator.update({1, 5, 6, 7}, timestamp=2)
        volatile_estimator.update({8, 9, 10, 11}, timestamp=5)
        volatile_estimator.update({12, 13}, timestamp=8)
        volatile_estimator.update({14, 15, 16}, timestamp=10)

        # Estimated volatility comparison
        lambda_stable = stable_estimator.estimated_lambda
        lambda_volatile = volatile_estimator.estimated_lambda

        assert lambda_stable == 0.0
        assert lambda_volatile > 0.3
        assert lambda_volatile > lambda_stable

        # Evaluate evidence at current_time = 15 (age = 15)
        w_stable = cache_stable.get(
            node_id="target_cell",
            current_time=15,
            lambda_hat=stable_estimator,
        )
        w_volatile = cache_volatile.get(
            node_id="target_cell",
            current_time=15,
            lambda_hat=volatile_estimator,
        )

        assert w_stable is not None
        assert w_volatile is not None

        # Evidence weight under stable conditions persists strongly
        assert math.isclose(w_stable.weight, 1.0, abs_tol=1e-9)

        # Evidence weight under volatile conditions decays significantly
        assert w_volatile.weight < 0.1
        assert w_stable.weight > w_volatile.weight


class TestAuxiliaryFeatures:
    """Tests for edge cases, helper utilities, and defensive guarantees."""

    def test_observation_integration_with_on_off_filtering(self) -> None:
        """Verify update_from_observation correctly filters ON vs OFF inspected edges."""
        estimator = ChurnEstimator()

        obs1 = _create_test_observation(
            walker_id="w1",
            current_node=1,
            time=1,
            checked_neighbors=(2, 3, 4),
            active_neighbors={2, 3},  # edge 4 is OFF
        )
        estimator.update_from_observation(obs1)
        assert estimator.current_neighborhood == frozenset({2, 3})

        obs2 = _create_test_observation(
            walker_id="w1",
            current_node=1,
            time=2,
            checked_neighbors=(2, 3, 4),
            active_neighbors={3, 4},  # edge 2 became OFF, 4 became ON
        )
        estimator.update_from_observation(obs2)
        assert estimator.current_neighborhood == frozenset({3, 4})
        assert estimator.previous_neighborhood == frozenset({2, 3})
        # Intersection = {3} (1), Union = {2, 3, 4} (3) -> overlap = 1/3
        assert math.isclose(estimator.overlap, 1.0 / 3.0, abs_tol=1e-9)

    def test_parameter_validation(self) -> None:
        """Verify defensive initialization validation."""
        with pytest.raises(ValueError, match="epsilon"):
            ChurnEstimator(epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon"):
            ChurnEstimator(epsilon=-1e-4)
        with pytest.raises(ValueError, match="initial_lambda"):
            ChurnEstimator(initial_lambda=-0.5)

    def test_reset_behavior(self) -> None:
        """Verify reset clears history and resets lambda."""
        estimator = ChurnEstimator(initial_lambda=0.1)
        estimator.update([1, 2], timestamp=0)
        estimator.update([3, 4], timestamp=1)

        estimator.reset(initial_lambda=0.05)
        assert estimator.update_count == 0
        assert estimator.current_neighborhood is None
        assert estimator.previous_neighborhood is None
        assert estimator.estimated_lambda == 0.05

        with pytest.raises(ValueError, match="initial_lambda"):
            estimator.reset(initial_lambda=-1.0)

    def test_clone_independence(self) -> None:
        """Verify clone creates an isolated exact copy."""
        est = ChurnEstimator(eta=0.4, initial_lambda=0.2)
        est.update([1, 2], timestamp=0)
        est.update([2, 3], timestamp=1)

        cloned = est.clone()
        assert cloned.estimated_lambda == est.estimated_lambda
        assert cloned.current_neighborhood == est.current_neighborhood

        # Mutating original does not mutate clone
        est.update([3, 4], timestamp=2)
        assert est.update_count == 3
        assert cloned.update_count == 2
        assert est.estimated_lambda != cloned.estimated_lambda

    def test_neighborhood_caller_mutation_safety(self) -> None:
        """Verify mutating caller's collection does not alter estimator's stored frozenset."""
        mutable_nbrs = [1, 2, 3]
        estimator = ChurnEstimator()
        estimator.update(mutable_nbrs, timestamp=0)

        # Mutate caller collection
        mutable_nbrs.append(999)

        assert 999 not in estimator.current_neighborhood

    def test_static_compute_overlap(self) -> None:
        """Verify static compute_overlap calculation."""
        assert ChurnEstimator.compute_overlap([], []) == 1.0
        assert ChurnEstimator.compute_overlap([1], []) == 0.0
        assert ChurnEstimator.compute_overlap([], [1]) == 0.0
        assert ChurnEstimator.compute_overlap([1, 2], [1, 2]) == 1.0
        assert math.isclose(ChurnEstimator.compute_overlap([1, 2], [2, 3]), 1.0 / 3.0)
