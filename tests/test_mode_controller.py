"""
Unit and Integration Tests for AESW Adaptive Mode Controller
============================================================
Tests dynamic switching between LOCAL exploration and LONG_JUMP dispersion
based on sliding W-step information acquisition and stagnation tracking.
"""

from __future__ import annotations

import pytest

from aesw.aesw.churn import ChurnEstimator
from aesw.aesw.communication import (
    CommunicationMode,
    NodeMediatedExchange,
)
from aesw.aesw.mode import (
    AdaptiveModeController,
    SearchMode,
    validate_search_mode,
)
from aesw.environment.observation import (
    Observation,
    ObservedEdgeInfo,
    TargetSignalObservation,
)
from aesw.environment.types import EdgeState
from aesw.memory.cache import EvidenceCache
from aesw.memory.models import NodeEvidence
from aesw.memory.types import EvidencePolarity, EvidenceSource


def _create_test_observation(
    walker_id: int | str,
    current_node: int | str,
    time: int,
    detected: bool,
    signal_strength: float = 1.0,
) -> Observation:
    """Helper to synthesize a valid local Observation snapshot."""
    return Observation(
        walker_id=walker_id,
        current_node=current_node,
        time=time,
        checked_neighbors=(2,),
        observed_edges={
            2: ObservedEdgeInfo(
                neighbor_id=2,
                state=EdgeState.ON,
                observed_at_time=time,
            )
        },
        target_signal=TargetSignalObservation(
            detected=detected,
            signal_strength=signal_strength if detected else 0.0,
            timestamp=time,
        ),
        neighbor_budget_used=1,
    )


def _create_sample_evidence(
    node_id: int | str,
    walker_id: int | str = "w1",
    timestamp: int = 0,
    target_found: bool = False,
    signal_strength: float = 0.0,
    confidence: float = 1.0,
    source: EvidenceSource = EvidenceSource.DIRECT_VISIT,
) -> NodeEvidence:
    """Helper to synthesize a valid NodeEvidence record."""
    return NodeEvidence(
        node_id=node_id,
        visited=True,
        target_found=target_found,
        signal_strength=signal_strength,
        timestamp=timestamp,
        walker_id=walker_id,
        confidence=confidence,
        source=source,
    )


class TestAdaptiveModeControllerCore:
    """Core state, window, stagnation, and transition tests for AdaptiveModeController."""

    def test_01_initial_mode_is_local(self) -> None:
        """Test 1: Initial search mode is strictly SearchMode.LOCAL."""
        controller = AdaptiveModeController(window_size=5)
        assert controller.mode == SearchMode.LOCAL
        assert controller.is_local is True
        assert controller.is_long_jump is False
        assert controller.current_mode == SearchMode.LOCAL

    def test_02_initial_stagnation_is_zero(self) -> None:
        """Test 2: Initial stagnation step counter is 0."""
        controller = AdaptiveModeController(window_size=5)
        assert controller.stagnation_steps == 0
        assert controller.transition_count == 0
        assert controller.recent_information == []
        assert controller.last_information_time is None

    def test_03_first_useful_evidence_maintains_local(self) -> None:
        """Test 3: First useful positive evidence keeps mode in LOCAL with stagnation 0."""
        controller = AdaptiveModeController(window_size=4)
        ev_pos = _create_sample_evidence(node_id=1, target_found=True)

        mode = controller.update(timestamp=0, evidence=ev_pos)
        assert mode == SearchMode.LOCAL
        assert controller.stagnation_steps == 0
        assert controller.last_information_time == 0
        assert controller.recent_information == [True]

    def test_04_positive_evidence_resets_stagnation(self) -> None:
        """Test 4: Acquisition of positive evidence resets accumulated stagnation back to 0."""
        controller = AdaptiveModeController(window_size=5)

        # 3 stagnant steps
        controller.update(timestamp=1, useful=False)
        controller.update(timestamp=2, useful=False)
        controller.update(timestamp=3, useful=False)
        assert controller.stagnation_steps == 3
        assert controller.mode == SearchMode.LOCAL

        # Step 4: useful positive evidence arrives
        ev_new = _create_sample_evidence(node_id=99, timestamp=4, signal_strength=0.9)
        mode = controller.update(timestamp=4, evidence=ev_new)

        assert mode == SearchMode.LOCAL
        assert controller.stagnation_steps == 0
        assert controller.last_information_time == 4

    def test_05_no_information_steps_increase_stagnation(self) -> None:
        """Test 5: Steps with no useful information increment stagnation counter monotonically."""
        controller = AdaptiveModeController(window_size=10)

        for t in range(1, 6):
            controller.update(timestamp=t, useful=False)
            assert controller.stagnation_steps == t

    def test_06_fewer_than_w_stagnant_steps_remains_local(self) -> None:
        """Test 6: Mode remains strictly LOCAL as long as stagnation_steps < W."""
        W = 5
        controller = AdaptiveModeController(window_size=W)

        # Steps 1 to W-1 (4 steps)
        for t in range(1, W):
            mode = controller.update(timestamp=t, useful=False)
            assert mode == SearchMode.LOCAL
            assert controller.stagnation_steps == t

    def test_07_exactly_w_stagnant_steps_triggers_long_jump(self) -> None:
        """Test 7: Reaching exactly W stagnant steps triggers transition to LONG_JUMP."""
        W = 4
        controller = AdaptiveModeController(window_size=W)

        controller.update(timestamp=1, useful=False)
        controller.update(timestamp=2, useful=False)
        controller.update(timestamp=3, useful=False)
        assert controller.mode == SearchMode.LOCAL

        # Step 4 = exactly W
        mode = controller.update(timestamp=4, useful=False)
        assert mode == SearchMode.LONG_JUMP
        assert controller.is_long_jump is True
        assert controller.stagnation_steps == 4
        assert controller.transition_count == 1

    def test_08_useful_information_after_long_jump_returns_to_local(self) -> None:
        """Test 8: New useful information while in LONG_JUMP returns mode to LOCAL."""
        controller = AdaptiveModeController(window_size=3)

        # Stagnate into LONG_JUMP
        controller.update(timestamp=1, useful=False)
        controller.update(timestamp=2, useful=False)
        controller.update(timestamp=3, useful=False)
        assert controller.mode == SearchMode.LONG_JUMP

        # Useful evidence appears at step 4
        ev = _create_sample_evidence(node_id=5, timestamp=4, signal_strength=1.0)
        mode = controller.update(timestamp=4, evidence=ev)

        assert mode == SearchMode.LOCAL
        assert controller.is_local is True
        assert controller.stagnation_steps == 0
        assert controller.transition_count == 2

    def test_09_window_bounded_to_w_steps(self) -> None:
        """Test 9: Sliding window history is strictly bounded to length W."""
        W = 3
        controller = AdaptiveModeController(window_size=W)

        for t in range(10):
            controller.update(timestamp=t, useful=(t % 2 == 0))
            assert len(controller.recent_information) <= W

        assert len(controller.recent_information) == W

    def test_10_old_information_falls_out_of_window(self) -> None:
        """Test 10: Old positive information falls out of window after W stagnant steps."""
        W = 3
        controller = AdaptiveModeController(window_size=W)

        # Useful info at step 0
        controller.update(timestamp=0, useful=True)
        assert controller.recent_information == [True]

        # Step 1: no info
        controller.update(timestamp=1, useful=False)
        assert controller.recent_information == [True, False]

        # Step 2: no info
        controller.update(timestamp=2, useful=False)
        assert controller.recent_information == [True, False, False]

        # Step 3: no info -> step 0's True falls out!
        controller.update(timestamp=3, useful=False)
        assert controller.recent_information == [False, False, False]
        assert controller.mode == SearchMode.LONG_JUMP

    def test_11_repeated_identical_stale_evidence_does_not_reset_stagnation(self) -> None:
        """Test 11: Repeatedly presenting the exact same stale record does not reset stagnation."""
        controller = AdaptiveModeController(window_size=3)

        stale_evidence = _create_sample_evidence(
            node_id=1,
            timestamp=0,
            signal_strength=1.0,
        )

        # Step 0: Initial acquisition (novel)
        controller.update(timestamp=0, evidence=stale_evidence)
        assert controller.stagnation_steps == 0

        # Steps 1, 2, 3: Presenting the EXACT SAME stale record
        controller.update(timestamp=1, evidence=stale_evidence)
        assert controller.stagnation_steps == 1

        controller.update(timestamp=2, evidence=stale_evidence)
        assert controller.stagnation_steps == 2

        controller.update(timestamp=3, evidence=stale_evidence)
        assert controller.stagnation_steps == 3
        assert controller.mode == SearchMode.LONG_JUMP

    def test_12_genuinely_new_evidence_resets_stagnation(self) -> None:
        """Test 12: Genuinely new evidence (new node or newer timestamp) resets stagnation."""
        controller = AdaptiveModeController(window_size=4)

        ev_old = _create_sample_evidence(node_id=1, timestamp=0, signal_strength=1.0)
        controller.update(timestamp=0, evidence=ev_old)

        controller.update(timestamp=1, useful=False)
        controller.update(timestamp=2, useful=False)
        assert controller.stagnation_steps == 2

        # Genuinely new positive evidence for node 2 at timestamp 3
        ev_new = _create_sample_evidence(node_id=2, timestamp=3, signal_strength=1.0)
        controller.update(timestamp=3, evidence=ev_new)

        assert controller.stagnation_steps == 0
        assert controller.mode == SearchMode.LOCAL

    def test_13_positive_sensor_reading_treated_as_information(self) -> None:
        """Test 13: Positive sensor observation resets stagnation regardless of true target presence."""
        controller = AdaptiveModeController(window_size=3)

        controller.update(timestamp=0, useful=False)
        controller.update(timestamp=1, useful=False)
        assert controller.stagnation_steps == 2

        # Walker receives noisy positive sensor reading
        obs_positive = _create_test_observation(
            walker_id="w1",
            current_node=10,
            time=2,
            detected=True,
            signal_strength=0.7,
        )
        controller.update(timestamp=2, evidence=obs_positive)

        assert controller.stagnation_steps == 0
        assert controller.mode == SearchMode.LOCAL

    def test_14_negative_evidence_does_not_count_as_strong_information(self) -> None:
        """Test 14: Negative evidence (absence confirmation) does not reset stagnation."""
        controller = AdaptiveModeController(window_size=3)

        ev_negative = _create_sample_evidence(
            node_id=1,
            timestamp=1,
            target_found=False,
            signal_strength=0.0,
        )
        assert ev_negative.is_negative is True

        controller.update(timestamp=1, evidence=ev_negative)
        assert controller.stagnation_steps == 1

        controller.update(timestamp=2, evidence=ev_negative)
        assert controller.stagnation_steps == 2

        obs_negative = _create_test_observation(
            walker_id="w1",
            current_node=2,
            time=3,
            detected=False,
        )
        controller.update(timestamp=3, evidence=obs_negative)
        assert controller.stagnation_steps == 3
        assert controller.mode == SearchMode.LONG_JUMP

    def test_15_received_exchange_evidence_affects_controller(self) -> None:
        """Test 15: Legitimate evidence received through node-mediated exchange can reset stagnation."""
        controller = AdaptiveModeController(window_size=3)

        controller.update(timestamp=1, useful=False)
        controller.update(timestamp=2, useful=False)
        assert controller.stagnation_steps == 2

        # Evidence received via exchange from teammate
        ev_shared = _create_sample_evidence(
            node_id=50,
            walker_id="Walker_Teammate",
            timestamp=2,
            signal_strength=0.9,
            source=EvidenceSource.RECEIVED_EXCHANGE,
        )

        controller.update(timestamp=3, evidence=ev_shared)
        assert controller.stagnation_steps == 0
        assert controller.mode == SearchMode.LOCAL

    def test_16_no_sharing_prevents_exchange_cues(self) -> None:
        """Test 16: Under NO_SHARING, no exchange records arrive, allowing natural stagnation."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.NO_SHARING)
        controller = AdaptiveModeController(window_size=2)

        # Walker B pulls under NO_SHARING -> receives nothing
        pulled = exchange.pull("B", at_node=1, timestamp=1)
        assert len(pulled) == 0

        controller.update(timestamp=1, evidence=pulled)
        assert controller.stagnation_steps == 1

        pulled_2 = exchange.pull("B", at_node=1, timestamp=2)
        controller.update(timestamp=2, evidence=pulled_2)
        assert controller.stagnation_steps == 2
        assert controller.mode == SearchMode.LONG_JUMP

    def test_17_deterministic_behavior(self) -> None:
        """Test 17: Identical inputs produce bit-for-bit identical mode and stagnation traces."""
        ctrl1 = AdaptiveModeController(window_size=4)
        ctrl2 = AdaptiveModeController(window_size=4)

        inputs = [
            (1, False),
            (2, False),
            (3, True),
            (4, False),
            (5, False),
            (6, False),
            (7, False),
        ]

        modes1 = [ctrl1.update(t, useful=u) for t, u in inputs]
        modes2 = [ctrl2.update(t, useful=u) for t, u in inputs]

        assert modes1 == modes2
        assert ctrl1.to_dict() == ctrl2.to_dict()

    def test_18_invalid_window_size_rejected(self) -> None:
        """Test 18: Non-positive window sizes are rejected with ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            AdaptiveModeController(window_size=0)

        with pytest.raises(ValueError, match="strictly positive"):
            AdaptiveModeController(window_size=-3)

    def test_19_reset_clears_state(self) -> None:
        """Test 19: reset() returns controller completely to initial unobserved state."""
        controller = AdaptiveModeController(window_size=3)

        controller.update(timestamp=1, useful=False)
        controller.update(timestamp=2, useful=False)
        controller.update(timestamp=3, useful=False)
        assert controller.mode == SearchMode.LONG_JUMP
        assert controller.transition_count == 1

        controller.reset()

        assert controller.mode == SearchMode.LOCAL
        assert controller.stagnation_steps == 0
        assert controller.recent_information == []
        assert controller.last_information_time is None
        assert controller.transition_count == 0
        assert controller.step_count == 0

    def test_20_multiple_controllers_maintain_isolated_state(self) -> None:
        """Test 20: Multiple walker controllers maintain independent isolated state."""
        ctrl_a = AdaptiveModeController(window_size=3)
        ctrl_b = AdaptiveModeController(window_size=3)

        # Walker A finds useful clues
        ctrl_a.update(timestamp=1, useful=True)
        ctrl_a.update(timestamp=2, useful=True)

        # Walker B stagnates
        ctrl_b.update(timestamp=1, useful=False)
        ctrl_b.update(timestamp=2, useful=False)
        ctrl_b.update(timestamp=3, useful=False)

        assert ctrl_a.mode == SearchMode.LOCAL
        assert ctrl_b.mode == SearchMode.LONG_JUMP
        assert ctrl_a.stagnation_steps == 0
        assert ctrl_b.stagnation_steps == 3

    def test_21_information_boundary_preserved(self) -> None:
        """Test 21: Epistemic firewall: controller cannot access ground-truth graph or target."""
        controller = AdaptiveModeController()

        forbidden_attributes = [
            "target_position",
            "target_state",
            "p_on",
            "p_off",
            "full_graph",
            "ActiveGraphView",
            "GroundTruthState",
            "future_graph_states",
            "shortest_path_to_target",
            "actual_target_distance",
        ]
        for attr in forbidden_attributes:
            assert not hasattr(controller, attr)

    def test_22_evidence_integration(self) -> None:
        """Test 22: Integrates directly with NodeEvidence polarity without duplicate models."""
        controller = AdaptiveModeController(window_size=3)
        cache = EvidenceCache()

        ev = NodeEvidence(
            node_id=12,
            visited=True,
            target_found=True,
            signal_strength=1.0,
            timestamp=1,
            walker_id="w1",
            confidence=1.0,
        )
        cache.store(ev)

        mode = controller.update(timestamp=1, evidence=cache.get_raw(12))
        assert mode == SearchMode.LOCAL
        assert controller.stagnation_steps == 0

    def test_23_churn_estimator_compatibility(self) -> None:
        """Test 23: Diagnostic churn rate lambda_hat can be passed and tracked."""
        estimator = ChurnEstimator(eta=0.5, initial_lambda=0.1)
        controller = AdaptiveModeController(window_size=4)

        estimator.update({1, 2}, timestamp=0)
        estimator.update({3, 4}, timestamp=2)

        controller.update(timestamp=2, useful=False, lambda_hat=estimator)
        assert controller.lambda_hat == estimator.estimated_lambda
        assert controller.lambda_hat > 0.0

    def test_24_received_exchange_integration(self) -> None:
        """Test 24: Direct integration with node exchange output."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        controller = AdaptiveModeController(window_size=3)

        # Walker A deposits evidence at hub node 0
        ev_pos = _create_sample_evidence(node_id=77, walker_id="A", signal_strength=1.0)
        exchange.push(walker_id="A", at_node=0, evidence=ev_pos, timestamp=5)

        # Stagnate for 2 steps
        controller.update(timestamp=5, useful=False)
        controller.update(timestamp=6, useful=False)
        assert controller.stagnation_steps == 2

        # Walker B pulls at hub node 0
        pulled = exchange.pull(walker_id="B", at_node=0, timestamp=7)
        mode = controller.update(timestamp=7, evidence=pulled)

        assert mode == SearchMode.LOCAL
        assert controller.stagnation_steps == 0

    def test_25_complete_mode_transition_cycle(self) -> None:
        """Test 25: Verification of full cycle: LOCAL -> W stagnation -> LONG_JUMP -> cue -> LOCAL."""
        W = 3
        controller = AdaptiveModeController(window_size=W)

        # Stage 1: Initially LOCAL
        assert controller.mode == SearchMode.LOCAL

        # Stage 2: Stagnate for W steps -> transitions to LONG_JUMP
        controller.update(timestamp=1, useful=False)
        controller.update(timestamp=2, useful=False)
        controller.update(timestamp=3, useful=False)
        assert controller.mode == SearchMode.LONG_JUMP
        assert controller.transition_count == 1

        # Stage 3: Remain in LONG_JUMP while stagnant
        controller.update(timestamp=4, useful=False)
        assert controller.mode == SearchMode.LONG_JUMP
        assert controller.transition_count == 1

        # Stage 4: Useful evidence arrives -> transitions back to LOCAL
        ev_target = _create_sample_evidence(node_id=1, timestamp=5, signal_strength=0.9)
        controller.update(timestamp=5, evidence=ev_target)
        assert controller.mode == SearchMode.LOCAL
        assert controller.transition_count == 2
        assert controller.stagnation_steps == 0


class TestScientificBehavior:
    """Scientific validation showing adaptive mode response vs fixed scheduling."""

    def test_adaptive_vs_scheduled_behavior(self) -> None:
        """Verify that mode switching responds to information density rather than elapsed time.

        Scientific Hypothesis:
        Under high information density (continuous clues), the walker remains in
        LOCAL exploration indefinitely, regardless of elapsed time. Under an information
        vacuum, the walker triggers LONG_JUMP exactly after W stagnant steps.
        """
        W = 4
        rich_controller = AdaptiveModeController(window_size=W)
        sparse_controller = AdaptiveModeController(window_size=W)

        # Simulation runs for 20 steps
        for t in range(20):
            # Rich environment: clues arrive regularly every 2 steps
            rich_info = (t % 2 == 0)
            rich_mode = rich_controller.update(timestamp=t, useful=rich_info)

            # Sparse environment: completely empty
            sparse_mode = sparse_controller.update(timestamp=t, useful=False)

            # In rich environment, walker NEVER enters LONG_JUMP despite 20 steps elapsed!
            assert rich_mode == SearchMode.LOCAL

            # In sparse environment, walker entered LONG_JUMP at step W-1 (3) and remained there
            if t >= W - 1:
                assert sparse_mode == SearchMode.LONG_JUMP


class TestAuxiliaryAndEdgeCases:
    """Auxiliary edge case testing."""

    def test_timestamp_validation(self) -> None:
        """Negative timestamps and backward time travel are rejected."""
        controller = AdaptiveModeController()

        with pytest.raises(ValueError, match="non-negative"):
            controller.update(timestamp=-1)

        controller.update(timestamp=5)
        with pytest.raises(ValueError, match="cannot decrease"):
            controller.update(timestamp=4)

    def test_search_mode_validation(self) -> None:
        """Mode validation handles strings and enums correctly."""
        assert validate_search_mode(SearchMode.LOCAL) == SearchMode.LOCAL
        assert validate_search_mode("local") == SearchMode.LOCAL
        assert validate_search_mode("LONG_JUMP") == SearchMode.LONG_JUMP

        with pytest.raises(ValueError, match="Invalid search mode"):
            validate_search_mode("TELEPORT")

    def test_lambda_hat_setter_validation(self) -> None:
        """Negative churn rate is rejected in setter."""
        controller = AdaptiveModeController()
        with pytest.raises(ValueError, match="non-negative"):
            controller.lambda_hat = -0.5
