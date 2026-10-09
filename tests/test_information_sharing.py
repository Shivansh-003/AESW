"""
Unit and Integration Tests for AESW Information Sharing & Node-Mediated Communication
=====================================================================================
Tests inter-walker communication via vertex-anchored shared caches under
four distinct communication modes: NO_SHARING, PUSH, PULL, and PUSH_PULL.
"""

from __future__ import annotations

import math
import pytest

from aesw.aesw.churn import ChurnEstimator
from aesw.aesw.communication import (
    CommunicationEvent,
    CommunicationMetrics,
    CommunicationMode,
    NodeMediatedExchange,
    SharedNodeCache,
    validate_communication_mode,
)
from aesw.memory.cache import EvidenceCache
from aesw.memory.decay import exponential_decay
from aesw.memory.models import NodeEvidence
from aesw.memory.types import EvidencePolarity, EvidenceSource


def _create_sample_evidence(
    node_id: int | str,
    walker_id: int | str,
    timestamp: int = 0,
    target_found: bool = False,
    signal_strength: float = 0.0,
    confidence: float = 1.0,
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
    )


class TestCommunicationCore:
    """Core tests for node-mediated exchange, modes, and message accounting."""

    def test_01_no_sharing_mode(self) -> None:
        """Test 1: NO_SHARING mode prevents any evidence exchange and keeps Q=0."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.NO_SHARING)
        cache_a = EvidenceCache()
        cache_b = EvidenceCache()

        ev_a = _create_sample_evidence(node_id=1, walker_id="A", signal_strength=1.0)
        cache_a.store(ev_a)

        # Walker A attempts to push at node 1
        pushed = exchange.push(walker_id="A", at_node=1, evidence=[ev_a], timestamp=1)
        assert pushed == 0
        assert exchange.total_messages == 0

        # Walker B attempts to pull at node 1
        pulled = exchange.pull(walker_id="B", at_node=1, timestamp=2, target_cache=cache_b)
        assert len(pulled) == 0
        assert cache_b.size == 0
        assert exchange.total_messages == 0

    def test_02_push_mode_semantics(self) -> None:
        """Test 2: PUSH mode allows depositing into node cache; accessible through node mediation."""
        # In PUSH mode, walkers push to node cache.
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH)
        ev_a = _create_sample_evidence(node_id=10, walker_id="A", target_found=True)

        # Walker A pushes to Node 5
        pushed = exchange.push(walker_id="A", at_node=5, evidence=ev_a, timestamp=1)
        assert pushed == 1
        assert exchange.total_messages == 1

        # Node 5 holds the record
        node_5_cache = exchange.get_node_cache(5)
        assert node_5_cache.size == 1
        assert node_5_cache.get_raw(10) is not None

        # Verify no direct connection exists between Walker A and Walker B
        assert not hasattr(exchange, "send_direct")
        assert not hasattr(exchange, "walker_to_walker")

    def test_03_pull_mode_semantics(self) -> None:
        """Test 3: PULL mode allows retrieving shared evidence from node cache."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PULL)
        # Pre-seed node 7's cache (as if evidence was deposited)
        node_7_cache = exchange.get_node_cache(7)
        ev_orig = _create_sample_evidence(node_id=25, walker_id="A", signal_strength=0.9)
        node_7_cache.store(ev_orig)

        # Walker B arrives at node 7 and pulls
        cache_b = EvidenceCache()
        pulled = exchange.pull(walker_id="B", at_node=7, timestamp=5, target_cache=cache_b)

        assert len(pulled) == 1
        assert exchange.total_messages == 1
        assert cache_b.size == 1

        # Pulled record is in B's cache
        retrieved = cache_b.get_raw(25)
        assert retrieved is not None
        assert retrieved.walker_id == "A"
        assert retrieved.source == EvidenceSource.RECEIVED_EXCHANGE

    def test_04_push_pull_mode(self) -> None:
        """Test 4: PUSH_PULL mode allows both publishing and retrieving through node cache."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        cache_a = EvidenceCache()
        cache_b = EvidenceCache()

        ev_a = _create_sample_evidence(node_id=1, walker_id="A", target_found=True)
        cache_a.store(ev_a)

        # Walker A visits node 100, executes exchange (push + pull = 2 messages)
        pushed, _ = exchange.exchange(walker_id="A", at_node=100, local_cache=cache_a, timestamp=1)
        assert pushed == 1
        assert exchange.total_messages == 2  # 1 push + 1 pull

        # Walker B visits node 100, pulls from node 100 into its cache
        ev_b = _create_sample_evidence(node_id=2, walker_id="B", signal_strength=0.8)
        cache_b.store(ev_b)

        # Walker B pushes its evidence and pulls available evidence (push + pull = 2 messages)
        pushed_b, pulled_b = exchange.exchange(walker_id="B", at_node=100, local_cache=cache_b, timestamp=2)
        assert pushed_b == 1
        assert pulled_b >= 1
        # Total messages = 2 (from A) + 2 (from B) = 4 messages
        assert exchange.total_messages == 4

        # Walker B now has Walker A's evidence
        assert cache_b.get_raw(1) is not None

    def test_05_three_walker_scenario(self) -> None:
        """Test 5: Three walkers: A's evidence reaches B and C strictly through node cache."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        cache_a = EvidenceCache()
        cache_b = EvidenceCache()
        cache_c = EvidenceCache()

        # Walker A discovers evidence at node 50
        ev_target = _create_sample_evidence(node_id=50, walker_id="Walker_A", target_found=True, confidence=1.0)
        cache_a.store(ev_target)

        # Walker A visits hub node 0 and publishes
        exchange.push_from_cache(walker_id="Walker_A", at_node=0, local_cache=cache_a, timestamp=10)

        # Walker B later visits hub node 0 and pulls
        exchange.pull(walker_id="Walker_B", at_node=0, timestamp=15, target_cache=cache_b)

        # Walker C later visits hub node 0 and pulls
        exchange.pull(walker_id="Walker_C", at_node=0, timestamp=20, target_cache=cache_c)

        # Both Walker B and C have Walker A's evidence
        assert cache_b.get_raw(50) is not None
        assert cache_c.get_raw(50) is not None
        assert cache_b.get_raw(50).walker_id == "Walker_A"
        assert cache_c.get_raw(50).walker_id == "Walker_A"

        # Communication is 1 push + 2 pulls = 3 messages
        assert exchange.total_messages == 3

    def test_06_source_identity_preservation(self) -> None:
        """Test 6: Evidence generated by A retains walker_id=A when received by B."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        ev_a = _create_sample_evidence(node_id=42, walker_id="Alpha", signal_strength=0.85)

        exchange.push(walker_id="Alpha", at_node=1, evidence=ev_a, timestamp=1)

        cache_beta = EvidenceCache()
        pulled = exchange.pull(walker_id="Beta", at_node=1, timestamp=2, target_cache=cache_beta)

        assert len(pulled) == 1
        rec = pulled[0]
        # Crucial invariant: source identity must NOT be rewritten to "Beta"
        assert rec.walker_id == "Alpha"
        assert cache_beta.get_raw(42).walker_id == "Alpha"

    def test_07_evidence_source_stamped_as_received_exchange(self) -> None:
        """Test 7: Exchanged evidence is tagged with EvidenceSource.RECEIVED_EXCHANGE."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        ev_direct = _create_sample_evidence(node_id=9, walker_id="W1", signal_strength=0.5)
        assert ev_direct.source == EvidenceSource.DIRECT_VISIT

        exchange.push(walker_id="W1", at_node=9, evidence=ev_direct, timestamp=1)

        pulled = exchange.pull(walker_id="W2", at_node=9, timestamp=2)
        assert len(pulled) == 1
        assert pulled[0].source == EvidenceSource.RECEIVED_EXCHANGE

    def test_08_duplicate_exchange_does_not_multiply_records(self) -> None:
        """Test 8: Repeated pulls do not duplicate records or cause memory explosion."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        cache_b = EvidenceCache()

        ev = _create_sample_evidence(node_id=7, walker_id="A", timestamp=5, target_found=True)
        exchange.push(walker_id="A", at_node=1, evidence=ev, timestamp=5)

        # Walker B pulls once
        exchange.pull(walker_id="B", at_node=1, timestamp=6, target_cache=cache_b)
        assert cache_b.size == 1

        # Walker B pulls again at later time
        exchange.pull(walker_id="B", at_node=1, timestamp=10, target_cache=cache_b)
        assert cache_b.size == 1  # Still exactly 1 unique node record!

        # Total messages Q is incremented for each operation
        assert exchange.total_messages == 3  # 1 push + 2 pulls

    def test_09_communication_counting_convention(self) -> None:
        """Test 9: Message counter Q increases exactly by 1 per transaction."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        metrics = exchange.metrics

        assert exchange.total_messages == 0
        assert metrics.push_messages == 0
        assert metrics.pull_messages == 0

        # Push operation with 3 evidence items = 1 push message
        items = [
            _create_sample_evidence(1, "A"),
            _create_sample_evidence(2, "A"),
            _create_sample_evidence(3, "A"),
        ]
        exchange.push(walker_id="A", at_node=10, evidence=items, timestamp=1)

        assert exchange.total_messages == 1
        assert metrics.push_messages == 1
        assert metrics.pull_messages == 0
        assert metrics.evidence_records_pushed == 3

        # Pull operation = 1 pull message
        exchange.pull(walker_id="B", at_node=10, timestamp=2)

        assert exchange.total_messages == 2
        assert metrics.push_messages == 1
        assert metrics.pull_messages == 1
        assert metrics.evidence_records_pulled == 3

    def test_10_no_sharing_guarantees_zero_messages(self) -> None:
        """Test 10: In NO_SHARING mode, Q is strictly 0 regardless of calls."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.NO_SHARING)
        cache = EvidenceCache()
        ev = _create_sample_evidence(1, "A")
        cache.store(ev)

        for t in range(10):
            exchange.push(walker_id="A", at_node=1, evidence=ev, timestamp=t)
            exchange.pull(walker_id="B", at_node=1, timestamp=t, target_cache=cache)
            exchange.exchange(walker_id="A", at_node=1, local_cache=cache, timestamp=t)

        assert exchange.total_messages == 0
        assert exchange.metrics.messages_sent == 0
        assert exchange.metrics.messages_received == 0

    def test_11_mode_validation(self) -> None:
        """Test 11: Invalid communication mode is rejected with ValueError."""
        with pytest.raises(ValueError, match="Invalid communication mode"):
            validate_communication_mode("INVALID_MODE")

        with pytest.raises(ValueError, match="Invalid communication mode"):
            NodeMediatedExchange(mode="BROADCAST_ALL")

        # Valid strings work seamlessly
        assert validate_communication_mode("push") == CommunicationMode.PUSH
        assert validate_communication_mode("PULL") == CommunicationMode.PULL

    def test_12_multiple_independent_node_caches(self) -> None:
        """Test 12: Evidence at Node N does not appear at unrelated Node M."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        ev = _create_sample_evidence(node_id=99, walker_id="A", target_found=True)

        # Walker A deposits evidence at Node 1
        exchange.push(walker_id="A", at_node=1, evidence=ev, timestamp=1)

        # Walker B checks Node 2
        pulled_at_node_2 = exchange.pull(walker_id="B", at_node=2, timestamp=2)
        assert len(pulled_at_node_2) == 0

        # Walker B checks Node 1
        pulled_at_node_1 = exchange.pull(walker_id="B", at_node=1, timestamp=3)
        assert len(pulled_at_node_1) == 1
        assert pulled_at_node_1[0].node_id == 99

    def test_13_information_boundary_preserved(self) -> None:
        """Test 13: Communication mediator does not hold or expose hidden environment state."""
        exchange = NodeMediatedExchange()

        forbidden_attributes = [
            "p_on",
            "p_off",
            "true_target_location",
            "ground_truth",
            "global_graph",
            "active_graph_view",
            "future_states",
        ]
        for attr in forbidden_attributes:
            assert not hasattr(exchange, attr)

    def test_14_m8_evidence_cache_integration(self) -> None:
        """Test 14: Received evidence enters existing EvidenceCache and obeys timestamp update rules."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        cache_b = EvidenceCache()

        # Older evidence published
        ev_old = _create_sample_evidence(node_id=5, walker_id="A", timestamp=10, confidence=0.5)
        exchange.push(walker_id="A", at_node=1, evidence=ev_old, timestamp=10)
        exchange.pull(walker_id="B", at_node=1, timestamp=11, target_cache=cache_b)

        assert cache_b.get_raw(5).confidence == 0.5

        # Newer evidence published with higher confidence
        ev_new = _create_sample_evidence(node_id=5, walker_id="A", timestamp=20, confidence=0.9)
        exchange.push(walker_id="A", at_node=1, evidence=ev_new, timestamp=20)
        exchange.pull(walker_id="B", at_node=1, timestamp=21, target_cache=cache_b)

        # Cleanly updated using EvidenceCache rules
        assert cache_b.get_raw(5).confidence == 0.9
        assert cache_b.get_raw(5).timestamp == 20

    def test_15_churn_estimator_integration(self) -> None:
        """Test 15: Received evidence is evaluated using receiver's own local churn estimate lambda_hat."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        cache_b = EvidenceCache()
        estimator_b = ChurnEstimator(eta=0.5)

        # Walker B experiences topological churn locally
        estimator_b.update({1, 2}, timestamp=0)
        estimator_b.update({3, 4}, timestamp=5)
        lambda_b = estimator_b.estimated_lambda
        assert lambda_b > 0.0

        # Walker A created evidence at t=0
        ev_a = _create_sample_evidence(node_id=100, walker_id="A", timestamp=0, confidence=1.0)
        exchange.push(walker_id="A", at_node=1, evidence=ev_a, timestamp=1)

        # Walker B pulls evidence at t=10
        exchange.pull(walker_id="B", at_node=1, timestamp=10, target_cache=cache_b)

        # Walker B evaluates received evidence using its own current lambda_b
        weighted = cache_b.get(node_id=100, current_time=10, lambda_hat=estimator_b)
        assert weighted is not None
        expected_w = exponential_decay(age=10, lambda_hat=lambda_b)
        assert math.isclose(weighted.weight, expected_w, abs_tol=1e-9)

    def test_16_deterministic_communication_behavior(self) -> None:
        """Test 16: Identical sequence of events produces identical states and message counts."""
        ex1 = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        ex2 = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)

        c1_b = EvidenceCache()
        c2_b = EvidenceCache()

        ev = _create_sample_evidence(1, "A", timestamp=5, target_found=True)

        ex1.push("A", 1, ev, 5)
        ex1.pull("B", 1, 6, target_cache=c1_b)

        ex2.push("A", 1, ev, 5)
        ex2.pull("B", 1, 6, target_cache=c2_b)

        assert ex1.total_messages == ex2.total_messages == 2
        assert ex1.metrics.to_dict() == ex2.metrics.to_dict()
        assert c1_b.get_raw(1) == c2_b.get_raw(1)

    def test_17_local_shared_memory_separation(self) -> None:
        """Test 17: Private local evidence remains private until explicitly published."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        cache_a = EvidenceCache()
        cache_b = EvidenceCache()

        # Walker A records private evidence in its local cache
        ev_private = _create_sample_evidence(node_id=42, walker_id="A", target_found=True)
        cache_a.store(ev_private)

        # Walker B checks the node cache before A publishes
        exchange.pull(walker_id="B", at_node=1, timestamp=1, target_cache=cache_b)
        assert cache_b.get_raw(42) is None

        # Walker A publishes to node 1
        exchange.push_from_cache(walker_id="A", at_node=1, local_cache=cache_a, timestamp=2)

        # Walker B pulls again
        exchange.pull(walker_id="B", at_node=1, timestamp=3, target_cache=cache_b)
        assert cache_b.get_raw(42) is not None

    def test_18_three_walker_communication_accounting(self) -> None:
        """Test 18: Exact message count accounting across a multi-walker scenario."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        cache_a = EvidenceCache()
        cache_b = EvidenceCache()
        cache_c = EvidenceCache()

        cache_a.store(_create_sample_evidence(1, "A"))
        cache_b.store(_create_sample_evidence(2, "B"))

        # Step 1: Walker A publishes at Node 1 (1 push = 1 message)
        exchange.push_from_cache("A", 1, cache_a, timestamp=1)

        # Step 2: Walker B visits Node 1 and exchanges (1 push + 1 pull = 2 messages)
        exchange.exchange("B", 1, cache_b, timestamp=2)

        # Step 3: Walker C visits Node 1 and pulls (1 pull = 1 message)
        exchange.pull("C", 1, timestamp=3, target_cache=cache_c)

        # Total expected: 1 + 2 + 1 = 4 messages
        assert exchange.total_messages == 4
        assert exchange.metrics.push_messages == 2
        assert exchange.metrics.pull_messages == 2


class TestScientificBehavior:
    """Scientific validation showing how communication mode affects information availability."""

    def test_information_availability_under_communication_modes(self) -> None:
        """Scientific experiment demonstrating knowledge propagation differences between PUSH_PULL and NO_SHARING.

        Hypothesis:
        Under PUSH_PULL, a searcher can acquire critical target detection clues
        deposited at shared vertices by teammates, expanding its effective visibility.
        Under NO_SHARING, searchers remain completely isolated, unable to benefit
        from teammate detections.
        """
        # Scenario 1: PUSH_PULL
        exchange_sharing = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        cache_a_sharing = EvidenceCache()
        cache_b_sharing = EvidenceCache()

        # Walker A discovers target cue at node 88
        clue = _create_sample_evidence(node_id=88, walker_id="A", timestamp=10, signal_strength=0.95)
        cache_a_sharing.store(clue)

        # Walker A publishes clue at communication hub node 0
        exchange_sharing.push_from_cache("A", 0, cache_a_sharing, timestamp=12)

        # Walker B arrives at hub node 0 and pulls
        exchange_sharing.pull("B", 0, timestamp=15, target_cache=cache_b_sharing)

        # Verification: Walker B now possesses the target clue!
        shared_clue = cache_b_sharing.get_raw(88)
        assert shared_clue is not None
        assert shared_clue.signal_strength == 0.95
        assert shared_clue.walker_id == "A"
        assert shared_clue.source == EvidenceSource.RECEIVED_EXCHANGE
        assert exchange_sharing.total_messages == 2

        # Scenario 2: NO_SHARING
        exchange_no_sharing = NodeMediatedExchange(mode=CommunicationMode.NO_SHARING)
        cache_a_isolated = EvidenceCache()
        cache_b_isolated = EvidenceCache()

        cache_a_isolated.store(clue)

        # Walker A attempts communication at node 0
        exchange_no_sharing.push_from_cache("A", 0, cache_a_isolated, timestamp=12)

        # Walker B arrives at node 0 and attempts communication
        exchange_no_sharing.pull("B", 0, timestamp=15, target_cache=cache_b_isolated)

        # Verification: Walker B remains completely blind to Walker A's discovery!
        assert cache_b_isolated.get_raw(88) is None
        assert cache_b_isolated.size == 0
        assert exchange_no_sharing.total_messages == 0


class TestAuxiliaryAndEdgeCases:
    """Tests for edge cases, reset, empty payloads, and capacity bounds."""

    def test_empty_push_sends_zero_messages(self) -> None:
        """Pushing empty evidence does not transmit a message."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH)
        pushed = exchange.push("A", 1, [], timestamp=1)
        assert pushed == 0
        assert exchange.total_messages == 0

    def test_shared_node_cache_capacity_limit(self) -> None:
        """SharedNodeCache respects maximum capacity with FIFO eviction."""
        node_cache = SharedNodeCache(node_id=1, max_capacity=2)
        ev1 = _create_sample_evidence(1, "A")
        ev2 = _create_sample_evidence(2, "A")
        ev3 = _create_sample_evidence(3, "A")

        node_cache.store(ev1)
        node_cache.store(ev2)
        assert node_cache.size == 2

        # Adding 3rd evicts oldest (1)
        node_cache.store(ev3)
        assert node_cache.size == 2
        assert node_cache.get_raw(1) is None
        assert node_cache.get_raw(2) is not None
        assert node_cache.get_raw(3) is not None

    def test_exclude_self_filter(self) -> None:
        """Pull with exclude_self=True filters out records authored by requesting walker."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        ev_a = _create_sample_evidence(1, "Walker_A")
        ev_b = _create_sample_evidence(2, "Walker_B")

        exchange.push("Walker_A", 10, ev_a, timestamp=1)
        exchange.push("Walker_B", 10, ev_b, timestamp=2)

        # Walker A pulls with exclude_self=True
        pulled = exchange.pull("Walker_A", 10, timestamp=3, exclude_self=True)
        assert len(pulled) == 1
        assert pulled[0].walker_id == "Walker_B"

    def test_reset_functionality(self) -> None:
        """Reset clears all caches and metrics."""
        exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
        exchange.push("A", 1, _create_sample_evidence(1, "A"), timestamp=1)
        assert exchange.total_messages == 1
        assert exchange.has_node_cache(1)

        exchange.reset()
        assert exchange.total_messages == 0
        assert not exchange.has_node_cache(1)
