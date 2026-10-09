"""
Automated Test Suite for AESW Evidence Memory
=============================================
Tests structured evidence representation, adaptive exponential decay,
confidence weighting, local node-level caching, update rules, and observation ingestion.
"""

import math
import pytest

from aesw.memory.types import EvidencePolarity, EvidenceSource
from aesw.memory.decay import (
    exponential_decay,
    compute_half_life,
    decay_rate_from_half_life,
)
from aesw.memory.models import NodeEvidence, WeightedEvidence
from aesw.memory.cache import EvidenceCache
from aesw.environment.observation import (
    Observation,
    TargetSignalObservation,
    ObservedEdgeInfo,
)
from aesw.environment.types import EdgeState


# ============================================================================
# 1. Exponential Decay & Mathematical Formulations
# ============================================================================

def test_exponential_decay_boundary_conditions():
    """Verify decay equals 1.0 at age=0 or lambda_hat=0."""
    # Age = 0: Newly collected evidence has weight 1.0 regardless of churn
    assert exponential_decay(age=0, lambda_hat=0.0) == 1.0
    assert exponential_decay(age=0, lambda_hat=0.05) == 1.0
    assert exponential_decay(age=0, lambda_hat=1.0) == 1.0

    # Churn = 0: Static graph evidence never decays
    assert exponential_decay(age=10, lambda_hat=0.0) == 1.0
    assert exponential_decay(age=1000, lambda_hat=0.0) == 1.0


def test_exponential_decay_monotonicity():
    """Verify that weight strictly decreases as age increases for lambda_hat > 0."""
    lambda_hat = 0.05
    w_0 = exponential_decay(age=0, lambda_hat=lambda_hat)
    w_10 = exponential_decay(age=10, lambda_hat=lambda_hat)
    w_50 = exponential_decay(age=50, lambda_hat=lambda_hat)
    w_100 = exponential_decay(age=100, lambda_hat=lambda_hat)

    assert w_0 > w_10 > w_50 > w_100
    assert 0.0 < w_100 < 1.0


def test_exponential_decay_churn_sensitivity():
    """Verify that faster churn rates produce more rapid decay for the same age."""
    age = 20
    w_slow = exponential_decay(age=age, lambda_hat=0.01)    # exp(-0.2) ≈ 0.8187
    w_medium = exponential_decay(age=age, lambda_hat=0.05)  # exp(-1.0) ≈ 0.3679
    w_fast = exponential_decay(age=age, lambda_hat=0.20)    # exp(-4.0) ≈ 0.0183

    assert w_slow > w_medium > w_fast
    assert pytest.approx(w_slow, 1e-4) == math.exp(-0.2)
    assert pytest.approx(w_medium, 1e-4) == math.exp(-1.0)
    assert pytest.approx(w_fast, 1e-4) == math.exp(-4.0)


def test_exponential_decay_half_life():
    """Verify that at age = half_life, weight is exactly 0.5."""
    lambda_hat = 0.05
    t_half = compute_half_life(lambda_hat)
    expected_half_life = math.log(2.0) / 0.05

    assert pytest.approx(t_half, 1e-6) == expected_half_life
    w_at_half_life = exponential_decay(age=t_half, lambda_hat=lambda_hat)
    assert pytest.approx(w_at_half_life, 1e-6) == 0.5

    # Reverse calculation
    recomputed_lambda = decay_rate_from_half_life(t_half)
    assert pytest.approx(recomputed_lambda, 1e-6) == lambda_hat


def test_exponential_decay_underflow_protection():
    """Verify large negative exponents return 0.0 gracefully without error."""
    w = exponential_decay(age=10000, lambda_hat=1.0)
    assert w == 0.0


def test_exponential_decay_invalid_inputs():
    """Verify negative age or negative lambda_hat raises ValueError."""
    with pytest.raises(ValueError, match="Evidence age must be non-negative"):
        exponential_decay(age=-1, lambda_hat=0.05)

    with pytest.raises(ValueError, match="lambda_hat must be non-negative"):
        exponential_decay(age=10, lambda_hat=-0.05)

    with pytest.raises(ValueError, match="lambda_hat must be non-negative"):
        compute_half_life(lambda_hat=-0.1)

    with pytest.raises(ValueError, match="Half-life must be strictly positive"):
        decay_rate_from_half_life(half_life=-5.0)


# ============================================================================
# 2. NodeEvidence Model Tests
# ============================================================================

def test_node_evidence_initialization():
    """Verify NodeEvidence stores attributes and infers polarity correctly."""
    # Negative evidence: visited, no target found, signal strength 0.0
    ev_neg = NodeEvidence(
        node_id=42,
        visited=True,
        target_found=False,
        signal_strength=0.0,
        timestamp=17,
        walker_id=2,
        confidence=0.8,
    )
    assert ev_neg.node_id == 42
    assert ev_neg.visited is True
    assert ev_neg.target_found is False
    assert ev_neg.signal_strength == 0.0
    assert ev_neg.timestamp == 17
    assert ev_neg.walker_id == 2
    assert ev_neg.confidence == 0.8
    assert ev_neg.polarity == EvidencePolarity.NEGATIVE
    assert ev_neg.is_negative is True
    assert ev_neg.is_positive is False

    # Positive evidence: sensor detection
    ev_pos = NodeEvidence(
        node_id=10,
        visited=False,
        target_found=False,
        signal_strength=0.9,
        timestamp=20,
        walker_id=1,
        confidence=0.85,
    )
    assert ev_pos.polarity == EvidencePolarity.POSITIVE
    assert ev_pos.is_positive is True
    assert ev_pos.is_negative is False


def test_node_evidence_validation():
    """Verify invariant assertions for NodeEvidence construction."""
    with pytest.raises(ValueError, match="node_id must not be None"):
        NodeEvidence(node_id=None, visited=True, target_found=False, signal_strength=0.0, timestamp=0, walker_id=1, confidence=1.0)

    with pytest.raises(ValueError, match="timestamp must be non-negative"):
        NodeEvidence(node_id=1, visited=True, target_found=False, signal_strength=0.0, timestamp=-5, walker_id=1, confidence=1.0)

    with pytest.raises(ValueError, match="signal_strength must be in"):
        NodeEvidence(node_id=1, visited=True, target_found=False, signal_strength=1.5, timestamp=0, walker_id=1, confidence=1.0)

    with pytest.raises(ValueError, match="confidence must be in"):
        NodeEvidence(node_id=1, visited=True, target_found=False, signal_strength=0.0, timestamp=0, walker_id=1, confidence=-0.1)

    with pytest.raises(ValueError, match="walker_id must not be None"):
        NodeEvidence(node_id=1, visited=True, target_found=False, signal_strength=0.0, timestamp=0, walker_id=None, confidence=1.0)


def test_node_evidence_age_and_weight():
    """Verify age, weight, and effective confidence evaluation."""
    ev = NodeEvidence(
        node_id=5,
        visited=True,
        target_found=False,
        signal_strength=0.0,
        timestamp=10,
        walker_id=0,
        confidence=0.8,
    )

    assert ev.age(current_time=10) == 0
    assert ev.age(current_time=30) == 20

    with pytest.raises(ValueError, match="cannot be earlier than evidence timestamp"):
        ev.age(current_time=5)

    # Effective confidence at t=30 with lambda_hat=0.05 (age=20, w=exp(-1) ≈ 0.367879)
    w = ev.weight(current_time=30, lambda_hat=0.05)
    assert pytest.approx(w, 1e-4) == math.exp(-1.0)
    eff_conf = ev.effective_confidence(current_time=30, lambda_hat=0.05)
    assert pytest.approx(eff_conf, 1e-4) == 0.8 * math.exp(-1.0)

    # Evaluate snapshot
    snap = ev.evaluate(current_time=30, lambda_hat=0.05)
    assert isinstance(snap, WeightedEvidence)
    assert snap.age == 20
    assert pytest.approx(snap.weight, 1e-4) == w
    assert pytest.approx(snap.effective_confidence, 1e-4) == eff_conf
    assert snap.node_id == 5


def test_node_evidence_serialization():
    """Verify NodeEvidence dictionary serialization and deserialization."""
    ev = NodeEvidence(
        node_id="node_x",
        visited=True,
        target_found=True,
        signal_strength=1.0,
        timestamp=42,
        walker_id=3,
        confidence=0.95,
        source=EvidenceSource.DIRECT_VISIT,
        metadata={"priority": "high"},
    )

    d = ev.to_dict()
    reconstructed = NodeEvidence.from_dict(d)

    assert reconstructed.node_id == ev.node_id
    assert reconstructed.visited == ev.visited
    assert reconstructed.target_found == ev.target_found
    assert reconstructed.signal_strength == ev.signal_strength
    assert reconstructed.timestamp == ev.timestamp
    assert reconstructed.walker_id == ev.walker_id
    assert reconstructed.confidence == ev.confidence
    assert reconstructed.polarity == ev.polarity
    assert reconstructed.source == ev.source
    assert reconstructed.metadata == ev.metadata


# ============================================================================
# 3. EvidenceCache Operations Tests
# ============================================================================

def test_evidence_cache_basic_storage_and_retrieval():
    """Verify storing, querying, and checking membership in EvidenceCache."""
    cache = EvidenceCache(default_lambda_hat=0.05)
    assert cache.size == 0
    assert len(cache) == 0

    ev = NodeEvidence(
        node_id=1,
        visited=True,
        target_found=False,
        signal_strength=0.0,
        timestamp=5,
        walker_id=0,
        confidence=0.9,
    )
    assert cache.store(ev) is True
    assert cache.size == 1
    assert 1 in cache
    assert cache.has_node(1) if hasattr(cache, "has_node") else (1 in cache)

    # Query unweighted
    raw = cache.get_raw(1)
    assert raw == ev

    # Query weighted at t=5 (age=0 -> w=1.0)
    w_ev_5 = cache.get(node_id=1, current_time=5)
    assert w_ev_5 is not None
    assert w_ev_5.age == 0
    assert w_ev_5.weight == 1.0
    assert w_ev_5.effective_confidence == 0.9

    # Query weighted at t=25 (age=20 -> w=exp(-1) ≈ 0.3679)
    w_ev_25 = cache.get(node_id=1, current_time=25)
    assert w_ev_25 is not None
    assert w_ev_25.age == 20
    assert pytest.approx(w_ev_25.weight, 1e-4) == math.exp(-1.0)
    assert pytest.approx(w_ev_25.effective_confidence, 1e-4) == 0.9 * math.exp(-1.0)

    # Query nonexistent node
    assert cache.get(node_id=999, current_time=25) is None
    assert cache.get_raw(999) is None


def test_evidence_cache_update_rules():
    """Verify that newer evidence updates older evidence, while older is ignored."""
    cache = EvidenceCache(default_lambda_hat=0.05)

    ev_t10 = NodeEvidence(
        node_id="A",
        visited=True,
        target_found=False,
        signal_strength=0.0,
        timestamp=10,
        walker_id=1,
        confidence=0.8,
    )
    cache.store(ev_t10)

    # 1. Strictly newer evidence at t=20 should overwrite t=10
    ev_t20 = NodeEvidence(
        node_id="A",
        visited=True,
        target_found=True,
        signal_strength=1.0,
        timestamp=20,
        walker_id=1,
        confidence=0.95,
    )
    assert cache.store(ev_t20) is True
    assert cache.get_raw("A").timestamp == 20
    assert cache.get_raw("A").target_found is True

    # 2. Strictly older evidence at t=5 should NOT overwrite t=20
    ev_t5 = NodeEvidence(
        node_id="A",
        visited=True,
        target_found=False,
        signal_strength=0.0,
        timestamp=5,
        walker_id=2,
        confidence=0.99,
    )
    assert cache.store(ev_t5) is False
    assert cache.get_raw("A").timestamp == 20

    # 3. Same timestamp: higher confidence overwrites lower confidence
    ev_t20_higher = NodeEvidence(
        node_id="A",
        visited=True,
        target_found=True,
        signal_strength=1.0,
        timestamp=20,
        walker_id=3,
        confidence=0.99,
    )
    assert cache.store(ev_t20_higher) is True
    assert cache.get_raw("A").confidence == 0.99


def test_evidence_cache_positive_and_negative_filtering():
    """Verify selective retrieval of positive vs negative evidence."""
    cache = EvidenceCache(default_lambda_hat=0.05)

    # Node 1: Positive detection at t=10
    cache.store(NodeEvidence(node_id=1, visited=True, target_found=False, signal_strength=1.0, timestamp=10, walker_id=0, confidence=0.8))
    # Node 2: Positive target found at t=10
    cache.store(NodeEvidence(node_id=2, visited=True, target_found=True, signal_strength=1.0, timestamp=10, walker_id=0, confidence=1.0))
    # Node 3: Negative absence at t=10
    cache.store(NodeEvidence(node_id=3, visited=True, target_found=False, signal_strength=0.0, timestamp=10, walker_id=0, confidence=0.9))
    # Node 4: Negative absence at t=10
    cache.store(NodeEvidence(node_id=4, visited=True, target_found=False, signal_strength=0.0, timestamp=10, walker_id=0, confidence=0.5))

    pos_list = cache.get_positive(current_time=10)
    neg_list = cache.get_negative(current_time=10)

    assert len(pos_list) == 2
    assert {p.node_id for p in pos_list} == {1, 2}
    # Sorted by effective confidence descending: Node 2 (1.0) then Node 1 (0.8)
    assert pos_list[0].node_id == 2
    assert pos_list[1].node_id == 1

    assert len(neg_list) == 2
    assert {n.node_id for n in neg_list} == {3, 4}
    assert neg_list[0].node_id == 3
    assert neg_list[1].node_id == 4


def test_evidence_cache_pruning_stale():
    """Verify pruning removes entries whose decay weight falls below threshold."""
    cache = EvidenceCache(default_lambda_hat=0.10)

    # Node 1: recorded at t=0
    cache.store(NodeEvidence(node_id=1, visited=True, target_found=False, signal_strength=0.0, timestamp=0, walker_id=0, confidence=0.8))
    # Node 2: recorded at t=95 (very fresh at t=100)
    cache.store(NodeEvidence(node_id=2, visited=True, target_found=False, signal_strength=0.0, timestamp=95, walker_id=0, confidence=0.8))

    # At t=100, lambda_hat=0.10:
    # Node 1 age = 100 -> w = exp(-10.0) ≈ 4.5e-5 < 1e-3 (stale)
    # Node 2 age = 5 -> w = exp(-0.5) ≈ 0.6065 > 1e-3 (fresh)
    pruned_count = cache.prune_stale(current_time=100, min_weight=1e-3)

    assert pruned_count == 1
    assert 1 not in cache
    assert 2 in cache
    assert cache.size == 1


def test_evidence_cache_store_from_observation():
    """Verify constructing and storing evidence directly from an Observation."""
    cache = EvidenceCache(default_lambda_hat=0.05)


    obs = Observation(
        walker_id=3,
        current_node=14,
        time=12,
        checked_neighbors=(1, 2),
        observed_edges={
            1: ObservedEdgeInfo(neighbor_id=1, state=EdgeState.ON, observed_at_time=12),
            2: ObservedEdgeInfo(neighbor_id=2, state=EdgeState.ON, observed_at_time=12),
        },
        target_signal=TargetSignalObservation(detected=True, signal_strength=1.0, timestamp=12),
        neighbor_budget_used=2,
    )

    ev = cache.store_from_observation(obs=obs, confidence=0.85)

    assert ev.node_id == 14
    assert ev.walker_id == 3
    assert ev.timestamp == 12
    assert ev.signal_strength == 1.0
    assert ev.is_positive is True
    assert ev.confidence == 0.85
    assert 14 in cache


def test_evidence_cache_merge():
    """Verify merging two caches updates entries according to timestamps."""
    cache1 = EvidenceCache()
    cache1.store(NodeEvidence(node_id=1, visited=True, target_found=False, signal_strength=0.0, timestamp=10, walker_id=0, confidence=0.8))
    cache1.store(NodeEvidence(node_id=2, visited=True, target_found=False, signal_strength=0.0, timestamp=20, walker_id=0, confidence=0.8))

    cache2 = EvidenceCache()
    # Newer update for node 1
    cache2.store(NodeEvidence(node_id=1, visited=True, target_found=True, signal_strength=1.0, timestamp=30, walker_id=1, confidence=0.9))
    # Older record for node 2
    cache2.store(NodeEvidence(node_id=2, visited=True, target_found=False, signal_strength=0.0, timestamp=5, walker_id=1, confidence=0.5))
    # Brand new node 3
    cache2.store(NodeEvidence(node_id=3, visited=True, target_found=False, signal_strength=0.0, timestamp=25, walker_id=1, confidence=0.7))

    updated = cache1.merge(cache2)

    assert updated == 2  # Node 1 (updated) and Node 3 (inserted). Node 2 rejected as older.
    assert cache1.get_raw(1).timestamp == 30
    assert cache1.get_raw(2).timestamp == 20
    assert cache1.get_raw(3).timestamp == 25
    assert cache1.size == 3


def test_evidence_cache_capacity_eviction():
    """Verify FIFO eviction when max_capacity is reached."""
    cache = EvidenceCache(max_capacity=3)

    cache.store(NodeEvidence(node_id="A", visited=True, target_found=False, signal_strength=0.0, timestamp=1, walker_id=0, confidence=1.0))
    cache.store(NodeEvidence(node_id="B", visited=True, target_found=False, signal_strength=0.0, timestamp=2, walker_id=0, confidence=1.0))
    cache.store(NodeEvidence(node_id="C", visited=True, target_found=False, signal_strength=0.0, timestamp=3, walker_id=0, confidence=1.0))
    assert cache.size == 3

    # Add 4th item: should evict oldest "A"
    cache.store(NodeEvidence(node_id="D", visited=True, target_found=False, signal_strength=0.0, timestamp=4, walker_id=0, confidence=1.0))
    assert cache.size == 3
    assert "A" not in cache
    assert "B" in cache
    assert "C" in cache
    assert "D" in cache


def test_evidence_cache_serialization():
    """Verify EvidenceCache serialization to dict and reconstruction."""
    cache = EvidenceCache(default_lambda_hat=0.08, max_capacity=50)
    cache.store(NodeEvidence(node_id=1, visited=True, target_found=False, signal_strength=0.0, timestamp=5, walker_id=0, confidence=0.8))
    cache.store(NodeEvidence(node_id=2, visited=True, target_found=True, signal_strength=1.0, timestamp=10, walker_id=1, confidence=0.9))

    d = cache.to_dict()
    reconstructed = EvidenceCache.from_dict(d)

    assert reconstructed.default_lambda_hat == 0.08
    assert reconstructed.max_capacity == 50
    assert reconstructed.size == 2
    assert reconstructed.get_raw(1) == cache.get_raw(1)
    assert reconstructed.get_raw(2) == cache.get_raw(2)
