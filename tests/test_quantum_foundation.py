"""
Tests for Quantum Integration Foundation (Milestone 1)
======================================================
Verifies:
1. Candidate representation, factory conversions, and validations.
2. Candidate register indexing, power-of-two / non-power-of-two padding, and reverse mapping.
3. Marking rules: ThresholdMarkingRule, PredicateMarkingRule, and TargetIdMarkingRule.
4. Quantum oracle phase flipping, query tracking, and independent verification.
5. Exact Grover statevector evolution, diffusion operator, and amplitude amplification.
6. Edge cases:
   - Empty candidate set
   - Single candidate
   - No eligible candidates
   - Exactly one marked candidate
   - Multiple marked candidates
   - Duplicate candidate IDs
   - Invalid or missing evidence scores
   - Non-power-of-two candidate counts (3, 5, 6, 7, 10, etc.)
   - Candidate-index-to-node-ID round trips
   - Reproducibility with supported seeds
   - Oracle behavior for valid and invalid indices
   - Padded index boundary enforcement (no phantom nodes)
"""

import numpy as np
import pytest

from aesw.aesw.next_hop import CandidateScore
from aesw.memory.models import NodeEvidence
from aesw.quantum import (
    CandidateRegister,
    GroverSearchEngine,
    PredicateMarkingRule,
    QuantumCandidate,
    QuantumOracle,
    QuantumSearchConfig,
    QuantumSearchStatus,
    TargetIdMarkingRule,
    ThresholdMarkingRule,
    compute_optimal_grover_iterations,
)

# ==============================================================================
# 1. Candidate Representation & Factory Conversions
# ==============================================================================

def test_quantum_candidate_creation_and_validation():
    """Verify valid candidate creation and input constraints."""
    cand = QuantumCandidate(node_id="v1", score=0.85)
    assert cand.node_id == "v1"
    assert cand.score == 0.85

    # Invalid node_id
    with pytest.raises(ValueError, match="node_id must not be None"):
        QuantumCandidate(node_id=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="node_id must not be an empty string"):
        QuantumCandidate(node_id="   ")

    # Invalid score
    with pytest.raises(ValueError, match="score must be a finite float"):
        QuantumCandidate(node_id="v2", score=float("nan"))


def test_quantum_candidate_from_candidate_score():
    """Verify construction from AESW CandidateScore breakdown."""
    cs = CandidateScore(
        node_id=42,
        positive_evidence=0.75,
        novelty=0.5,
        revisit_penalty=1.0,
        delay=1.0,
        total_score=0.45,
    )
    cand = QuantumCandidate.from_candidate_score(cs)
    assert cand.node_id == 42
    assert cand.score == 0.75
    assert cand.metadata["total_score"] == 0.45
    assert cand.evidence == cs


def test_quantum_candidate_from_weighted_evidence():
    """Verify construction from time-decayed WeightedEvidence."""
    ev = NodeEvidence(
        node_id="alpha",
        visited=True,
        target_found=True,
        signal_strength=1.0,
        timestamp=10,
        walker_id=0,
        confidence=0.9,
    )
    weighted = ev.evaluate(current_time=12, lambda_hat=0.1)
    cand = QuantumCandidate.from_weighted_evidence(weighted)
    assert cand.node_id == "alpha"
    assert cand.score == pytest.approx(weighted.effective_confidence)
    assert cand.metadata["lambda_hat"] == 0.1


# ==============================================================================
# 2. Candidate Register & Non-Power-of-Two Mapping
# ==============================================================================

def test_candidate_register_power_of_two():
    """Verify register allocation for exact powers of two (K = 4, 8)."""
    cands = [QuantumCandidate(node_id=f"n{i}", score=0.1 * i) for i in range(4)]
    reg = CandidateRegister(cands)
    assert reg.num_candidates == 4
    assert reg.num_qubits == 2
    assert reg.dimension == 4
    assert reg.num_padded == 0
    assert reg.is_power_of_two is True

    # Round trip
    for i in range(4):
        assert reg.get_candidate(i).node_id == f"n{i}"
        assert reg.get_node_id(i) == f"n{i}"
        assert reg.get_index(f"n{i}") == i
        assert reg.reverse_map(i) == f"n{i}"


@pytest.mark.parametrize("k,expected_n,expected_dim,expected_padded", [
    (1, 1, 2, 1),
    (3, 2, 4, 1),
    (5, 3, 8, 3),
    (6, 3, 8, 2),
    (7, 3, 8, 1),
    (10, 4, 16, 6),
    (15, 4, 16, 1),
])
def test_candidate_register_non_power_of_two_padding(k, expected_n, expected_dim, expected_padded):
    """Verify non-power-of-two padding calculation and index safety."""
    cands = [QuantumCandidate(node_id=f"node_{i}", score=0.5) for i in range(k)]
    reg = CandidateRegister(cands)

    assert reg.num_candidates == k
    assert reg.num_qubits == expected_n
    assert reg.dimension == expected_dim
    assert reg.num_padded == expected_padded

    # Physical indices in [0, k-1] must be valid
    for i in range(k):
        assert reg.is_valid_index(i) is True
        assert reg.is_padded_index(i) is False
        assert reg.reverse_map(i) == f"node_{i}"

    # Padded indices in [k, dim-1] must be marked as padded and return None on reverse_map
    for i in range(k, expected_dim):
        assert reg.is_valid_index(i) is False
        assert reg.is_padded_index(i) is True
        assert reg.reverse_map(i) is None
        with pytest.raises(IndexError, match="padded ancillary basis state"):
            reg.get_candidate(i)
        with pytest.raises(IndexError, match="padded ancillary basis state"):
            reg.get_node_id(i)


def test_candidate_register_duplicates_rejected():
    """Verify duplicate candidate node IDs raise ValueError."""
    cands = [
        QuantumCandidate(node_id="A", score=0.5),
        QuantumCandidate(node_id="B", score=0.7),
        QuantumCandidate(node_id="A", score=0.9),
    ]
    with pytest.raises(ValueError, match="Duplicate candidate node ID 'A' detected"):
        CandidateRegister(cands)


def test_candidate_register_empty():
    """Verify empty candidate set produces safe zero-register state."""
    reg = CandidateRegister([])
    assert reg.is_empty is True
    assert reg.num_candidates == 0
    assert reg.num_qubits == 0
    assert reg.dimension == 0
    assert reg.num_padded == 0
    assert reg.reverse_map(0) is None


# ==============================================================================
# 3. Marking Rules & Eligibility Predicates
# ==============================================================================

def test_threshold_marking_rule():
    """Verify threshold marking condition and score inspection."""
    rule = ThresholdMarkingRule(threshold=0.6, score_field="score")

    c1 = QuantumCandidate(node_id="low", score=0.4)
    c2 = QuantumCandidate(node_id="edge", score=0.6)
    c3 = QuantumCandidate(node_id="high", score=0.85)

    assert rule.is_marked(c1) is False
    assert rule.is_marked(c2) is True
    assert rule.is_marked(c3) is True

    filtered = rule.filter_marked([c1, c2, c3])
    assert [c.node_id for c in filtered] == ["edge", "high"]


def test_threshold_marking_missing_score():
    """Verify strict mode raises error on missing score and non-strict returns False."""
    strict_rule = ThresholdMarkingRule(threshold=0.5, strict=True)
    c_none = QuantumCandidate(node_id="unscored", score=None)

    with pytest.raises(ValueError, match="missing required score field"):
        strict_rule.is_marked(c_none)

    lenient_rule = ThresholdMarkingRule(threshold=0.5, strict=False)
    assert lenient_rule.is_marked(c_none) is False


def test_predicate_and_target_id_marking_rules():
    """Verify PredicateMarkingRule and TargetIdMarkingRule behavior."""
    # Predicate rule
    pred_rule = PredicateMarkingRule(
        predicate=lambda c: str(c.node_id).startswith("target"),
        name="target_prefix",
    )
    assert pred_rule.is_marked(QuantumCandidate("target_1")) is True
    assert pred_rule.is_marked(QuantumCandidate("distractor_1")) is False

    # Target ID rule
    target_rule = TargetIdMarkingRule(target_ids={"node_42", "node_99"})
    assert target_rule.is_marked(QuantumCandidate("node_42")) is True
    assert target_rule.is_marked(QuantumCandidate("node_7")) is False


# ==============================================================================
# 4. Quantum Oracle & Phase Flipping
# ==============================================================================

def test_quantum_oracle_phase_flipping_and_queries():
    """Verify phase flip operator O|x> = (-1)^f(x)|x> and query counting."""
    cands = [
        QuantumCandidate(node_id="A", score=0.2),
        QuantumCandidate(node_id="B", score=0.9),  # marked
        QuantumCandidate(node_id="C", score=0.4),
        QuantumCandidate(node_id="D", score=0.8),  # marked
    ]
    reg = CandidateRegister(cands)
    rule = ThresholdMarkingRule(threshold=0.7)
    oracle = QuantumOracle(register=reg, marking_rule=rule)

    assert oracle.num_marked == 2
    assert oracle.marked_indices == (1, 3)
    assert oracle.marked_node_ids == ("B", "D")
    assert oracle.call_count == 0

    # Query individual basis states
    assert oracle.is_marked(0) is False
    assert oracle.is_marked(1) is True
    assert oracle.is_marked(2) is False
    assert oracle.is_marked(3) is True
    assert oracle.call_count == 4

    # Test phase vector evaluation
    state = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    flipped = oracle.apply_phase_flip(state)

    # 1 oracle gate applied to statevector increments call_count by 1
    assert oracle.call_count == 5
    # Amplitudes at indices 1 and 3 should be inverted to -0.5
    expected = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)
    np.testing.assert_allclose(flipped, expected)


def test_quantum_oracle_padded_indices_never_marked():
    """Verify that padded states (i >= K) in non-power-of-two registers are never marked."""
    # K = 3 candidates -> dimension 4 (index 3 is padded)
    cands = [
        QuantumCandidate(node_id="X", score=0.9),  # marked
        QuantumCandidate(node_id="Y", score=0.1),
        QuantumCandidate(node_id="Z", score=0.9),  # marked
    ]
    reg = CandidateRegister(cands)
    rule = ThresholdMarkingRule(threshold=0.5)
    oracle = QuantumOracle(register=reg, marking_rule=rule)

    assert reg.dimension == 4
    assert oracle.num_marked == 2
    assert oracle.marked_indices == (0, 2)

    # Index 3 is a padded dummy state
    assert oracle.is_marked(3) is False
    assert oracle.evaluate_phase(3) == 1.0

    # Out of bounds indices
    with pytest.raises(IndexError, match="out of basis bounds"):
        oracle.is_marked(4)
    with pytest.raises(IndexError, match="out of basis bounds"):
        oracle.is_marked(-1)


# ==============================================================================
# 5. Grover Search Engine & Amplitude Amplification
# ==============================================================================

def test_diffusion_operator_inversion_about_mean():
    """Verify Grover diffusion D = 2|s><s| - I."""
    engine = GroverSearchEngine()
    # For a 4-dimensional state [0.5, -0.5, 0.5, 0.5]
    # mean = (0.5 - 0.5 + 0.5 + 0.5) / 4 = 1.0 / 4 = 0.25
    # (D v)_i = 2 * 0.25 - v_i = 0.5 - v_i
    # For 0.5: 0.5 - 0.5 = 0.0
    # For -0.5: 0.5 - (-0.5) = 1.0
    state = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    diffused = engine.apply_diffusion(state)
    expected = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(diffused, expected, atol=1e-12)


def test_grover_search_single_marked_target_amplification():
    """Verify Grover search amplifies a single marked candidate to near 100% probability."""
    # K = 4 candidates, exactly 1 marked ("target")
    cands = [
        QuantumCandidate(node_id="node_0", score=0.1),
        QuantumCandidate(node_id="node_1", score=0.1),
        QuantumCandidate(node_id="target", score=0.9),  # index 2
        QuantumCandidate(node_id="node_3", score=0.1),
    ]
    reg = CandidateRegister(cands)
    rule = ThresholdMarkingRule(threshold=0.5)

    engine = GroverSearchEngine(QuantumSearchConfig(seed=12345))
    res = engine.search(reg, rule)

    assert res.success is True
    assert res.status == QuantumSearchStatus.SUCCESS
    assert res.selected_candidate == "target"
    assert res.selected_index == 2
    assert res.marked_candidates == ("target",)
    assert res.iteration_count == 1  # For N=4, |M|=1, R = 1 gives 100% target probability!
    assert res.probabilities[2] == pytest.approx(1.0, abs=1e-6)


def test_grover_search_multiple_marked_candidates():
    """Verify Grover search correctly handles multiple marked targets."""
    # K = 8 candidates, 2 marked
    cands = [
        QuantumCandidate(node_id=f"n{i}", score=0.85 if i in (2, 5) else 0.1)
        for i in range(8)
    ]
    reg = CandidateRegister(cands)
    rule = ThresholdMarkingRule(threshold=0.5)

    engine = GroverSearchEngine(QuantumSearchConfig(seed=42))
    res = engine.search(reg, rule)

    assert res.success is True
    assert res.status == QuantumSearchStatus.SUCCESS
    assert res.selected_candidate in ("n2", "n5")
    assert set(res.marked_candidates) == {"n2", "n5"}
    assert res.iteration_count == compute_optimal_grover_iterations(8, 2)
    # Total probability concentrated on marked states
    total_marked_prob = res.probabilities[2] + res.probabilities[5]
    assert total_marked_prob > 0.95


def test_grover_search_non_power_of_two_candidates():
    """Verify Grover search on non-power-of-two candidate set (K = 6)."""
    # K = 6 candidates, dimension N = 8 (indices 6, 7 padded)
    cands = [
        QuantumCandidate(node_id=f"n{i}", score=0.9 if i == 4 else 0.1)
        for i in range(6)
    ]
    reg = CandidateRegister(cands)
    rule = ThresholdMarkingRule(threshold=0.5)

    engine = GroverSearchEngine(QuantumSearchConfig(seed=999))
    res = engine.search(reg, rule)

    assert res.success is True
    assert res.status == QuantumSearchStatus.SUCCESS
    assert res.selected_candidate == "n4"
    assert res.selected_index == 4
    # Padded indices 6 and 7 must never be selected
    assert res.selected_index < 6


# ==============================================================================
# 6. Edge Cases & Safety Constraints
# ==============================================================================

def test_grover_search_empty_candidates():
    """Edge Case: Empty candidate set must return EMPTY_CANDIDATES without failing."""
    reg = CandidateRegister([])
    rule = ThresholdMarkingRule(threshold=0.5)
    engine = GroverSearchEngine()

    res = engine.search(reg, rule)
    assert res.success is False
    assert res.status == QuantumSearchStatus.EMPTY_CANDIDATES
    assert res.selected_candidate is None
    assert res.iteration_count == 0


def test_grover_search_single_candidate():
    """Edge Case: Single candidate (K = 1)."""
    # 1. Candidate is marked
    reg_marked = CandidateRegister([QuantumCandidate(node_id="only_node", score=0.9)])
    rule = ThresholdMarkingRule(threshold=0.5)
    engine = GroverSearchEngine(QuantumSearchConfig(seed=42))

    res_marked = engine.search(reg_marked, rule)
    assert res_marked.success is True
    assert res_marked.selected_candidate == "only_node"
    assert res_marked.status == QuantumSearchStatus.SUCCESS

    # 2. Candidate is NOT marked
    reg_unmarked = CandidateRegister([QuantumCandidate(node_id="only_node", score=0.1)])
    res_unmarked = engine.search(reg_unmarked, rule)
    assert res_unmarked.success is False
    assert res_unmarked.status == QuantumSearchStatus.NO_MARKED_CANDIDATES
    assert res_unmarked.selected_candidate is None


def test_grover_search_no_eligible_candidates():
    """Edge Case: Candidates exist, but none satisfy the marking rule."""
    cands = [QuantumCandidate(node_id=f"n{i}", score=0.2) for i in range(5)]
    reg = CandidateRegister(cands)
    rule = ThresholdMarkingRule(threshold=0.8)

    engine = GroverSearchEngine()
    res = engine.search(reg, rule)

    assert res.success is False
    assert res.status == QuantumSearchStatus.NO_MARKED_CANDIDATES
    assert res.selected_candidate is None
    assert res.marked_candidates == ()
    assert res.iteration_count == 0


def test_reproducibility_with_seed():
    """Verify bit-for-bit reproducibility of search results given the same RNG seed."""
    cands = [
        QuantumCandidate(node_id=f"c_{i}", score=0.85 if i in (1, 3, 6) else 0.2)
        for i in range(7)
    ]
    reg = CandidateRegister(cands)
    rule = ThresholdMarkingRule(threshold=0.5)

    res1 = GroverSearchEngine(QuantumSearchConfig(seed=777)).search(reg, rule)
    res2 = GroverSearchEngine(QuantumSearchConfig(seed=777)).search(reg, rule)

    assert res1.selected_candidate == res2.selected_candidate
    assert res1.selected_index == res2.selected_index
    assert res1.probabilities == res2.probabilities
    assert res1.status == res2.status
