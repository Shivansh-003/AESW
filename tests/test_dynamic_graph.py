"""
Comprehensive Test Suite for Dynamic Graph Engine.
Verifies:
- Edge state representation and initialization (ALL_ON, ALL_OFF, STATIONARY)
- Synchronous ON <-> OFF Markovian transition dynamics
- Extreme transition boundaries (p_on=1, p_on=0, p_off=1, p_off=0)
- STATIC regime invariant G_0 = G_1 = ... = G_T
- Static topology immutability throughout time evolution
- Active graph view and active neighbor queries
- Snapshot creation and transition statistics tracking
- Compatibility across all synthetic graph families (ER, BA, WS, Grid, RGG)
- Deterministic trajectory reproduction via seeds
- Statistical transition sanity assertions
"""

import pytest
import numpy as np

from aesw.environment.types import EdgeState, DynamicRegime
from aesw.environment.problem import DynamicsSpecification
from aesw.graph import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_grid_with_obstacles,
    generate_random_geometric,
)
from aesw.dynamics.types import InitializationPolicy
from aesw.dynamics.engine import DynamicGraphState
from aesw.dynamics.view import ActiveGraphView
from aesw.dynamics.factory import (
    create_dynamic_graph,
    create_dynamic_graph_from_spec,
    create_dynamic_graph_from_config,
    REGIME_PRESETS,
)


# ==========================================
# A. Parameter Validation
# ==========================================

def test_dynamics_parameter_validation():
    """Verify validation of p_on, p_off, and regime arguments."""
    g = generate_erdos_renyi(n=10, p=0.3, seed=42)

    with pytest.raises(ValueError, match="p_on must be in \\[0.0, 1.0\\]"):
        DynamicGraphState(static_graph=g, p_on=-0.1)

    with pytest.raises(ValueError, match="p_on must be in \\[0.0, 1.0\\]"):
        DynamicGraphState(static_graph=g, p_on=1.5)

    with pytest.raises(ValueError, match="p_off must be in \\[0.0, 1.0\\]"):
        DynamicGraphState(static_graph=g, p_off=-0.05)

    with pytest.raises(ValueError, match="p_off must be in \\[0.0, 1.0\\]"):
        DynamicGraphState(static_graph=g, p_off=1.05)

    with pytest.raises(ValueError, match="static_graph must not be None"):
        DynamicGraphState(static_graph=None)

    with pytest.raises(TypeError, match="regime must be a DynamicRegime"):
        DynamicGraphState(static_graph=g, regime="INVALID_REGIME")  # type: ignore


# ==========================================
# B. Initialization Policies
# ==========================================

def test_initialization_all_on_and_all_off():
    """Verify ALL_ON and ALL_OFF edge initialization policies."""
    g = generate_erdos_renyi(n=20, p=0.3, seed=42)
    m = g.edge_count

    # ALL_ON
    dyn_on = DynamicGraphState(static_graph=g, initialization_policy=InitializationPolicy.ALL_ON)
    assert dyn_on.active_edge_count() == m
    assert dyn_on.inactive_edge_count() == 0

    # ALL_OFF
    dyn_off = DynamicGraphState(static_graph=g, initialization_policy=InitializationPolicy.ALL_OFF)
    assert dyn_off.active_edge_count() == 0
    assert dyn_off.inactive_edge_count() == m


def test_initialization_stationary():
    """Verify stationary initialization policy pi_ON = p_on / (p_on + p_off)."""
    g = generate_erdos_renyi(n=50, p=0.5, seed=42)
    # p_on = 0.8, p_off = 0.2 -> pi_on = 0.8 / 1.0 = 0.8
    dyn = DynamicGraphState(
        static_graph=g,
        p_on=0.8,
        p_off=0.2,
        initialization_policy=InitializationPolicy.STATIONARY,
        seed=100,
    )
    # Approximately 80% of edges should be ON
    ratio = dyn.active_edge_count() / g.edge_count
    assert 0.65 <= ratio <= 0.95


# ==========================================
# C. STATIC Regime Invariant
# ==========================================

def test_static_regime_invariant():
    """Verify that STATIC regime guarantees G_0 = G_1 = ... = G_T with 0 transitions."""
    g = generate_erdos_renyi(n=30, p=0.2, seed=42)
    dyn = DynamicGraphState(
        static_graph=g,
        regime=DynamicRegime.STATIC,
        p_on=0.0,
        p_off=0.0,
        initialization_policy=InitializationPolicy.ALL_ON,
    )

    initial_states = dyn.edge_states

    for _ in range(10):
        stats = dyn.advance()
        assert stats.on_to_off == 0
        assert stats.off_to_on == 0
        assert stats.total_transitions == 0
        assert dyn.edge_states == initial_states


# ==========================================
# D. Deterministic Transition Boundaries
# ==========================================

def test_p_off_equals_one():
    """Verify p_off = 1: all currently ON edges immediately turn OFF."""
    g = generate_erdos_renyi(n=20, p=0.3, seed=42)
    dyn = DynamicGraphState(
        static_graph=g,
        p_on=0.0,
        p_off=1.0,
        initialization_policy=InitializationPolicy.ALL_ON,
    )
    assert dyn.active_edge_count() == g.edge_count

    stats = dyn.advance()
    assert stats.on_to_off == g.edge_count
    assert dyn.active_edge_count() == 0
    assert dyn.inactive_edge_count() == g.edge_count


def test_p_on_equals_one():
    """Verify p_on = 1: all currently OFF edges immediately turn ON."""
    g = generate_erdos_renyi(n=20, p=0.3, seed=42)
    dyn = DynamicGraphState(
        static_graph=g,
        p_on=1.0,
        p_off=0.0,
        initialization_policy=InitializationPolicy.ALL_OFF,
    )
    assert dyn.active_edge_count() == 0

    stats = dyn.advance()
    assert stats.off_to_on == g.edge_count
    assert dyn.active_edge_count() == g.edge_count
    assert dyn.inactive_edge_count() == 0


def test_p_off_zero_prevents_deactivation():
    """Verify p_off = 0: no active edge ever deactivates."""
    g = generate_erdos_renyi(n=20, p=0.3, seed=42)
    dyn = DynamicGraphState(
        static_graph=g,
        p_on=0.5,
        p_off=0.0,
        initialization_policy=InitializationPolicy.ALL_ON,
        seed=42,
    )
    for _ in range(10):
        stats = dyn.advance()
        assert stats.on_to_off == 0
        assert dyn.active_edge_count() == g.edge_count


def test_p_on_zero_prevents_activation():
    """Verify p_on = 0: no inactive edge ever activates."""
    g = generate_erdos_renyi(n=20, p=0.3, seed=42)
    dyn = DynamicGraphState(
        static_graph=g,
        p_on=0.0,
        p_off=0.5,
        initialization_policy=InitializationPolicy.ALL_OFF,
        seed=42,
    )
    for _ in range(10):
        stats = dyn.advance()
        assert stats.off_to_on == 0
        assert dyn.active_edge_count() == 0


# ==========================================
# E. Determinism and Seed Repeatability
# ==========================================

def test_dynamic_trajectory_determinism():
    """Verify identical seeds produce identical state trajectories, while different seeds differ."""
    g = generate_barabasi_albert(n=30, m=2, seed=42)

    dyn1 = DynamicGraphState(static_graph=g, p_on=0.2, p_off=0.2, seed=123)
    dyn2 = DynamicGraphState(static_graph=g, p_on=0.2, p_off=0.2, seed=123)
    dyn3 = DynamicGraphState(static_graph=g, p_on=0.2, p_off=0.2, seed=999)

    seq1 = []
    seq2 = []
    seq3 = []

    for _ in range(5):
        dyn1.advance()
        dyn2.advance()
        dyn3.advance()
        seq1.append(dyn1.active_edge_count())
        seq2.append(dyn2.active_edge_count())
        seq3.append(dyn3.active_edge_count())

    assert seq1 == seq2
    assert seq1 != seq3
    assert dyn1.edge_states == dyn2.edge_states


# ==========================================
# F. Static Topology Immutability
# ==========================================

def test_static_topology_immutability():
    """Verify that static GraphInstance remains completely unmodified after dynamic transitions."""
    g = generate_watts_strogatz(n=25, k=4, p=0.1, seed=42)
    initial_nodes = list(g.nodes.keys())
    initial_edges = tuple(g.edges)

    dyn = DynamicGraphState(static_graph=g, p_on=0.3, p_off=0.3, seed=42)
    for _ in range(15):
        dyn.advance()

    # Assert underlying static graph properties remain identical
    assert list(g.nodes.keys()) == initial_nodes
    assert tuple(g.edges) == initial_edges
    assert dyn.static_graph.node_count == 25


# ==========================================
# G. Active Graph View & Queries
# ==========================================

def test_active_graph_view_and_queries():
    """Verify ActiveGraphView queries match DynamicGraphState active edges."""
    g = generate_grid_with_obstacles(rows=5, cols=5, obstacle_ratio=0.1, seed=42)
    dyn = DynamicGraphState(static_graph=g, p_on=0.2, p_off=0.2, seed=42)
    view = ActiveGraphView(dyn)

    assert view.time == 0
    assert view.node_count == g.node_count
    assert view.active_edge_count == dyn.active_edge_count()

    dyn.advance()
    assert view.time == 1
    assert view.active_edge_count == dyn.active_edge_count()

    # Query active neighbors of a valid node
    sample_node = list(g.nodes.keys())[0]
    active_nbrs = view.active_neighbors(sample_node)
    all_nbrs = g.neighbors(sample_node)
    assert set(active_nbrs).issubset(set(all_nbrs))
    assert view.active_degree(sample_node) == len(active_nbrs)

    for nbr in active_nbrs:
        assert view.has_active_edge(sample_node, nbr) is True
        assert dyn.is_edge_active(sample_node, nbr) is True


# ==========================================
# H. Snapshot & Statistics Tracking
# ==========================================

def test_snapshots_and_statistics():
    """Verify snapshot immutability and transition statistics history."""
    g = generate_erdos_renyi(n=25, p=0.2, seed=42)
    dyn = DynamicGraphState(static_graph=g, p_on=0.1, p_off=0.05, seed=42)

    s0 = dyn.snapshot()
    assert s0.time == 0
    assert s0.active_edge_count == g.edge_count

    stats1 = dyn.advance()
    assert stats1.time == 1
    s1 = dyn.snapshot()
    assert s1.time == 1
    assert s1.active_edge_count == stats1.active_edge_count

    history = dyn.transition_history()
    assert len(history) == 1
    assert history[0] == stats1


# ==========================================
# I. Compatibility Across All Graph Families
# ==========================================

@pytest.mark.parametrize("generator_fn", [
    lambda: generate_erdos_renyi(n=20, p=0.2, seed=42),
    lambda: generate_barabasi_albert(n=20, m=2, seed=42),
    lambda: generate_watts_strogatz(n=20, k=4, p=0.1, seed=42),
    lambda: generate_grid_with_obstacles(rows=4, cols=5, obstacle_ratio=0.1, seed=42),
    lambda: generate_random_geometric(n=20, radius=0.3, seed=42),
])
def test_dynamic_engine_across_all_graph_families(generator_fn):
    """Verify dynamic transition engine executes consistently on all supported graph families."""
    g = generator_fn()
    dyn = DynamicGraphState(static_graph=g, p_on=0.1, p_off=0.1, seed=42)
    assert dyn.time == 0
    stats = dyn.advance()
    assert stats.time == 1
    assert dyn.active_edge_count() + dyn.inactive_edge_count() == g.edge_count


# ==========================================
# J. Factory Functions and Named Regimes
# ==========================================

def test_create_dynamic_graph_presets():
    """Verify create_dynamic_graph factory and regime presets."""
    g = generate_erdos_renyi(n=20, p=0.2, seed=42)

    for regime in DynamicRegime:
        dyn = create_dynamic_graph(static_graph=g, regime=regime)
        expected_p_on, expected_p_off = REGIME_PRESETS[regime]
        assert dyn.regime == regime
        assert dyn.p_on == expected_p_on
        assert dyn.p_off == expected_p_off

    # Test from DynamicsSpecification
    spec = DynamicsSpecification(regime=DynamicRegime.FAST, p_on=0.15, p_off=0.08)
    dyn_spec = create_dynamic_graph_from_spec(g, spec)
    assert dyn_spec.p_on == 0.15
    assert dyn_spec.p_off == 0.08

    # Test from config dict
    cfg = {"dynamics": {"regime": "SLOW", "p_on": 0.02, "p_off": 0.01}}
    dyn_cfg = create_dynamic_graph_from_config(g, cfg)
    assert dyn_cfg.regime == DynamicRegime.SLOW
    assert dyn_cfg.p_on == 0.02
    assert dyn_cfg.p_off == 0.01


# ==========================================
# K. Statistical Transition Sanity Assertion
# ==========================================

def test_statistical_transition_sanity():
    """Verify stochastic transition frequencies converge toward expected probabilities."""
    g = generate_erdos_renyi(n=100, p=0.6, seed=42)  # Large graph: ~3000 edges
    p_off = 0.2
    p_on = 0.3

    dyn = DynamicGraphState(
        static_graph=g,
        p_on=p_on,
        p_off=p_off,
        initialization_policy=InitializationPolicy.ALL_ON,
        seed=1234,
    )

    # First step: all edges are ON, so on_to_off / total_edges should be approx p_off
    stats = dyn.advance()
    observed_p_off = stats.on_to_off / g.edge_count
    # Generous tolerance to ensure robust pass (e.g. within +/- 0.05)
    assert abs(observed_p_off - p_off) < 0.05
