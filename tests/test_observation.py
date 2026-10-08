"""
Tests for Observation Layer & Partial Visibility Engine
========================================================
Validates the Controlled Observation Principle, information firewall boundaries,
neighbor checking budget B, detection signal projection, local history snapshots,
PRNG stream isolation, and absence vs unknown distinction.
"""

import copy
import pytest
import numpy as np

from aesw.graph import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_grid_with_obstacles,
    generate_random_geometric,
)
from aesw.dynamics import (
    create_dynamic_graph,
    DynamicRegime,
    ActiveGraphView,
)
from aesw.environment.types import EdgeState, DetectionOutcome, TargetMode
from aesw.environment.models import (
    WalkerState,
    TargetState,
    Node,
    Edge,
)
from aesw.environment.problem import (
    ObservationSpecification,
    DetectionSpecification,
    TargetSpecification,
)
from aesw.environment.detection import DetectionEngine, DetectionResult
from aesw.environment.target import TargetEngine
from aesw.environment.state import GroundTruthState
from aesw.environment.observation import (
    Observation,
    ObservedEdgeInfo,
    TargetSignalObservation,
    LocalObservationHistory,
)
from aesw.environment.builder import ObservationBuilder


# ==========================================
# A. Current Node & Position Integrity
# ==========================================

def test_observation_reports_correct_walker_position():
    """Verify that observation reports the exact current node of the walker."""
    g = generate_erdos_renyi(n=20, p=0.3, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)
    walker = WalkerState(walker_id=1, current_node=5)

    builder = ObservationBuilder(neighbor_budget=4, seed=10)
    obs = builder.build_observation(walker=walker, active_graph=view, target_node=10, time=0)

    assert obs.walker_id == 1
    assert obs.current_node == 5
    assert obs.time == 0


# ==========================================
# B. Local Neighbors & C. OFF Edges
# ==========================================

def test_local_neighbors_and_off_edges_omission():
    """Verify that only currently active (ON) neighbors appear as available links."""
    g = generate_erdos_renyi(n=10, p=0.5, seed=42)
    # Start with ALL_ON, then manually turn an edge OFF
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    node = 0
    active_nbrs = view.active_neighbors(node)
    assert len(active_nbrs) > 0

    builder = ObservationBuilder(neighbor_budget=10, seed=42)
    obs = builder.build_observation(
        walker=WalkerState(walker_id=0, current_node=node),
        active_graph=view,
        target_node=9,
        time=0,
    )

    # Every neighbor in checked_neighbors must be an active neighbor in G_t
    for nbr in obs.checked_neighbors:
        assert nbr in active_nbrs
        assert obs.observed_edges[nbr].state == EdgeState.ON
        assert obs.is_edge_known_active(nbr) is True
        assert obs.is_edge_known_inactive(nbr) is False
        assert obs.is_edge_unknown(nbr) is False

    # Now turn an edge OFF and ensure it no longer appears in default active candidate observation
    target_nbr = active_nbrs[0]
    dyn.set_edge_state(node, target_nbr, EdgeState.OFF)
    new_view = ActiveGraphView(dyn)
    new_active_nbrs = new_view.active_neighbors(node)
    assert target_nbr not in new_active_nbrs

    obs2 = builder.build_observation(
        walker=WalkerState(walker_id=0, current_node=node),
        active_graph=new_view,
        target_node=9,
        time=1,
    )
    assert target_nbr not in obs2.checked_neighbors
    assert target_nbr not in obs2.observed_edges
    assert obs2.is_edge_unknown(target_nbr) is True


# ==========================================
# D. Multi-Hop Isolation
# ==========================================

def test_multi_hop_isolation():
    """Verify that nodes 2 or more hops away never appear in observation."""
    # A - B - C - D path
    # In a path graph, if walker is at B, it can only see A and C; D must never appear
    g = generate_watts_strogatz(n=10, k=2, p=0.0, seed=42)  # Ring lattice: degree 2
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    # Ring connections for node 5 are 4 and 6
    walker = WalkerState(walker_id=0, current_node=5)
    builder = ObservationBuilder(neighbor_budget=5, seed=42)
    obs = builder.build_observation(walker=walker, active_graph=view, target_node=0, time=0)

    assert set(obs.checked_neighbors) == {4, 6}
    for non_neighbor in [0, 1, 2, 3, 7, 8, 9]:
        assert non_neighbor not in obs.checked_neighbors
        assert non_neighbor not in obs.observed_edges
        assert obs.is_edge_unknown(non_neighbor) is True


# ==========================================
# E. Neighbor Checking Budget B & F. No Duplicates
# ==========================================

def test_neighbor_budget_under_and_over_threshold():
    """Verify that budget B limits observations when degree > B, and captures all when degree <= B."""
    g = generate_barabasi_albert(n=30, m=3, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    # Find a hub node with degree > 4
    hub_node = max(view.nodes.keys(), key=lambda n: len(view.active_neighbors(n)))
    hub_degree = len(view.active_neighbors(hub_node))
    assert hub_degree > 4

    # Case 1: degree > B (B = 2)
    builder_b2 = ObservationBuilder(neighbor_budget=2, seed=42)
    obs_b2 = builder_b2.build_observation(
        walker=WalkerState(walker_id=0, current_node=hub_node),
        active_graph=view,
        target_node=0,
        time=0,
    )
    assert len(obs_b2.checked_neighbors) == 2
    assert obs_b2.neighbor_budget_used == 2
    assert len(obs_b2.observed_edges) == 2
    assert len(set(obs_b2.checked_neighbors)) == 2  # Unique, no duplicates
    assert len(obs_b2.visible_neighbors) == 2

    # Case 2: degree <= B (B = hub_degree + 5)
    builder_large = ObservationBuilder(neighbor_budget=hub_degree + 5, seed=42)
    obs_large = builder_large.build_observation(
        walker=WalkerState(walker_id=0, current_node=hub_node),
        active_graph=view,
        target_node=0,
        time=0,
    )
    assert len(obs_large.checked_neighbors) == hub_degree
    assert obs_large.neighbor_budget_used == hub_degree
    assert len(obs_large.observed_edges) == hub_degree
    assert set(obs_large.checked_neighbors) == set(view.active_neighbors(hub_node))


# ==========================================
# G. Determinism & H. RNG Stream Decoupling
# ==========================================

def test_observation_determinism_and_rng_isolation():
    """Verify that same observation seed yields identical sampled subsets, and changing seed does not affect target."""
    g = generate_erdos_renyi(n=30, p=0.4, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    node = 0
    walker = WalkerState(walker_id=0, current_node=node)

    # 1. Determinism
    builder_1 = ObservationBuilder(neighbor_budget=3, seed=123)
    builder_2 = ObservationBuilder(neighbor_budget=3, seed=123)
    obs_1 = builder_1.build_observation(walker=walker, active_graph=view, target_node=10, time=0)
    obs_2 = builder_2.build_observation(walker=walker, active_graph=view, target_node=10, time=0)
    assert obs_1.checked_neighbors == obs_2.checked_neighbors

    # 2. RNG Decoupling: Changing observation seed does not alter TargetEngine locomotion
    t_spec = TargetSpecification(mode=TargetMode.MOVING, p_move=0.5, initial_node=15)
    target_engine_a = TargetEngine(spec=t_spec, seed=999)
    target_engine_b = TargetEngine(spec=t_spec, seed=999)

    builder_seed_a = ObservationBuilder(neighbor_budget=2, seed=111)
    builder_seed_b = ObservationBuilder(neighbor_budget=2, seed=222)

    # Step target and observe across multiple timesteps
    for step in range(5):
        state_a = target_engine_a.step(view)
        state_b = target_engine_b.step(view)
        # Even with different observation builders running
        _ = builder_seed_a.build_observation(walker, view, state_a.current_node, step)
        _ = builder_seed_b.build_observation(walker, view, state_b.current_node, step)
        assert state_a.current_node == state_b.current_node


# ==========================================
# I. Detection Integration & J/K. Ground-Truth Hiding
# ==========================================

def test_detection_signal_projection_and_hiding():
    """Verify that walker receives strictly binary signal and cannot access ground-truth outcome."""
    g = generate_erdos_renyi(n=10, p=0.5, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    det_spec = DetectionSpecification(signal_radius=1, p_d=0.9, p_fa=0.05)
    det_engine = DetectionEngine(spec=det_spec, seed=42)

    builder = ObservationBuilder(neighbor_budget=4, detection_engine=det_engine, seed=42)
    walker = WalkerState(walker_id=0, current_node=0)

    # Place target at node 0 (distance 0 <= radius 1)
    obs = builder.build_observation(walker=walker, active_graph=view, target_node=0, time=0)

    # 1. Binary signal available
    assert isinstance(obs.target_signal.detected, bool)
    assert isinstance(obs.detected, bool)
    assert obs.target_signal.detected == obs.detected

    # 2. Outcome category is strictly None on walker observation
    assert obs.target_signal.outcome_category is None

    # 3. No leakage attributes on observation
    assert not hasattr(obs, "outcome")
    assert not hasattr(obs, "outcome_category")
    assert not hasattr(obs, "true_outcome")
    assert not hasattr(obs, "within_radius")
    assert not hasattr(obs, "graph_distance")
    assert not hasattr(obs, "target_distance")
    assert not hasattr(obs, "target_node")


# ==========================================
# L, M, N, O. Architectural Information Leakage Tests
# ==========================================

def test_architectural_information_firewall_introspection():
    """Introspection test ensuring Observation contains zero forbidden ground-truth fields."""
    g = generate_erdos_renyi(n=15, p=0.3, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    builder = ObservationBuilder(neighbor_budget=3, seed=42)
    walker = WalkerState(walker_id=0, current_node=2)
    obs = builder.build_observation(walker=walker, active_graph=view, target_node=8, time=0)

    forbidden_attributes = [
        "graph",
        "full_graph",
        "static_graph",
        "dynamic_state",
        "ground_truth",
        "environment",
        "target",
        "target_node",
        "target_state",
        "target_position",
        "target_distance",
        "true_distance",
        "graph_distance",
        "outcome",
        "true_outcome",
        "outcome_category",
        "all_nodes",
        "all_edges",
        "future_edges",
        "next_graph",
    ]

    for attr in forbidden_attributes:
        assert not hasattr(obs, attr), f"Observation leaked forbidden attribute: '{attr}'"

    # Introspect observed edges
    for nbr, edge_info in obs.observed_edges.items():
        assert not hasattr(edge_info, "future_state")
        assert not hasattr(edge_info, "transition_prob")
        assert not hasattr(edge_info, "weight")


# ==========================================
# P. Local History & Immutability / Snapshot Semantics
# ==========================================

def test_local_history_snapshot_immutability():
    """Verify that earlier observations remain unchanged when environment and walker state mutate."""
    g = generate_erdos_renyi(n=10, p=0.5, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.MEDIUM, seed=42)
    view = ActiveGraphView(dyn)

    builder = ObservationBuilder(neighbor_budget=3, seed=42)
    walker = WalkerState(walker_id=0, current_node=0)

    # Initial observation at t=0
    history = LocalObservationHistory()
    obs_0, history = builder.observe_and_update_history(
        walker=walker,
        active_graph=view,
        target_node=5,
        time=0,
        history=history,
    )

    initial_checked = copy.deepcopy(obs_0.checked_neighbors)
    initial_edge_states = {k: v.state for k, v in obs_0.observed_edges.items()}

    # Mutate environment: advance dynamic graph by multiple steps
    for _ in range(3):
        dyn.advance()
    mutated_view = ActiveGraphView(dyn)

    # Move walker to node 1 at t=3
    walker_next = WalkerState(walker_id=0, current_node=1)
    obs_1, history = builder.observe_and_update_history(
        walker=walker_next,
        active_graph=mutated_view,
        target_node=7,
        time=3,
        history=history,
    )

    # Verify obs_0 in history is completely unchanged
    assert history[0].current_node == 0
    assert history[0].time == 0
    assert history[0].checked_neighbors == initial_checked
    for k, v in history[0].observed_edges.items():
        assert v.state == initial_edge_states[k]

    # Verify history tracking
    assert len(history) == 2
    assert history.visited_nodes == (0, 1)
    assert history.latest == obs_1


# ==========================================
# Q. Unknown vs Absent Distinction
# ==========================================

def test_unknown_vs_absent_distinction():
    """Verify that uninspected nodes return unknown, rather than absent."""
    g = generate_erdos_renyi(n=20, p=0.2, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    # Inspect candidate subset where one edge is ON and one edge is OFF
    node = 0
    active_nbrs = view.active_neighbors(node)
    assert len(active_nbrs) > 0
    active_nbr = active_nbrs[0]

    # Find a non-neighbor
    non_nbr = [n for n in range(20) if n != node and n not in active_nbrs][0]

    builder = ObservationBuilder(neighbor_budget=1, seed=42)
    # Check specifically active_nbr
    obs = builder.build_observation(
        walker=WalkerState(walker_id=0, current_node=node),
        active_graph=view,
        target_node=15,
        time=0,
        candidate_neighbors=[active_nbr],
    )

    # active_nbr is known active
    assert obs.is_edge_known_active(active_nbr) is True
    assert obs.is_edge_known_inactive(active_nbr) is False
    assert obs.is_edge_unknown(active_nbr) is False
    assert obs.get_edge_status(active_nbr) == EdgeState.ON

    # non_nbr was not inspected: it is UNKNOWN, NOT absent!
    assert obs.is_edge_unknown(non_nbr) is True
    assert obs.is_edge_known_inactive(non_nbr) is False
    assert obs.get_edge_status(non_nbr) is None


# ==========================================
# R. Cross-Family Graph Compatibility
# ==========================================

@pytest.mark.parametrize("generator_fn", [
    lambda: generate_erdos_renyi(n=25, p=0.2, seed=42),
    lambda: generate_barabasi_albert(n=25, m=2, seed=42),
    lambda: generate_watts_strogatz(n=25, k=4, p=0.1, seed=42),
    lambda: generate_grid_with_obstacles(rows=5, cols=5, obstacle_ratio=0.1, seed=42),
    lambda: generate_random_geometric(n=25, radius=0.4, seed=42),
])
def test_observation_builder_across_graph_families(generator_fn):
    """Verify ObservationBuilder functions properly across all synthetic graph families."""
    g = generator_fn()
    dyn = create_dynamic_graph(g, regime=DynamicRegime.SLOW, seed=42)
    view = ActiveGraphView(dyn)

    det_spec = DetectionSpecification(signal_radius=1, p_d=0.85, p_fa=0.05)
    det_engine = DetectionEngine(spec=det_spec, seed=42)

    builder = ObservationBuilder(neighbor_budget=3, detection_engine=det_engine, seed=42)

    sample_node = list(view.nodes.keys())[0]
    walker = WalkerState(walker_id=0, current_node=sample_node)
    target_node = list(view.nodes.keys())[-1]

    obs = builder.build_observation(walker=walker, active_graph=view, target_node=target_node, time=0)

    assert obs.current_node == sample_node
    assert obs.neighbor_budget_used <= 3
    assert len(obs.checked_neighbors) <= 3
    assert isinstance(obs.detected, bool)
    assert obs.target_signal.outcome_category is None


# ==========================================
# S. Dynamic Graph Step Compatibility & GroundTruthState Projection
# ==========================================

def test_observation_from_ground_truth_state():
    """Verify build_from_ground_truth projects GroundTruthState into Observation cleanly."""
    g = generate_erdos_renyi(n=10, p=0.4, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.MEDIUM, seed=42)

    target_state = TargetState(current_node=9, p_move=0.1, time=0)
    walker_state = WalkerState(walker_id=0, current_node=1)

    gt_state = GroundTruthState(
        time=0,
        nodes=g.nodes,
        edges=tuple(g.edges),
        active_edge_states=dyn.edge_states,
        target_state=target_state,
        walker_states={0: walker_state},
    )

    builder = ObservationBuilder(neighbor_budget=4, seed=42)
    obs = builder.build_from_ground_truth(walker_id=0, ground_truth=gt_state)

    assert obs.walker_id == 0
    assert obs.current_node == 1
    assert obs.time == 0
    assert not hasattr(obs, "ground_truth")
    assert not hasattr(obs, "target_state")


def test_missed_detection_and_false_alarm_indistinguishability():
    """Verify that walker receiving detected=True or False cannot distinguish true/false classifications."""
    g = generate_erdos_renyi(n=10, p=0.5, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    # Observer at 0, target at 0 (inside radius) with p_d = 0.0 -> Guaranteed MISSED detection
    det_spec_miss = DetectionSpecification(signal_radius=1, p_d=0.0, p_fa=0.0)
    engine_miss = DetectionEngine(spec=det_spec_miss, seed=42)
    builder_miss = ObservationBuilder(neighbor_budget=2, detection_engine=engine_miss, seed=42)

    walker = WalkerState(walker_id=0, current_node=0)
    obs_miss = builder_miss.build_observation(walker=walker, active_graph=view, target_node=0, time=0)

    # Observer at 0, target at 9 (outside radius) with p_fa = 0.0 -> Guaranteed TRUE NEGATIVE
    obs_neg = builder_miss.build_observation(walker=walker, active_graph=view, target_node=9, time=0)

    # To the walker, BOTH signals are identical: detected is False, outcome is None!
    assert obs_miss.detected is False
    assert obs_neg.detected is False
    assert obs_miss.target_signal.outcome_category is None
    assert obs_neg.target_signal.outcome_category is None
    assert obs_miss.target_signal.detected == obs_neg.target_signal.detected

    # Observer at 0, target at 0 with p_d = 1.0 -> Guaranteed TRUE POSITIVE
    det_spec_pos = DetectionSpecification(signal_radius=1, p_d=1.0, p_fa=1.0)
    engine_pos = DetectionEngine(spec=det_spec_pos, seed=42)
    builder_pos = ObservationBuilder(neighbor_budget=2, detection_engine=engine_pos, seed=42)

    obs_tp = builder_pos.build_observation(walker=walker, active_graph=view, target_node=0, time=0)
    # Observer at 0, target at 9 with p_fa = 1.0 -> Guaranteed FALSE POSITIVE
    obs_fp = builder_pos.build_observation(walker=walker, active_graph=view, target_node=9, time=0)

    # To the walker, BOTH signals are identical: detected is True, outcome is None!
    assert obs_tp.detected is True
    assert obs_fp.detected is True
    assert obs_tp.target_signal.outcome_category is None
    assert obs_fp.target_signal.outcome_category is None


def test_observation_dynamic_evolution_sequence():
    """Verify that observations accurately track dynamic graph churn over multiple discrete steps."""
    g = generate_erdos_renyi(n=15, p=0.4, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.FAST, seed=42)
    builder = ObservationBuilder(neighbor_budget=3, seed=42)

    walker = WalkerState(walker_id=0, current_node=0)
    history = LocalObservationHistory()

    for step in range(5):
        view = ActiveGraphView(dyn)
        obs, history = builder.observe_and_update_history(
            walker=walker,
            active_graph=view,
            target_node=14,
            time=step,
            history=history,
        )

        assert obs.time == step
        assert len(obs.checked_neighbors) <= 3
        # Each observed edge must be active in that step's active view
        for nbr in obs.checked_neighbors:
            assert view.has_active_edge(0, nbr) is True
            assert obs.is_edge_known_active(nbr) is True

        dyn.advance()

    assert len(history) == 5
    # Historical observations preserve their respective time stamps
    for step in range(5):
        assert history[step].time == step
