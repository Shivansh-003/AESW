"""
Comprehensive Test Suite for Target and Uncertainty Engine.
Verifies:
- Target initialization (explicit node, random selection, validation)
- Target locomotion (stationary mode, p_move=0, p_move=1, uniform neighbor selection)
- Dynamic graph constraints (never moves over OFF edges, trapped when isolated)
- Synchronization with dynamic graph time evolution
- Detection engine distance calculation in active graph G_t
- Detection outcome probabilities (p_d, 1-p_d, p_fa, 1-p_fa)
- Four outcome categories (POSITIVE, MISSED, FALSE_POSITIVE, NEGATIVE)
- Blind search condition (p_d == p_fa)
- Unreachable/disconnected target handling (dist = inf -> outside radius)
- Information boundary: Observation contains NO ground-truth target information
- Determinism and isolated RNG streams (target vs detection)
- Cross-topology compatibility across all synthetic graph families (ER, BA, WS, Grid, RGG)
- Statistical sanity tests for movement and detection frequencies
"""

import pytest
import numpy as np

from aesw.environment.types import TargetMode, DetectionOutcome
from aesw.environment.models import TargetState
from aesw.environment.problem import TargetSpecification, DetectionSpecification
from aesw.environment.target import TargetEngine
from aesw.environment.detection import DetectionEngine, DetectionResult
from aesw.environment.observation import Observation, ObservedEdgeInfo, TargetSignalObservation
from aesw.graph import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_grid_with_obstacles,
    generate_random_geometric,
)
from aesw.dynamics import create_dynamic_graph, ActiveGraphView, DynamicRegime, InitializationPolicy


# ==========================================
# A. Target Initialization & Validation
# ==========================================

def test_target_engine_explicit_initial_node():
    """Verify target initializes at specified node."""
    spec = TargetSpecification(mode=TargetMode.MOVING, p_move=0.2, initial_node=5)
    engine = TargetEngine(spec=spec, initial_node=5)
    assert engine.current_node == 5
    assert engine.time == 0
    assert engine.trajectory == [5]
    assert engine.state.mode == TargetMode.MOVING


def test_target_engine_random_initial_node():
    """Verify target initializes deterministically from candidate nodes when not explicit."""
    spec = TargetSpecification(mode=TargetMode.MOVING, p_move=0.2)
    candidates = [10, 20, 30, 40]
    engine1 = TargetEngine(spec=spec, candidate_nodes=candidates, seed=42)
    engine2 = TargetEngine(spec=spec, candidate_nodes=candidates, seed=42)
    assert engine1.current_node in candidates
    assert engine1.current_node == engine2.current_node


def test_target_engine_invalid_init():
    """Verify target engine raises error when no initial node or candidates provided."""
    spec = TargetSpecification(mode=TargetMode.STATIC)
    with pytest.raises(ValueError, match="Must provide either initial_node"):
        TargetEngine(spec=spec)


# ==========================================
# B. Target Locomotion & Boundaries
# ==========================================

def test_target_static_mode_never_moves():
    """Verify target in STATIC mode remains stationary regardless of p_move."""
    g = generate_erdos_renyi(n=10, p=0.8, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC)
    view = ActiveGraphView(dyn)

    spec = TargetSpecification(mode=TargetMode.STATIC, p_move=1.0, initial_node=0)
    engine = TargetEngine(spec=spec)

    for _ in range(5):
        state = engine.step(view)
        assert state.current_node == 0
        assert state.move_attempted is False
        assert state.move_succeeded is False

    assert engine.trajectory == [0, 0, 0, 0, 0, 0]


def test_target_pmove_zero_never_moves():
    """Verify target with p_move = 0 remains stationary even in MOVING mode."""
    g = generate_erdos_renyi(n=10, p=0.8, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC)
    view = ActiveGraphView(dyn)

    spec = TargetSpecification(mode=TargetMode.MOVING, p_move=0.0, initial_node=0)
    engine = TargetEngine(spec=spec)

    for _ in range(5):
        state = engine.step(view)
        assert state.current_node == 0


def test_target_pmove_one_always_moves_to_active_neighbor():
    """Verify target with p_move = 1 moves to an active neighbor at each step."""
    g = generate_erdos_renyi(n=15, p=0.7, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, initialization_policy=InitializationPolicy.ALL_ON)
    view = ActiveGraphView(dyn)

    spec = TargetSpecification(mode=TargetMode.MOVING, p_move=1.0, initial_node=0)
    engine = TargetEngine(spec=spec, seed=99)

    curr = 0
    for _ in range(5):
        active_nbrs = view.active_neighbors(curr)
        state = engine.step(view)
        assert state.current_node in active_nbrs
        assert state.move_attempted is True
        assert state.move_succeeded is True
        assert state.previous_node == curr
        curr = state.current_node


def test_target_trapped_when_isolated():
    """Verify target remains stationary when all incident edges are OFF."""
    g = generate_erdos_renyi(n=10, p=0.5, seed=42)
    # ALL_OFF initialization: target has 0 active neighbors
    dyn = create_dynamic_graph(g, initialization_policy=InitializationPolicy.ALL_OFF)
    view = ActiveGraphView(dyn)

    spec = TargetSpecification(mode=TargetMode.MOVING, p_move=1.0, initial_node=0)
    engine = TargetEngine(spec=spec)

    state = engine.step(view)
    assert state.current_node == 0
    assert state.move_attempted is True
    assert state.move_succeeded is False


def test_target_never_traverses_off_edges():
    """Verify target strictly moves across ON edges and never across OFF edges."""
    g = generate_erdos_renyi(n=20, p=0.3, seed=42)
    dyn = create_dynamic_graph(g, p_on=0.1, p_off=0.1, seed=42)
    view = ActiveGraphView(dyn)

    # Pick an initial node with some neighbors
    initial = list(g.nodes.keys())[0]
    spec = TargetSpecification(mode=TargetMode.MOVING, p_move=1.0, initial_node=initial)
    engine = TargetEngine(spec=spec, seed=123)

    for _ in range(10):
        prev = engine.current_node
        active_nbrs = view.active_neighbors(prev)
        state = engine.step(view)
        if state.current_node != prev:
            assert state.current_node in active_nbrs
            assert dyn.is_edge_active(prev, state.current_node) is True
        dyn.advance()


# ==========================================
# C. Detection Engine & Distance Calculation
# ==========================================

def test_detection_distance_calculation():
    """Verify breadth-first search distance calculation in active graph G_t."""
    g = generate_grid_with_obstacles(rows=3, cols=3, obstacle_ratio=0.0, seed=42)
    dyn = create_dynamic_graph(g, initialization_policy=InitializationPolicy.ALL_ON)
    view = ActiveGraphView(dyn)

    spec = DetectionSpecification(signal_radius=2, p_d=0.9, p_fa=0.05)
    det_engine = DetectionEngine(spec=spec)

    # (0,0) to (0,0) distance is 0
    assert det_engine.compute_distance("(0,0)", "(0,0)", view) == 0.0
    # (0,0) to (0,1) distance is 1
    assert det_engine.compute_distance("(0,0)", "(0,1)", view) == 1.0
    # (0,0) to (2,2) distance in 3x3 grid is 4
    assert det_engine.compute_distance("(0,0)", "(2,2)", view) == 4.0


def test_detection_disconnected_target_is_infinite_distance():
    """Verify that when target is disconnected, distance is inf and treated as outside radius."""
    g = generate_erdos_renyi(n=10, p=0.5, seed=42)
    dyn = create_dynamic_graph(g, initialization_policy=InitializationPolicy.ALL_OFF)
    view = ActiveGraphView(dyn)

    spec = DetectionSpecification(signal_radius=2, p_d=0.9, p_fa=0.05)
    det_engine = DetectionEngine(spec=spec)

    dist = det_engine.compute_distance(0, 1, view)
    assert dist == float("inf")

    res = det_engine.detect(observer_node=0, target_node=1, active_graph=view)
    assert res.within_radius is False
    assert res.graph_distance == float("inf")


# ==========================================
# D. Detection Probabilities & Outcomes
# ==========================================

def test_detection_outcomes_inside_and_outside_radius():
    """Verify deterministic outcomes under extreme probability thresholds."""
    g = generate_erdos_renyi(n=10, p=0.8, seed=42)
    dyn = create_dynamic_graph(g, initialization_policy=InitializationPolicy.ALL_ON)
    view = ActiveGraphView(dyn)

    # 1. p_d = 1.0: inside radius always produces POSITIVE (True Positive)
    spec_tp = DetectionSpecification(signal_radius=1, p_d=1.0, p_fa=0.0)
    engine_tp = DetectionEngine(spec=spec_tp)
    # Node 0 to itself has dist 0 <= 1
    res_tp = engine_tp.detect(observer_node=0, target_node=0, active_graph=view)
    assert res_tp.within_radius is True
    assert res_tp.positive_signal is True
    assert res_tp.outcome == DetectionOutcome.POSITIVE

    # 2. p_d = 0.0: inside radius always produces MISSED detection
    spec_miss = DetectionSpecification(signal_radius=1, p_d=0.0, p_fa=0.0)
    engine_miss = DetectionEngine(spec=spec_miss)
    res_miss = engine_miss.detect(observer_node=0, target_node=0, active_graph=view)
    assert res_miss.within_radius is True
    assert res_miss.positive_signal is False
    assert res_miss.outcome == DetectionOutcome.MISSED

    # 3. p_fa = 1.0: outside radius always produces FALSE_POSITIVE
    # In 10-node graph with ALL_OFF, dist = inf > 1
    dyn_off = create_dynamic_graph(g, initialization_policy=InitializationPolicy.ALL_OFF)
    view_off = ActiveGraphView(dyn_off)

    spec_fp = DetectionSpecification(signal_radius=1, p_d=0.0, p_fa=1.0)
    engine_fp = DetectionEngine(spec=spec_fp)
    res_fp = engine_fp.detect(observer_node=0, target_node=1, active_graph=view_off)
    assert res_fp.within_radius is False
    assert res_fp.positive_signal is True
    assert res_fp.outcome == DetectionOutcome.FALSE_POSITIVE

    # 4. p_fa = 0.0: outside radius always produces NEGATIVE
    spec_tn = DetectionSpecification(signal_radius=1, p_d=0.0, p_fa=0.0)
    engine_tn = DetectionEngine(spec=spec_tn)
    res_tn = engine_tn.detect(observer_node=0, target_node=1, active_graph=view_off)
    assert res_tn.within_radius is False
    assert res_tn.positive_signal is False
    assert res_tn.outcome == DetectionOutcome.NEGATIVE


def test_blind_search_condition_pd_equals_pfa():
    """Verify blind-search condition p_d == p_fa: signal gives identical positive probability inside & outside."""
    g = generate_erdos_renyi(n=10, p=0.5, seed=42)
    dyn = create_dynamic_graph(g, initialization_policy=InitializationPolicy.ALL_ON)
    view = ActiveGraphView(dyn)

    p_val = 0.4
    spec_blind = DetectionSpecification(signal_radius=0, p_d=p_val, p_fa=p_val)
    engine_blind = DetectionEngine(spec=spec_blind, seed=777)

    # Check a large number of draws at observer=target (inside) vs observer!=target (outside)
    inside_pos = sum(
        engine_blind.detect(observer_node=0, target_node=0, active_graph=view).positive_signal
        for _ in range(500)
    )
    outside_pos = sum(
        engine_blind.detect(observer_node=1, target_node=0, active_graph=view).positive_signal
        for _ in range(500)
    )

    freq_inside = inside_pos / 500
    freq_outside = outside_pos / 500
    assert abs(freq_inside - p_val) < 0.08
    assert abs(freq_outside - p_val) < 0.08


# ==========================================
# E. Information Boundary Test
# ==========================================

def test_ground_truth_isolation_from_observation():
    """Verify that walker Observation has no attribute leaking true target state or distance."""
    from aesw.environment.types import EdgeState
    obs_edge_real = ObservedEdgeInfo(neighbor_id=1, state=EdgeState.ON, observed_at_time=0)
    sig_obs = TargetSignalObservation(
        detected=True,
        signal_strength=0.9,
        outcome_category=DetectionOutcome.POSITIVE,
        timestamp=0,
    )
    walker_obs = Observation(
        walker_id=0,
        current_node=0,
        time=0,
        checked_neighbors=(1,),
        observed_edges={1: obs_edge_real},
        target_signal=sig_obs,
        neighbor_budget_used=1,
    )

    # Ground-truth boundary assertions:
    assert not hasattr(walker_obs, "target_node")
    assert not hasattr(walker_obs, "true_target")
    assert not hasattr(walker_obs, "target_distance")
    assert not hasattr(walker_obs, "graph_distance")
    assert not hasattr(walker_obs, "outcome_ground_truth")


# ==========================================
# F. Determinism & RNG Stream Decoupling
# ==========================================

def test_target_and_detection_rng_isolation():
    """Verify that changing detection seed does not alter target movement trajectory."""
    g = generate_erdos_renyi(n=15, p=0.5, seed=42)
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    spec_target = TargetSpecification(mode=TargetMode.MOVING, p_move=0.5, initial_node=0)
    spec_det = DetectionSpecification(signal_radius=1, p_d=0.8, p_fa=0.1)

    # Run 1: target_seed = 100, det_seed = 200
    t_engine1 = TargetEngine(spec=spec_target, seed=100)
    d_engine1 = DetectionEngine(spec=spec_det, seed=200)

    # Run 2: target_seed = 100, det_seed = 999 (different detection seed)
    t_engine2 = TargetEngine(spec=spec_target, seed=100)
    d_engine2 = DetectionEngine(spec=spec_det, seed=999)

    traj1 = []
    traj2 = []
    det1 = []
    det2 = []

    for _ in range(10):
        s1 = t_engine1.step(view)
        s2 = t_engine2.step(view)
        traj1.append(s1.current_node)
        traj2.append(s2.current_node)

        r1 = d_engine1.detect(observer_node=0, target_node=s1.current_node, active_graph=view)
        r2 = d_engine2.detect(observer_node=0, target_node=s2.current_node, active_graph=view)
        det1.append(r1.positive_signal)
        det2.append(r2.positive_signal)

    # Target trajectory is identical regardless of detection seed
    assert traj1 == traj2
    # Detection outcomes differ due to different detection seed
    assert det1 != det2


# ==========================================
# G. Statistical Sanity Checks
# ==========================================

def test_statistical_target_movement_frequency():
    """Verify observed target movement frequency approximates p_move when active neighbors exist."""
    g = generate_erdos_renyi(n=30, p=0.8, seed=42)
    dyn = create_dynamic_graph(g, initialization_policy=InitializationPolicy.ALL_ON)
    view = ActiveGraphView(dyn)

    p_move = 0.3
    spec = TargetSpecification(mode=TargetMode.MOVING, p_move=p_move, initial_node=0)
    engine = TargetEngine(spec=spec, seed=4242)

    moves = 0
    steps = 1000
    for _ in range(steps):
        prev = engine.current_node
        st = engine.step(view)
        if st.current_node != prev:
            moves += 1

    observed_p_move = moves / steps
    assert abs(observed_p_move - p_move) < 0.05


def test_statistical_detection_pd_and_pfa_frequencies():
    """Verify observed detection frequencies approach p_d and p_fa."""
    g = generate_erdos_renyi(n=40, p=0.8, seed=42)
    dyn = create_dynamic_graph(g, initialization_policy=InitializationPolicy.ALL_ON)
    view = ActiveGraphView(dyn)

    p_d = 0.85
    p_fa = 0.15
    spec = DetectionSpecification(signal_radius=0, p_d=p_d, p_fa=p_fa)
    engine = DetectionEngine(spec=spec, seed=8888)

    inside_pos = sum(
        engine.detect(observer_node=0, target_node=0, active_graph=view).positive_signal
        for _ in range(1000)
    )
    outside_pos = sum(
        engine.detect(observer_node=1, target_node=0, active_graph=view).positive_signal
        for _ in range(1000)
    )

    assert abs(inside_pos / 1000 - p_d) < 0.04
    assert abs(outside_pos / 1000 - p_fa) < 0.04


# ==========================================
# H. Compatibility Across All Five Graph Families
# ==========================================

@pytest.mark.parametrize("gen_fn", [
    lambda: generate_erdos_renyi(n=15, p=0.4, seed=42),
    lambda: generate_barabasi_albert(n=15, m=2, seed=42),
    lambda: generate_watts_strogatz(n=15, k=4, p=0.1, seed=42),
    lambda: generate_grid_with_obstacles(rows=4, cols=4, obstacle_ratio=0.1, seed=42),
    lambda: generate_random_geometric(n=15, radius=0.4, seed=42),
])
def test_target_and_detection_across_graph_families(gen_fn):
    """Verify target locomotion and detection work on all five synthetic graph families."""
    g = gen_fn()
    dyn = create_dynamic_graph(g, p_on=0.1, p_off=0.1, seed=42)
    view = ActiveGraphView(dyn)

    init_node = list(g.nodes.keys())[0]
    t_spec = TargetSpecification(mode=TargetMode.MOVING, p_move=0.5, initial_node=init_node)
    d_spec = DetectionSpecification(signal_radius=1, p_d=0.8, p_fa=0.1)

    t_engine = TargetEngine(spec=t_spec, seed=42)
    d_engine = DetectionEngine(spec=d_spec, seed=42)

    for _ in range(5):
        t_state = t_engine.step(view)
        det_res = d_engine.detect(observer_node=init_node, target_node=t_state.current_node, active_graph=view)
        assert isinstance(det_res, DetectionResult)
        assert det_res.outcome in DetectionOutcome
        dyn.advance()
