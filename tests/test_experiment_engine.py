"""
Automated Test Suite for Experimental Harness & Simulation Coordinator
=======================================================================
Verifies deterministic reproducibility, fair multi-algorithm execution,
simulation coordination, epistemic boundary enforcement, metrics aggregation,
and JSON artifact serialization.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from aesw.environment.types import (
    EdgeState,
    SearchTerminationStatus,
    DynamicRegime,
    TargetMode,
)
from aesw.environment.problem import (
    ProblemDefinition,
    GraphSpecification,
    DynamicsSpecification,
    TargetSpecification,
    DetectionSpecification,
    ObservationSpecification,
    WalkerSpecification,
    SimulationSpecification,
    CostSpecification,
)
from aesw.environment.coordinator import SimulationCoordinator
from aesw.graph.factory import generate_graph
from aesw.baselines.types import ActionType, BaselineType
from aesw.baselines.actions import SearchAction
from aesw.evaluation.metrics import RunMetrics, AggregatedMetrics, compute_aggregated_metrics
from aesw.evaluation.experiment import (
    ExperimentInstance,
    ExperimentResult,
    BenchmarkResult,
    generate_experiment,
    run_experiment,
    run_benchmark,
)


# ============================================================================
# 1. SimulationCoordinator Tests
# ============================================================================

def test_simulation_coordinator_initialization():
    """Verify coordinator sets up initial state, tracking, and observations properly."""
    g_spec = GraphSpecification(graph_type="er", num_nodes=20, seed=42, parameters={"p": 0.2})
    graph = generate_graph(g_spec, seed=42)
    p_def = ProblemDefinition(
        graph=g_spec,
        simulation=SimulationSpecification(time_horizon=50, seed=42),
        walker=WalkerSpecification(count=2),
    )

    coord = SimulationCoordinator(
        problem_def=p_def,
        static_graph=graph,
        initial_target_node=0,
        initial_walker_nodes={0: 1, 1: 2},
        seed_dynamics=10,
        seed_target=20,
        seed_detection=30,
        seed_observation=40,
    )

    assert coord.time == 0
    assert not coord.is_terminated
    assert coord.termination_status is None
    assert coord.total_moves == 0
    assert coord.total_messages == 0
    assert coord.visited_nodes == {1, 2}

    obs = coord.get_observations()
    assert len(obs) == 2
    assert obs[0].current_node == 1
    assert obs[1].current_node == 2
    assert obs[0].time == 0


def test_simulation_coordinator_step_and_moves():
    """Verify walker actions mutate state, increment moves and revisits, and track trajectories."""
    g_spec = GraphSpecification(graph_type="er", num_nodes=20, seed=42, parameters={"p": 0.3})
    graph = generate_graph(g_spec, seed=42)
    p_def = ProblemDefinition(
        graph=g_spec,
        simulation=SimulationSpecification(time_horizon=50, seed=42),
        walker=WalkerSpecification(count=1),
    )

    coord = SimulationCoordinator(
        problem_def=p_def,
        static_graph=graph,
        initial_target_node=19,
        initial_walker_nodes={0: 0},
        seed_dynamics=10,
        seed_target=20,
    )

    obs = coord.get_observations()
    active_nbrs = [n for n in obs[0].checked_neighbors if obs[0].is_edge_known_active(n)]
    assert len(active_nbrs) > 0
    target_nbr = active_nbrs[0]

    action = SearchAction(action_type=ActionType.MOVE, destination=target_nbr)
    new_obs, terminated, status = coord.step({0: action})

    assert coord.time == 1
    assert coord.total_moves == 1
    assert coord.walker_states[0].current_node == target_nbr
    assert target_nbr in coord.visited_nodes
    assert len(coord.walker_trajectories[0]) == 2


def test_simulation_coordinator_immediate_acquisition():
    """Verify coordinator detects target acquisition at t=0 when walker co-locates with target."""
    g_spec = GraphSpecification(graph_type="er", num_nodes=10, seed=42, parameters={"p": 0.3})
    graph = generate_graph(g_spec, seed=42)
    p_def = ProblemDefinition(graph=g_spec)

    coord = SimulationCoordinator(
        problem_def=p_def,
        static_graph=graph,
        initial_target_node=5,
        initial_walker_nodes={0: 5},
    )

    assert coord.is_terminated
    assert coord.termination_status == SearchTerminationStatus.SUCCESS
    metrics = coord.get_run_metrics(algorithm_name="test", seed=42)
    assert metrics.success is True
    assert metrics.search_time == 0


def test_simulation_coordinator_budget_exhaustion():
    """Verify coordinator terminates with BUDGET_EXHAUSTED when time horizon is reached."""
    g_spec = GraphSpecification(graph_type="er", num_nodes=10, seed=42, parameters={"p": 0.3})
    graph = generate_graph(g_spec, seed=42)
    p_def = ProblemDefinition(
        graph=g_spec,
        simulation=SimulationSpecification(time_horizon=3, seed=42),
        walker=WalkerSpecification(count=1),
    )

    coord = SimulationCoordinator(
        problem_def=p_def,
        static_graph=graph,
        initial_target_node=9,
        initial_walker_nodes={0: 0},
    )

    # Perform STAY actions until horizon
    for _ in range(3):
        new_obs, term, status = coord.step({0: SearchAction(action_type=ActionType.STAY)})

    assert coord.is_terminated
    assert coord.termination_status == SearchTerminationStatus.BUDGET_EXHAUSTED
    metrics = coord.get_run_metrics(algorithm_name="test", seed=42)
    assert metrics.success is False
    assert metrics.search_time == 3


def test_simulation_coordinator_disconnection_status():
    """Verify coordinator terminates with DISCONNECTED if target is unreachable and p_on=0."""
    g_spec = GraphSpecification(graph_type="er", num_nodes=10, seed=42, parameters={"p": 0.0})
    graph = generate_graph(g_spec, seed=42)  # Disconnected empty graph
    p_def = ProblemDefinition(
        graph=g_spec,
        dynamics=DynamicsSpecification(regime=DynamicRegime.STATIC, p_on=0.0, p_off=0.0),
        simulation=SimulationSpecification(time_horizon=10, seed=42),
        walker=WalkerSpecification(count=1),
    )

    coord = SimulationCoordinator(
        problem_def=p_def,
        static_graph=graph,
        initial_target_node=9,
        initial_walker_nodes={0: 0},
    )

    new_obs, term, status = coord.step({0: SearchAction(action_type=ActionType.STAY)})
    assert term is True
    assert status == SearchTerminationStatus.DISCONNECTED


# ============================================================================
# 2. generate_experiment & ExperimentInstance Tests
# ============================================================================

def test_generate_experiment_determinism():
    """Verify generate_experiment produces identical environments for the same seed."""
    exp1 = generate_experiment(seed=42, num_nodes=30)
    exp2 = generate_experiment(seed=42, num_nodes=30)

    assert exp1.master_seed == exp2.master_seed
    assert exp1.sub_seeds == exp2.sub_seeds
    assert exp1.initial_target_node == exp2.initial_target_node
    assert exp1.initial_walker_nodes == exp2.initial_walker_nodes
    assert exp1.static_graph.node_count == exp2.static_graph.node_count
    assert exp1.static_graph.edge_count == exp2.static_graph.edge_count


def test_generate_experiment_seed_variation():
    """Verify different master seeds produce different environments and sub-seeds."""
    exp1 = generate_experiment(seed=42, num_nodes=30)
    exp2 = generate_experiment(seed=100, num_nodes=30)

    assert exp1.master_seed != exp2.master_seed
    assert exp1.sub_seeds["dynamics"] != exp2.sub_seeds["dynamics"]
    assert exp1.sub_seeds["target"] != exp2.sub_seeds["target"]


def test_generate_experiment_overrides():
    """Verify parameter overrides update ProblemDefinition specs properly."""
    exp = generate_experiment(
        seed=42,
        graph_type="watts_strogatz",
        num_nodes=50,
        regime="FAST",
        p_on=0.1,
        p_off=0.05,
        p_move=0.08,
        p_d=0.75,
        p_fa=0.05,
        walker_count=3,
        neighbor_budget=6,
        time_horizon=200,
    )

    assert exp.problem_def.graph.graph_type == "watts_strogatz"
    assert exp.problem_def.graph.num_nodes == 50
    assert exp.problem_def.dynamics.regime == DynamicRegime.FAST
    assert exp.problem_def.dynamics.p_on == 0.1
    assert exp.problem_def.target.p_move == 0.08
    assert exp.problem_def.detection.p_d == 0.75
    assert exp.problem_def.walker.count == 3
    assert len(exp.initial_walker_nodes) == 3


# ============================================================================
# 3. Fair Multi-Algorithm Execution & Reproducibility Tests
# ============================================================================

def test_fair_comparison_on_identical_environment():
    """CORE RESEARCH REQUIREMENT:

    Evaluating different algorithms on the exact same ExperimentInstance
    exposes them to the identical initial topology, initial target location,
    initial walker locations, and dynamic edge/target transition sequences.
    """
    exp = generate_experiment(
        seed=42,
        num_nodes=40,
        time_horizon=20,
        walker_count=2,
    )

    res_rw = run_experiment(exp, "random_walk")
    res_nb = run_experiment(exp, "non_backtracking")
    res_db = run_experiment(exp, "degree_based")
    res_fl = run_experiment(exp, "flooding")
    res_ac = run_experiment(exp, "ant_colony")
    res_irw = run_experiment(exp, "independent_random_walkers")

    # Initial target and walker positions must be identical across all runs
    for res in (res_rw, res_nb, res_db, res_fl, res_ac, res_irw):
        assert res.parameters["target"]["initial_node"] == exp.initial_target_node
        assert res.parameters["walker"]["initial_nodes"] == dict(exp.initial_walker_nodes)
        assert res.seed == 42

    # Target trajectories must be bit-for-bit identical across runs for common steps
    common_len = min(len(res.target_trajectory) for res in (res_rw, res_nb, res_db, res_fl, res_ac, res_irw))
    for res in (res_nb, res_db, res_fl, res_ac, res_irw):
        assert res.target_trajectory[:common_len] == res_rw.target_trajectory[:common_len]


def test_algorithm_rng_isolation_from_environment():
    """Verify that changing algorithm RNG seed cannot alter environment realizations."""
    from aesw.baselines.random_walk import RandomWalkPolicy
    exp = generate_experiment(seed=42, num_nodes=40, time_horizon=25, walker_count=1)

    res_seed_a = run_experiment(exp, RandomWalkPolicy(seed=101))
    res_seed_b = run_experiment(exp, RandomWalkPolicy(seed=999))

    common_len = min(len(res_seed_a.target_trajectory), len(res_seed_b.target_trajectory))
    assert res_seed_a.target_trajectory[:common_len] == res_seed_b.target_trajectory[:common_len]

    # Verify dynamic edge transitions match identically across coordinators
    c1 = exp.create_coordinator()
    c2 = exp.create_coordinator()
    for _ in range(10):
        assert c1._dynamic_graph.edge_states == c2._dynamic_graph.edge_states
        c1._dynamic_graph.advance()
        c2._dynamic_graph.advance()



def test_run_experiment_reproducibility():
    """Verify running the same algorithm twice on the same ExperimentInstance produces bit-for-bit identical results."""
    exp = generate_experiment(seed=42, num_nodes=30, time_horizon=30, walker_count=1)

    res1 = run_experiment(exp, "random_walk")
    res2 = run_experiment(exp, "random_walk")

    assert res1.search_time == res2.search_time
    assert res1.total_moves == res2.total_moves
    assert res1.total_cost == res2.total_cost
    assert res1.status == res2.status
    assert res1.success == res2.success
    assert res1.target_trajectory == res2.target_trajectory
    assert res1.walker_trajectories == res2.walker_trajectories


def test_run_experiment_all_baseline_types():
    """Verify all six baseline algorithms execute cleanly through run_experiment."""
    exp = generate_experiment(seed=42, num_nodes=25, time_horizon=15, walker_count=2)

    baselines = [
        "random_walk",
        "non_backtracking",
        "independent_random_walkers",
        "degree_based",
        "flooding",
        "ant_colony",
    ]

    for b in baselines:
        res = run_experiment(exp, b)
        assert isinstance(res, ExperimentResult)
        assert res.search_time >= 0
        assert res.total_moves >= 0
        assert res.total_cost >= 0.0
        assert res.status in (
            SearchTerminationStatus.SUCCESS,
            SearchTerminationStatus.BUDGET_EXHAUSTED,
            SearchTerminationStatus.DISCONNECTED,
        )


# ============================================================================
# 4. JSON Serialization & Artifact Tests
# ============================================================================

def test_experiment_result_json_serialization(tmp_path: Path):
    """Verify ExperimentResult serializes to valid, self-contained JSON."""
    exp = generate_experiment(seed=42, num_nodes=20, time_horizon=10, walker_count=1)
    res = run_experiment(exp, "random_walk")

    json_str = res.to_json()
    parsed = json.loads(json_str)

    assert parsed["experiment_id"] == res.experiment_id
    assert parsed["algorithm"] == "random_walk"
    assert parsed["seed"] == 42
    assert "metrics" in parsed
    assert "parameters" in parsed
    assert "target_trajectory" in parsed
    assert "walker_trajectories" in parsed

    # Test saving to disk
    artifact_path = res.save(tmp_path)
    assert artifact_path.exists()
    assert artifact_path.suffix == ".json"

    with open(artifact_path, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["experiment_id"] == res.experiment_id


# ============================================================================
# 5. Metrics Aggregation & Benchmark Suite Tests
# ============================================================================

def test_compute_aggregated_metrics_basic():
    """Verify statistical aggregation across a collection of RunMetrics."""
    runs = [
        RunMetrics(
            algorithm="test",
            seed=1,
            status=SearchTerminationStatus.SUCCESS,
            success=True,
            search_time=10,
            nodes_visited=5,
            node_revisits=2,
            total_moves=8,
            total_messages=0,
            total_cost=8.0,
        ),
        RunMetrics(
            algorithm="test",
            seed=2,
            status=SearchTerminationStatus.SUCCESS,
            success=True,
            search_time=20,
            nodes_visited=8,
            node_revisits=4,
            total_moves=16,
            total_messages=0,
            total_cost=16.0,
        ),
        RunMetrics(
            algorithm="test",
            seed=3,
            status=SearchTerminationStatus.BUDGET_EXHAUSTED,
            success=False,
            search_time=30,
            nodes_visited=10,
            node_revisits=8,
            total_moves=24,
            total_messages=0,
            total_cost=24.0,
        ),
    ]

    agg = compute_aggregated_metrics(runs)
    assert agg.num_runs == 3
    assert pytest.approx(agg.success_rate, 1e-4) == 2.0 / 3.0
    assert pytest.approx(agg.mean_search_time, 1e-4) == 20.0
    assert pytest.approx(agg.mean_total_cost, 1e-4) == 16.0
    assert agg.ci_lower_cost <= agg.mean_total_cost <= agg.ci_upper_cost


def test_compute_aggregated_metrics_empty():
    """Verify empty run sequence raises ValueError."""
    with pytest.raises(ValueError, match="Cannot aggregate empty"):
        compute_aggregated_metrics([])


def test_run_benchmark_execution(tmp_path: Path):
    """Verify running a comparative benchmark suite across algorithms and seeds."""
    benchmark = run_benchmark(
        algorithms=["random_walk", "degree_based"],
        seeds=[42, 43],
        output_dir=tmp_path,
        suite_name="test_suite",
        num_nodes=20,
        time_horizon=10,
        walker_count=1,
    )

    assert isinstance(benchmark, BenchmarkResult)
    assert benchmark.name == "test_suite"
    assert benchmark.num_runs_per_algorithm == 2
    assert len(benchmark.run_results) == 4  # 2 algorithms * 2 seeds
    assert "random_walk" in benchmark.aggregated_metrics
    assert "degree_based" in benchmark.aggregated_metrics

    # Check saved benchmark summary file
    summary_file = tmp_path / "test_suite_summary.json"
    assert summary_file.exists()


# ============================================================================
# 6. Multi-Topology Compatibility
# ============================================================================

@pytest.mark.parametrize("graph_family", [
    ("er", {"p": 0.25}),
    ("ba", {"m": 2}),
    ("watts_strogatz", {"k": 4, "p": 0.1}),
    ("grid", {"rows": 5, "cols": 5, "obstacle_ratio": 0.1}),
    ("random_geometric", {"radius": 0.35, "dim": 2}),
])
def test_experiment_engine_across_all_topologies(graph_family):
    """Verify experiment engine operates seamlessly across all synthetic graph families."""
    g_type, params = graph_family
    exp = generate_experiment(
        seed=42,
        graph_type=g_type,
        num_nodes=25,
        graph_parameters=params,
        time_horizon=15,
        walker_count=1,
    )

    res = run_experiment(exp, "random_walk")
    assert isinstance(res, ExperimentResult)
    assert res.search_time >= 0
    assert res.total_moves >= 0
