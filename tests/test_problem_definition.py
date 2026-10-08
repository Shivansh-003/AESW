"""
Tests for Formal Problem Definition, Data Models, State Separation, and Validation.
"""

from pathlib import Path
import pytest

from aesw.environment.types import (
    EdgeState,
    TargetMode,
    DynamicRegime,
    DetectionOutcome,
    SearchTerminationStatus,
)
from aesw.environment.models import (
    Node,
    Edge,
    DelaySpecification,
    TargetState,
    WalkerState,
)
from aesw.environment.state import GroundTruthState
from aesw.environment.observation import (
    ObservedEdgeInfo,
    TargetSignalObservation,
    Observation,
)
from aesw.environment.problem import (
    GraphSpecification,
    DynamicsSpecification,
    TargetSpecification,
    DetectionSpecification,
    ObservationSpecification,
    WalkerSpecification,
    SimulationSpecification,
    CostSpecification,
    ProblemDefinition,
)
from aesw.evaluation.metrics import RunMetrics, AggregatedMetrics
from aesw.utils.config import load_config


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# ==========================================
# A. Construction & Default Values
# ==========================================

def test_problem_definition_default_construction():
    """Verify default instantiation of ProblemDefinition."""
    problem = ProblemDefinition()
    assert problem.graph.num_nodes == 100
    assert problem.graph.graph_type == "er"
    assert problem.dynamics.regime == DynamicRegime.MEDIUM
    assert problem.target.mode == TargetMode.MOVING
    assert problem.detection.signal_radius == 1
    assert problem.observation.neighbor_budget == 4
    assert problem.walker.count == 5
    assert problem.cost.c_m == 0.1
    assert problem.delay.min_delay == 1


# ==========================================
# B. Parameter Validation
# ==========================================

def test_invalid_graph_specification():
    """Verify validation for invalid graph parameters."""
    with pytest.raises(ValueError, match="num_nodes must be positive"):
        GraphSpecification(num_nodes=0)

    with pytest.raises(ValueError, match="graph_type must not be empty"):
        GraphSpecification(graph_type="")


def test_invalid_dynamics_specification():
    """Verify validation for invalid dynamics probabilities."""
    with pytest.raises(ValueError, match="p_on must be in \\[0.0, 1.0\\]"):
        DynamicsSpecification(p_on=-0.1)

    with pytest.raises(ValueError, match="p_on must be in \\[0.0, 1.0\\]"):
        DynamicsSpecification(p_on=1.5)

    with pytest.raises(ValueError, match="p_off must be in \\[0.0, 1.0\\]"):
        DynamicsSpecification(p_off=-0.05)


def test_invalid_target_specification():
    """Verify validation for invalid target locomotion parameters."""
    with pytest.raises(ValueError, match="p_move must be in \\[0.0, 1.0\\]"):
        TargetSpecification(p_move=-0.2)

    with pytest.raises(ValueError, match="p_move must be in \\[0.0, 1.0\\]"):
        TargetSpecification(p_move=1.2)


def test_invalid_detection_specification():
    """Verify validation for detection probabilities and signal radius."""
    with pytest.raises(ValueError, match="signal_radius s must be non-negative"):
        DetectionSpecification(signal_radius=-1)

    with pytest.raises(ValueError, match="p_d must be in \\[0.0, 1.0\\]"):
        DetectionSpecification(p_d=1.1)

    with pytest.raises(ValueError, match="p_fa must be in \\[0.0, 1.0\\]"):
        DetectionSpecification(p_fa=-0.01)


def test_invalid_observation_and_walker_spec():
    """Verify validation for observation budget and walker count."""
    with pytest.raises(ValueError, match="neighbor_budget B must be strictly positive"):
        ObservationSpecification(neighbor_budget=0)

    with pytest.raises(ValueError, match="walker count must be strictly positive"):
        WalkerSpecification(count=0)


def test_invalid_simulation_and_cost_spec():
    """Verify validation for simulation time horizon and cost parameters."""
    with pytest.raises(ValueError, match="time_horizon must be strictly positive"):
        SimulationSpecification(time_horizon=0)

    with pytest.raises(ValueError, match="communication cost coefficient c_m must be non-negative"):
        CostSpecification(c_m=-0.5)

    cost_spec = CostSpecification(c_m=0.2)
    assert cost_spec.compute_cost(num_moves=10, num_messages=5) == 10 + 0.2 * 5
    with pytest.raises(ValueError, match="num_moves must be non-negative"):
        cost_spec.compute_cost(num_moves=-1, num_messages=5)


def test_invalid_delay_specification():
    """Verify validation for traversal delay limits."""
    with pytest.raises(ValueError, match="min_delay must be at least 1 step"):
        DelaySpecification(min_delay=0)

    with pytest.raises(ValueError, match="cannot be less than min_delay"):
        DelaySpecification(min_delay=5, max_delay=3)

    with pytest.raises(ValueError, match="mean_delay .* must be within"):
        DelaySpecification(min_delay=2, max_delay=6, mean_delay=1.0)


# ==========================================
# C. Enums and Data Models
# ==========================================

def test_edge_states_and_models():
    """Verify Edge, Node, and EdgeState behavior."""
    node1 = Node(node_id=1, metadata={"coords": (0.0, 1.0)})
    node2 = Node(node_id=2)
    assert node1.node_id == 1

    edge = Edge(source=1, target=2, state=EdgeState.ON)
    assert edge.endpoints == frozenset([1, 2])
    assert edge.state == EdgeState.ON

    with pytest.raises(ValueError, match="node_id must not be None"):
        Node(node_id="")

    with pytest.raises(ValueError, match="Self-loops are not permitted"):
        Edge(source=1, target=1)


def test_walker_and_target_state():
    """Verify WalkerState and TargetState data models."""
    target = TargetState(current_node=10, mode=TargetMode.MOVING, p_move=0.1)
    assert target.current_node == 10

    walker = WalkerState(walker_id="w1", current_node=1, step_count=3, message_count=2)
    assert walker.step_count == 3
    assert walker.message_count == 2


# ==========================================
# D. Ground Truth State Model
# ==========================================

def test_ground_truth_state_representation():
    """Verify GroundTruthState accurately encapsulates unobserved world state."""
    nodes = {1: Node(node_id=1), 2: Node(node_id=2), 3: Node(node_id=3)}
    edges = (
        Edge(source=1, target=2, state=EdgeState.ON),
        Edge(source=2, target=3, state=EdgeState.OFF),
    )
    active_edge_states = {
        frozenset([1, 2]): EdgeState.ON,
        frozenset([2, 3]): EdgeState.OFF,
    }
    target = TargetState(current_node=3, mode=TargetMode.STATIC)
    gt = GroundTruthState(
        time=0,
        nodes=nodes,
        edges=edges,
        active_edge_states=active_edge_states,
        target_state=target,
    )

    assert gt.num_nodes == 3
    assert len(gt.active_edges) == 1
    assert gt.active_edges[0].source == 1

    # Invariant check: target must exist in V
    with pytest.raises(ValueError, match="Target location '99' does not exist"):
        GroundTruthState(
            time=0,
            nodes=nodes,
            edges=edges,
            active_edge_states=active_edge_states,
            target_state=TargetState(current_node=99),
        )


# ==========================================
# E. Observation Model & Information Boundary
# ==========================================

def test_observation_model_and_information_boundary():
    """Verify that Observation represents local noisy signals without exposing ground truth."""
    obs_edge = ObservedEdgeInfo(neighbor_id=2, state=EdgeState.ON, observed_at_time=0)
    sig_obs = TargetSignalObservation(
        detected=True,
        signal_strength=0.85,
        outcome_category=DetectionOutcome.POSITIVE,
        timestamp=0,
    )
    observation = Observation(
        walker_id=0,
        current_node=1,
        time=0,
        checked_neighbors=(2,),
        observed_edges={2: obs_edge},
        target_signal=sig_obs,
        neighbor_budget_used=1,
    )

    assert observation.current_node == 1
    assert observation.target_signal.detected is True

    # Critical Controlled Observation Principle Invariants:
    # 1. Observation has NO attribute named target_node or true_target
    assert not hasattr(observation, "target_node")
    assert not hasattr(observation, "true_target")
    assert not hasattr(observation, "ground_truth")

    # 2. Observation budget check consistency
    with pytest.raises(ValueError, match="must equal neighbor_budget_used"):
        Observation(
            walker_id=0,
            current_node=1,
            time=0,
            checked_neighbors=(2,),
            observed_edges={2: obs_edge},
            target_signal=sig_obs,
            neighbor_budget_used=2,  # mismatch
        )


# ==========================================
# F. Serialization Determinism
# ==========================================

def test_problem_definition_serialization():
    """Verify deterministic serialization to dictionary format."""
    problem = ProblemDefinition()
    data = problem.to_dict()
    assert isinstance(data, dict)
    assert data["graph"]["graph_type"] == "er"
    assert data["dynamics"]["regime"] == "MEDIUM"
    assert data["target"]["mode"] == "MOVING"
    assert data["cost"]["c_m"] == 0.1


# ==========================================
# G. Configuration Integration
# ==========================================

def test_problem_definition_from_configs():
    """Verify mapping from YAML configuration files into ProblemDefinition."""
    default_cfg = load_config(CONFIGS_DIR / "default.yaml")
    graph_cfg = load_config(CONFIGS_DIR / "graph.yaml")
    dynamics_cfg = load_config(CONFIGS_DIR / "dynamics.yaml")
    experiments_cfg = load_config(CONFIGS_DIR / "experiments.yaml")

    problem = ProblemDefinition.from_configs(
        default_config=default_cfg,
        graph_config=graph_cfg,
        dynamics_config=dynamics_cfg,
        experiments_config=experiments_cfg,
    )

    assert problem.graph.num_nodes == 100
    assert problem.dynamics.regime == DynamicRegime.MEDIUM
    assert problem.target.mode == TargetMode.MOVING
    assert problem.walker.count == 5
    assert problem.simulation.seed == 42


# ==========================================
# H. Reproducibility
# ==========================================

def test_reproducibility_problem_definition_equality():
    """Verify that identical configurations produce identical ProblemDefinition instances."""
    p1 = ProblemDefinition()
    p2 = ProblemDefinition()
    assert p1 == p2


# ==========================================
# I. Metrics Contract
# ==========================================

def test_metrics_contracts():
    """Verify RunMetrics and AggregatedMetrics validation."""
    run = RunMetrics(
        algorithm="aesw",
        seed=42,
        status=SearchTerminationStatus.SUCCESS,
        success=True,
        search_time=150,
        nodes_visited=45,
        node_revisits=12,
        total_moves=60,
        total_messages=15,
        total_cost=61.5,
    )
    assert run.success is True
    assert run.total_cost == 61.5

    agg = AggregatedMetrics(
        algorithm="aesw",
        num_runs=50,
        success_rate=0.96,
        mean_search_time=142.5,
        std_search_time=12.1,
        mean_total_cost=60.8,
        std_total_cost=5.4,
        ci_lower_cost=59.3,
        ci_upper_cost=62.3,
    )
    assert agg.success_rate == 0.96
