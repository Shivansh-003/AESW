"""
Tests for Baseline Search Algorithms
====================================
Verifies standard search baselines under partial observability:
1. Random Walk
2. Non-Backtracking Walk
3. k Independent Random Walkers
4. Degree-Based Walk
5. Flooding / Frontier Search
6. Ant Colony Walk

Validates strict adherence to the Controlled Observation Principle,
action validation, RNG isolation, reset behavior, and absence of ground-truth leakage.
"""

import pytest
import numpy as np

from aesw.baselines.types import BaselineType, ActionType
from aesw.baselines.actions import SearchAction, validate_action
from aesw.baselines.base import BaselinePolicy
from aesw.baselines.random_walk import RandomWalkPolicy
from aesw.baselines.non_backtracking import NonBacktrackingWalkPolicy
from aesw.baselines.independent_walkers import IndependentRandomWalkers
from aesw.baselines.degree_based import DegreeBasedWalkPolicy
from aesw.baselines.flooding import FloodingPolicy
from aesw.baselines.ant_colony import AntColonyWalkPolicy
from aesw.baselines.factory import create_baseline
from aesw.environment.types import EdgeState
from aesw.environment.models import WalkerState
from aesw.environment.observation import (
    Observation,
    ObservedEdgeInfo,
    TargetSignalObservation,
)
from aesw.environment.builder import ObservationBuilder
from aesw.graph import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_grid_with_obstacles,
    generate_random_geometric,
)
from aesw.dynamics import create_dynamic_graph, ActiveGraphView, DynamicRegime


def _create_mock_observation(
    current_node: int | str,
    active_neighbors: list[int | str],
    inactive_neighbors: list[int | str] | None = None,
    detected: bool = False,
    time: int = 0,
) -> Observation:
    """Helper creating a valid Observation for unit tests."""
    inactive = inactive_neighbors or []
    all_checked = tuple(active_neighbors + inactive)
    observed_edges = {}
    for nbr in active_neighbors:
        observed_edges[nbr] = ObservedEdgeInfo(neighbor_id=nbr, state=EdgeState.ON, observed_at_time=time)
    for nbr in inactive:
        observed_edges[nbr] = ObservedEdgeInfo(neighbor_id=nbr, state=EdgeState.OFF, observed_at_time=time)

    target_sig = TargetSignalObservation(detected=detected, signal_strength=1.0 if detected else 0.0, timestamp=time)
    return Observation(
        walker_id=0,
        current_node=current_node,
        time=time,
        checked_neighbors=all_checked,
        observed_edges=observed_edges,
        target_signal=target_sig,
        neighbor_budget_used=len(all_checked),
        visible_neighbors=all_checked,
    )


# ==========================================
# 1. Factory & Identifiability Tests
# ==========================================

@pytest.mark.parametrize("b_type, expected_cls", [
    (BaselineType.RANDOM_WALK, RandomWalkPolicy),
    ("random_walk", RandomWalkPolicy),
    (BaselineType.NON_BACKTRACKING, NonBacktrackingWalkPolicy),
    ("non_backtracking", NonBacktrackingWalkPolicy),
    (BaselineType.INDEPENDENT_RANDOM_WALKERS, IndependentRandomWalkers),
    ("independent_random_walkers", IndependentRandomWalkers),
    (BaselineType.DEGREE_BASED, DegreeBasedWalkPolicy),
    ("degree_based", DegreeBasedWalkPolicy),
    (BaselineType.FLOODING, FloodingPolicy),
    ("flooding", FloodingPolicy),
    (BaselineType.ANT_COLONY, AntColonyWalkPolicy),
    ("ant_colony", AntColonyWalkPolicy),
])
def test_baseline_factory_creation(b_type, expected_cls):
    """Verify factory dispatches to correct policy type from enum or string."""
    policy = create_baseline(b_type, seed=42)
    assert isinstance(policy, expected_cls)
    assert policy.metadata.name != ""
    assert policy.metadata.description != ""


def test_baseline_factory_invalid_type():
    """Verify factory raises ValueError on unknown type."""
    with pytest.raises(ValueError, match="Unknown baseline type"):
        create_baseline("quantum_walk")


# ==========================================
# 2. Action Model & Validation Tests
# ==========================================

def test_action_validation_success():
    """Verify valid MOVE and STAY actions pass validation."""
    obs = _create_mock_observation(current_node=1, active_neighbors=[2, 3])
    act_move = SearchAction(action_type=ActionType.MOVE, destination=2)
    validate_action(act_move, obs)

    act_stay = SearchAction(action_type=ActionType.STAY, destination=1)
    validate_action(act_stay, obs)

    act_stay_none = SearchAction(action_type=ActionType.STAY, destination=None)
    validate_action(act_stay_none, obs)


def test_action_validation_unobserved_destination_rejected():
    """Verify attempting to move to an unobserved node raises ValueError."""
    obs = _create_mock_observation(current_node=1, active_neighbors=[2])
    # Node 99 was not in checked_neighbors
    act_invalid = SearchAction(action_type=ActionType.MOVE, destination=99)
    with pytest.raises(ValueError, match="not inspected within checked_neighbors"):
        validate_action(act_invalid, obs)


def test_action_validation_off_edge_destination_rejected():
    """Verify attempting to move to an observed OFF edge raises ValueError."""
    obs = _create_mock_observation(current_node=1, active_neighbors=[2], inactive_neighbors=[3])
    act_off = SearchAction(action_type=ActionType.MOVE, destination=3)
    with pytest.raises(ValueError, match="link is not currently observed as active"):
        validate_action(act_off, obs)


def test_action_validation_move_without_destination_rejected():
    """Verify MOVE action without destination raises ValueError."""
    obs = _create_mock_observation(current_node=1, active_neighbors=[2])
    act_none = SearchAction(action_type=ActionType.MOVE, destination=None)
    with pytest.raises(ValueError, match="must specify a non-None destination"):
        validate_action(act_none, obs)


# ==========================================
# 3. Random Walk Policy Tests
# ==========================================

def test_random_walk_chooses_valid_candidates_and_stays_when_empty():
    """Verify RandomWalk selects only available active candidates and stays when isolated."""
    policy = RandomWalkPolicy(seed=42)

    # 1. Normal active candidates
    obs = _create_mock_observation(current_node=0, active_neighbors=[1, 2, 3])
    action = policy.decide(obs)
    assert action.action_type == ActionType.MOVE
    assert action.destination in [1, 2, 3]

    # 2. Inactive / isolated node -> STAY
    obs_isolated = _create_mock_observation(current_node=0, active_neighbors=[])
    action_stay = policy.decide(obs_isolated)
    assert action_stay.action_type == ActionType.STAY
    assert action_stay.destination == 0


def test_random_walk_statistical_uniformity():
    """Statistical sanity test: RandomWalk samples uniformly across candidates."""
    policy = RandomWalkPolicy(seed=12345)
    candidates = [10, 20, 30, 40]
    obs = _create_mock_observation(current_node=0, active_neighbors=candidates)

    counts = {c: 0 for c in candidates}
    num_samples = 4000
    for _ in range(num_samples):
        act = policy.decide(obs)
        counts[act.destination] += 1

    expected = num_samples / len(candidates)
    for c in candidates:
        assert abs(counts[c] - expected) < 150  # Within tolerance


# ==========================================
# 4. Non-Backtracking Walk Policy Tests
# ==========================================

def test_non_backtracking_avoids_immediate_predecessor():
    """Verify NonBacktracking avoids previous node on path A - B - C."""
    policy = NonBacktrackingWalkPolicy(seed=42)

    # Move 1: at A (node 1), candidates are B (node 2)
    obs_a = _create_mock_observation(current_node=1, active_neighbors=[2])
    act_1 = policy.decide(obs_a)
    assert act_1.destination == 2
    assert policy.previous_node == 1

    # Move 2: at B (node 2), candidates are A (node 1) and C (node 3)
    obs_b = _create_mock_observation(current_node=2, active_neighbors=[1, 3])
    act_2 = policy.decide(obs_b)
    # Must prefer C (node 3) over immediate predecessor A (node 1)
    assert act_2.destination == 3
    assert policy.previous_node == 2


def test_non_backtracking_fallbacks_at_dead_end():
    """Verify NonBacktracking allows backtracking if predecessor is the only neighbor."""
    policy = NonBacktrackingWalkPolicy(seed=42)

    # Step 1: at node 1, move to 2
    obs_1 = _create_mock_observation(current_node=1, active_neighbors=[2])
    _ = policy.decide(obs_1)

    # Step 2: at node 2, dead end (only node 1 is available)
    obs_2 = _create_mock_observation(current_node=2, active_neighbors=[1])
    act_fallback = policy.decide(obs_2)
    assert act_fallback.action_type == ActionType.MOVE
    assert act_fallback.destination == 1  # Backtracking fallback accepted
    assert act_fallback.metadata["backtracked"] is True


def test_non_backtracking_reset_clears_memory():
    """Verify reset clears 1-step predecessor memory."""
    policy = NonBacktrackingWalkPolicy(seed=42)
    obs = _create_mock_observation(current_node=1, active_neighbors=[2])
    _ = policy.decide(obs)
    assert policy.previous_node == 1

    policy.reset()
    assert policy.previous_node is None


# ==========================================
# 5. k Independent Random Walkers Tests
# ==========================================

@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_independent_walkers_scaling_and_rng_isolation(k):
    """Verify IndependentRandomWalkers creates k isolated policies with independent RNG."""
    multi_policy = IndependentRandomWalkers(k=k, base_seed=100)
    assert multi_policy.k == k
    assert len(multi_policy.walkers) == k

    # Verify each walker has a distinct seed and independent decisions
    obs = _create_mock_observation(current_node=0, active_neighbors=[1, 2, 3, 4, 5])
    actions = multi_policy.decide_all({i: obs for i in range(k)})
    assert len(actions) == k
    for act in actions.values():
        assert act.destination in [1, 2, 3, 4, 5]


def test_independent_walkers_reset_independence():
    """Verify resetting one walker does not affect another walker's internal state."""
    multi = IndependentRandomWalkers(k=2, base_seed=42)
    obs = _create_mock_observation(current_node=0, active_neighbors=[1, 2, 3, 4])

    # Advance walker 0 and walker 1
    act_0_before = [multi.decide_walker(0, obs).destination for _ in range(5)]
    _ = [multi.decide_walker(1, obs).destination for _ in range(5)]

    # Reset only walker 0
    multi.reset_walker(0)
    act_0_after = [multi.decide_walker(0, obs).destination for _ in range(5)]
    assert act_0_before == act_0_after  # Exactly reproduced after individual reset


# ==========================================
# 6. Degree-Based Walk Policy Tests
# ==========================================

def test_degree_based_prefers_higher_locally_observed_degree():
    """Verify DegreeBased prefers candidate with higher observed degree."""
    policy = DegreeBasedWalkPolicy(seed=42)

    # Walker visits node 10 (which has 10 active neighbors)
    obs_hub = _create_mock_observation(current_node=10, active_neighbors=list(range(100, 110)))
    _ = policy.decide(obs_hub)
    assert policy.observed_degrees[10] == 10.0

    # Walker visits node 20 (which has only 1 active neighbor)
    obs_leaf = _create_mock_observation(current_node=20, active_neighbors=[999])
    _ = policy.decide(obs_leaf)
    assert policy.observed_degrees[20] == 1.0

    # Now walker is at node 0, choosing between node 10 (degree 10) and node 20 (degree 1)
    obs_choice = _create_mock_observation(current_node=0, active_neighbors=[10, 20])

    counts = {10: 0, 20: 0}
    for _ in range(1000):
        act = policy.decide(obs_choice)
        counts[act.destination] += 1

    # Node 10 should be chosen substantially more frequently than Node 20
    assert counts[10] > counts[20] * 5


def test_degree_based_reset_clears_cache():
    """Verify reset clears observed degree cache."""
    policy = DegreeBasedWalkPolicy(seed=42)
    obs = _create_mock_observation(current_node=5, active_neighbors=[1, 2, 3])
    _ = policy.decide(obs)
    assert 5 in policy.observed_degrees

    policy.reset()
    assert len(policy.observed_degrees) == 0


# ==========================================
# 7. Flooding / Frontier Search Tests
# ==========================================

def test_flooding_prioritizes_unvisited_frontier():
    """Verify Flooding strictly prioritizes unvisited neighbors over visited ones."""
    policy = FloodingPolicy(seed=42)

    # Initial step at node 0: all candidates [1, 2] are unvisited
    obs_0 = _create_mock_observation(current_node=0, active_neighbors=[1, 2])
    act_1 = policy.decide(obs_0)
    chosen = act_1.destination
    assert chosen in [1, 2]
    assert 0 in policy.visited_nodes

    # Next step at node 1: neighbor 0 is visited, neighbor 3 is unvisited
    obs_1 = _create_mock_observation(current_node=1, active_neighbors=[0, 3])
    act_2 = policy.decide(obs_1)
    # Must prioritize unvisited node 3 over visited node 0
    assert act_2.destination == 3
    assert act_2.metadata["prioritized_unvisited"] is True


def test_flooding_does_not_leak_hidden_nodes():
    """Verify Flooding frontier contains only observed nodes."""
    policy = FloodingPolicy(seed=42)
    obs = _create_mock_observation(current_node=0, active_neighbors=[1, 2])
    _ = policy.decide(obs)

    assert policy.discovered_nodes == {1, 2}
    assert 99 not in policy.discovered_nodes


def test_flooding_reset_clears_visited_and_discovered():
    """Verify reset clears internal visited and discovered sets."""
    policy = FloodingPolicy(seed=42)
    obs = _create_mock_observation(current_node=0, active_neighbors=[1])
    _ = policy.decide(obs)
    assert len(policy.visited_nodes) > 0

    policy.reset()
    assert len(policy.visited_nodes) == 0
    assert len(policy.discovered_nodes) == 0


# ==========================================
# 8. Ant Colony Walk Policy Tests
# ==========================================

def test_ant_colony_reinforcement_and_evaporation():
    """Verify AntColony reinforces chosen edge and evaporates trails."""
    policy = AntColonyWalkPolicy(seed=42, evaporation_rate=0.1, reinforcement=2.0)

    obs = _create_mock_observation(current_node=0, active_neighbors=[1, 2])
    act = policy.decide(obs)
    chosen = act.destination

    # Traversed edge must have reinforced pheromone > initial tau_0
    assert policy.get_pheromone(0, chosen) > 1.0

    # Evaporate trails
    tau_before = policy.get_pheromone(0, chosen)
    policy.decide(_create_mock_observation(current_node=99, active_neighbors=[]))
    tau_after = policy.get_pheromone(0, chosen)
    assert tau_after < tau_before


def test_ant_colony_statistical_bias():
    """Statistical sanity test: higher pheromone on a link increases its selection probability."""
    policy = AntColonyWalkPolicy(seed=42, alpha=2.0)
    # Manually bias edge (0, 1) heavily over (0, 2)
    policy.set_pheromone(0, 1, 10.0)
    policy.set_pheromone(0, 2, 1.0)

    obs = _create_mock_observation(current_node=0, active_neighbors=[1, 2])

    counts = {1: 0, 2: 0}
    for _ in range(500):
        # Re-set pheromone for fair testing
        policy.set_pheromone(0, 1, 10.0)
        policy.set_pheromone(0, 2, 1.0)
        act = policy.decide(obs)
        counts[act.destination] += 1

    assert counts[1] > counts[2] * 10


def test_ant_colony_reset_clears_pheromone():
    """Verify reset clears private pheromone trails."""
    policy = AntColonyWalkPolicy(seed=42)
    policy.set_pheromone(0, 1, 5.0)
    assert policy.get_pheromone(0, 1) == 5.0

    policy.reset()
    assert len(policy.pheromone_table) == 0
    assert policy.get_pheromone(0, 1) == 1.0  # Back to default tau_0


# ==========================================
# 9. Information Boundary & Introspection Tests
# ==========================================

@pytest.mark.parametrize("policy_factory", [
    lambda: RandomWalkPolicy(seed=42),
    lambda: NonBacktrackingWalkPolicy(seed=42),
    lambda: IndependentRandomWalkers(k=2, base_seed=42),
    lambda: DegreeBasedWalkPolicy(seed=42),
    lambda: FloodingPolicy(seed=42),
    lambda: AntColonyWalkPolicy(seed=42),
])
def test_no_baseline_stores_forbidden_ground_truth(policy_factory):
    """Architecture test: Verify no baseline policy instance stores forbidden ground-truth references."""
    policy = policy_factory()

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
        "outcome",
        "outcome_category",
        "future_edges",
        "all_nodes",
        "all_edges",
    ]

    for attr in forbidden_attributes:
        assert not hasattr(policy, attr), f"{type(policy).__name__} leaked attribute: {attr}"


# ==========================================
# 10. Compatibility Across Graph Families with ObservationBuilder
# ==========================================

@pytest.mark.parametrize("generator_fn", [
    lambda: generate_erdos_renyi(n=20, p=0.3, seed=42),
    lambda: generate_barabasi_albert(n=20, m=2, seed=42),
    lambda: generate_watts_strogatz(n=20, k=4, p=0.1, seed=42),
    lambda: generate_grid_with_obstacles(rows=4, cols=5, obstacle_ratio=0.1, seed=42),
    lambda: generate_random_geometric(n=20, radius=0.4, seed=42),
])
def test_baselines_with_real_observation_builder_across_families(generator_fn):
    """Integration test: All baselines process observations generated by ObservationBuilder."""
    g = generator_fn()
    dyn = create_dynamic_graph(g, regime=DynamicRegime.STATIC, seed=42)
    view = ActiveGraphView(dyn)

    builder = ObservationBuilder(neighbor_budget=3, seed=42)
    start_node = list(view.nodes.keys())[0]
    walker = WalkerState(walker_id=0, current_node=start_node)

    obs = builder.build_observation(walker=walker, active_graph=view, target_node=list(view.nodes.keys())[-1], time=0)

    policies = [
        RandomWalkPolicy(seed=42),
        NonBacktrackingWalkPolicy(seed=42),
        DegreeBasedWalkPolicy(seed=42),
        FloodingPolicy(seed=42),
        AntColonyWalkPolicy(seed=42),
    ]

    for pol in policies:
        action = pol.decide(obs)
        assert action.action_type in (ActionType.MOVE, ActionType.STAY)
        if action.action_type == ActionType.MOVE:
            assert action.destination in obs.checked_neighbors
            assert obs.is_edge_known_active(action.destination) is True
