"""
Automated Test Suite for AESW Agent & Experiment Harness Integration
======================================================================
Verifies complete functionality and integration of the proposed method:
Adaptive Evidence-Sharing Walkers (AESW)

Test Coverage:
1. Agent Construction & Initialization
2. Observation -> Evidence Storage (Positive & Negative)
3. Observation -> Churn Estimation & Memory Decay Coupling
4. Node-Mediated Communication (NO_SHARING, PUSH, PULL, PUSH_PULL)
5. Adaptive Mode Controller (LOCAL <-> LONG_JUMP Transitions)
6. Next-Hop Selection & Candidate Scoring
7. Environment-Mediated Long Jump Execution
8. Epistemic Firewall Invariants (Zero Leakage)
9. Traversal Delay Handling
10. False Positive Handling & Decay
11. Multi-Walker State Isolation
12. Cross-Walker Node-Mediated Clue Transfer
13. Communication Overhead (Q) and Total Cost Accounting (C = M + c_m * Q)
14. Deterministic Reproducibility
15. Baseline Factory Integration
16. Cross-Topology Evaluation (ER, BA, WS, Grid, RGG)
17. Multi-Algorithm Fairness Benchmark on Identical ExperimentInstance
18. Trace Logging & Explainability
"""

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
    DelaySpecification,
)
from aesw.environment.coordinator import SimulationCoordinator
from aesw.environment.observation import (
    Observation,
    ObservedEdgeInfo,
    TargetSignalObservation,
)
from aesw.graph.factory import generate_graph
from aesw.baselines.types import ActionType, BaselineType
from aesw.baselines.actions import SearchAction, validate_action
from aesw.baselines.factory import create_baseline
from aesw.memory.types import EvidencePolarity, EvidenceSource
from aesw.aesw.churn import ChurnEstimator
from aesw.aesw.communication import (
    CommunicationMode,
    NodeMediatedExchange,
)
from aesw.aesw.mode import SearchMode, AdaptiveModeController
from aesw.aesw.agent import AESWAgent, AESWPolicy
from aesw.evaluation.experiment import (
    generate_experiment,
    run_experiment,
    ExperimentInstance,
)


def _make_obs(
    walker_id: int | str = 0,
    current_node: int | str = "A",
    time: int = 0,
    neighbors: tuple[int | str, ...] = ("B",),
    active_neighbors: tuple[int | str, ...] = ("B",),
    detected: bool = False,
    signal_strength: float = 0.0,
) -> Observation:
    """Helper to synthesize valid Observation instances with explicit edge states."""
    observed_edges = {
        nbr: ObservedEdgeInfo(
            neighbor_id=nbr,
            state=EdgeState.ON if nbr in active_neighbors else EdgeState.OFF,
            observed_at_time=time,
        )
        for nbr in neighbors
    }
    return Observation(
        walker_id=walker_id,
        current_node=current_node,
        time=time,
        checked_neighbors=neighbors,
        observed_edges=observed_edges,
        target_signal=TargetSignalObservation(
            detected=detected,
            signal_strength=signal_strength,
            timestamp=time,
        ),
        neighbor_budget_used=len(neighbors),
    )


# ============================================================================
# 1. AESWAgent Unit Tests
# ============================================================================

def test_aesw_agent_initialization():
    """Verify AESWAgent initializes with isolated cognitive components."""
    agent = AESWAgent(walker_id=0, seed=42, window_size=5)
    assert agent.walker_id == 0
    assert agent.cache.size == 0
    assert agent.churn_estimator.update_count == 0
    assert agent.mode_controller.mode == SearchMode.LOCAL
    assert agent.mode_controller.window_size == 5
    assert len(agent.visit_counts) == 0
    assert len(agent.step_trace) == 0
    assert agent.last_decision is None


def test_aesw_agent_observation_to_evidence():
    """Verify observations are correctly processed into positive and negative evidence."""
    agent = AESWAgent(walker_id=1, seed=42)

    # 1. Negative observation (no target detected)
    obs_neg = _make_obs(
        walker_id=1,
        current_node="node_A",
        time=0,
        neighbors=("node_B", "node_C"),
        active_neighbors=("node_B",),
        detected=False,
    )
    ev_neg = agent.update_evidence(obs_neg)
    assert ev_neg.node_id == "node_A"
    assert ev_neg.polarity == EvidencePolarity.NEGATIVE
    assert ev_neg.confidence == 0.8
    assert "node_A" in agent.cache

    # 2. Positive observation (target signal detected)
    obs_pos = _make_obs(
        walker_id=1,
        current_node="node_B",
        time=1,
        neighbors=("node_A", "node_D"),
        active_neighbors=("node_D",),
        detected=True,
        signal_strength=0.9,
    )
    ev_pos = agent.update_evidence(obs_pos)
    assert ev_pos.node_id == "node_B"
    assert ev_pos.polarity == EvidencePolarity.POSITIVE
    assert ev_pos.confidence == 0.9
    assert ev_pos.signal_strength == 0.9
    assert "node_B" in agent.cache


def test_aesw_agent_churn_and_memory_coupling():
    """Verify that churn updates alter the default decay rate in local evidence cache."""
    agent = AESWAgent(walker_id=0, seed=123, eta=0.5)

    obs1 = _make_obs(
        walker_id=0,
        current_node="n0",
        time=0,
        neighbors=("n1", "n2", "n3"),
        active_neighbors=("n1", "n2"),
    )
    agent.update_churn(obs1)
    assert agent.churn_estimator.update_count == 1

    # In step 2, active edges change completely (high churn)
    obs2 = _make_obs(
        walker_id=0,
        current_node="n0",
        time=1,
        neighbors=("n1", "n2", "n3"),
        active_neighbors=("n3",),
    )
    agent.update_churn(obs2)
    assert agent.churn_estimator.update_count == 2
    assert agent.churn_estimator.lambda_hat > 0.0
    # Cache's default lambda_hat should be aligned with churn estimate
    assert agent.cache.default_lambda_hat == pytest.approx(agent.churn_estimator.lambda_hat, rel=1e-5)


def test_aesw_agent_mode_stagnation_and_recovery():
    """Verify transition from LOCAL to LONG_JUMP on stagnation and back to LOCAL on cue."""
    agent = AESWAgent(walker_id=0, seed=42, window_size=3)

    # Step 0-2: Non-detection observations
    for t in range(3):
        obs = _make_obs(
            walker_id=0,
            current_node=f"node_{t}",
            time=t,
            neighbors=(f"node_{t+1}",),
            active_neighbors=(f"node_{t+1}",),
            detected=False,
        )
        agent.step(obs)

    # After 3 consecutive uninformative steps (stagnation == 3 >= window_size 3):
    assert agent.mode_controller.stagnation_steps == 3
    assert agent.mode_controller.is_long_jump
    # In LONG_JUMP, next step must request a JUMP action
    obs3 = _make_obs(
        walker_id=0,
        current_node="node_3",
        time=3,
        neighbors=("node_4",),
        active_neighbors=("node_4",),
        detected=False,
    )
    action_jump = agent.step(obs3)
    assert action_jump.action_type == ActionType.JUMP
    assert action_jump.destination is None

    # Step 4: Receives a positive detection
    obs_cue = _make_obs(
        walker_id=0,
        current_node="node_4",
        time=4,
        neighbors=("node_5",),
        active_neighbors=("node_5",),
        detected=True,
        signal_strength=0.85,
    )
    action_recovered = agent.step(obs_cue)
    assert agent.mode_controller.is_local
    assert agent.mode_controller.stagnation_steps == 0
    assert action_recovered.action_type == ActionType.MOVE
    assert action_recovered.destination == "node_5"


def test_aesw_agent_legal_action_generation():
    """Verify that decisions are strictly restricted to legal active incident links."""
    agent = AESWAgent(walker_id=0, seed=42)

    # Neighbor B is active, neighbor C is inactive (OFF)
    obs = _make_obs(
        walker_id=0,
        current_node="A",
        time=0,
        neighbors=("B", "C"),
        active_neighbors=("B",),
    )
    action = agent.step(obs)
    validate_action(action, obs)
    assert action.action_type == ActionType.MOVE
    assert action.destination == "B"


def test_aesw_agent_isolated_node_handling():
    """Verify that an agent on an isolated node (no active links) falls back to STAY."""
    agent = AESWAgent(walker_id=0, seed=42)

    obs = _make_obs(
        walker_id=0,
        current_node="isolated",
        time=0,
        neighbors=("other",),
        active_neighbors=(),
    )
    action = agent.step(obs)
    validate_action(action, obs)
    assert action.action_type == ActionType.STAY
    assert action.destination == "isolated"


# ============================================================================
# 2. Node-Mediated Communication Integration Tests
# ============================================================================

def test_communication_modes_behavior():
    """Verify behavior under NO_SHARING, PUSH, PULL, and PUSH_PULL."""
    obs = _make_obs(
        walker_id=0,
        current_node="hub",
        time=0,
        neighbors=("nbr",),
        active_neighbors=("nbr",),
        detected=True,
        signal_strength=1.0,
    )

    # 1. NO_SHARING
    ex_none = NodeMediatedExchange(mode=CommunicationMode.NO_SHARING)
    agent_none = AESWAgent(walker_id=0, seed=42)
    agent_none.step(obs, exchange=ex_none)
    assert ex_none.total_messages == 0
    assert not ex_none.has_node_cache("hub")

    # 2. PUSH
    ex_push = NodeMediatedExchange(mode=CommunicationMode.PUSH)
    agent_push = AESWAgent(walker_id=0, seed=42)
    agent_push.step(obs, exchange=ex_push)
    assert ex_push.total_messages == 1
    assert ex_push.has_node_cache("hub")

    # 3. PUSH_PULL
    ex_pp = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
    agent_pp = AESWAgent(walker_id=0, seed=42)
    agent_pp.step(obs, exchange=ex_pp)
    # 1 push + 1 pull = 2 messages
    assert ex_pp.total_messages == 2


def test_cross_walker_clue_transfer_via_node():
    """Verify walker 0 deposits clue at rendezvous node, walker 1 later visits and pulls it."""
    exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)
    agent0 = AESWAgent(walker_id=0, seed=1)
    agent1 = AESWAgent(walker_id=1, seed=2)

    # Walker 0 discovers strong target cue at clue_node
    obs_clue = _make_obs(
        walker_id=0,
        current_node="clue_node",
        time=0,
        neighbors=("rendezvous",),
        active_neighbors=("rendezvous",),
        detected=True,
        signal_strength=0.95,
    )
    agent0.step(obs_clue, exchange=exchange)

    # Walker 0 moves to rendezvous node at time 1 and deposits its cache
    obs_rendezvous0 = _make_obs(
        walker_id=0,
        current_node="rendezvous",
        time=1,
        neighbors=("clue_node",),
        active_neighbors=("clue_node",),
        detected=False,
    )
    agent0.step(obs_rendezvous0, exchange=exchange)

    # Walker 1 is currently empty of clue_node
    assert "clue_node" not in agent1.cache

    # Walker 1 visits rendezvous at time 2
    obs_rendezvous1 = _make_obs(
        walker_id=1,
        current_node="rendezvous",
        time=2,
        neighbors=("other",),
        active_neighbors=("other",),
        detected=False,
    )
    agent1.step(obs_rendezvous1, exchange=exchange)

    # Walker 1 should now have clue_node in its local cache
    assert "clue_node" in agent1.cache
    raw_ev = agent1.cache.get_raw("clue_node")
    assert raw_ev is not None
    assert raw_ev.walker_id == 0  # original creator identity preserved
    assert raw_ev.source == EvidenceSource.RECEIVED_EXCHANGE


# ============================================================================
# 3. AESWPolicy (Multi-Walker) Tests
# ============================================================================

def test_aesw_policy_multi_walker_state_isolation():
    """Verify each walker in AESWPolicy has completely independent local state."""
    policy = AESWPolicy(k=3, seed=42, communication_mode=CommunicationMode.NO_SHARING)
    assert policy.k == 3
    assert len(policy.agents) == 3

    # Different seeds per walker
    seeds = [a._seed for a in policy.agents]
    assert len(set(seeds)) == 3

    # Step walker 0 only
    obs0 = _make_obs(
        walker_id=0,
        current_node="node_0",
        time=0,
        neighbors=("n1",),
        active_neighbors=("n1",),
        detected=True,
    )
    policy.decide_walker(0, obs0)

    # Walkers 1 and 2 must have untouched caches and visit counts
    assert len(policy.agents[0].cache) > 0
    assert len(policy.agents[1].cache) == 0
    assert len(policy.agents[2].cache) == 0
    assert len(policy.agents[0].visit_counts) == 1
    assert len(policy.agents[1].visit_counts) == 0


def test_aesw_policy_decide_all():
    """Verify decide_all coordinates actions for all active walkers deterministically."""
    policy = AESWPolicy(k=2, seed=100)

    observations = {
        0: _make_obs(
            walker_id=0,
            current_node="A",
            time=0,
            neighbors=("B",),
            active_neighbors=("B",),
        ),
        1: _make_obs(
            walker_id=1,
            current_node="C",
            time=0,
            neighbors=("D",),
            active_neighbors=("D",),
        ),
    }

    actions = policy.decide_all(observations)
    assert set(actions.keys()) == {0, 1}
    assert actions[0].action_type in (ActionType.MOVE, ActionType.STAY, ActionType.JUMP)
    assert actions[1].action_type in (ActionType.MOVE, ActionType.STAY, ActionType.JUMP)


def test_baseline_factory_aesw():
    """Verify Baseline factory constructs AESWPolicy with proper configuration."""
    policy = create_baseline(
        BaselineType.AESW,
        seed=42,
        k=3,
        communication_mode="push_pull",
    )
    assert isinstance(policy, AESWPolicy)
    assert policy.k == 3
    assert policy.communication_mode == CommunicationMode.PUSH_PULL
    assert policy.baseline_type == BaselineType.AESW

    # String identifier construction
    policy2 = create_baseline("aesw", seed=99)
    assert isinstance(policy2, AESWPolicy)


# ============================================================================
# 4. Experiment Harness Integration Tests
# ============================================================================

def test_run_experiment_aesw_end_to_end():
    """Verify run_experiment executes AESW successfully on an ExperimentInstance."""
    exp = generate_experiment(
        graph_type="er",
        num_nodes=25,
        p=0.25,
        walkers=2,
        time_horizon=30,
        seed=42,
    )

    result = run_experiment(exp, "aesw")
    assert result.algorithm == "aesw"
    assert result.seed == 42
    assert result.search_time <= 30
    assert result.total_moves >= 0
    assert result.total_cost >= 0.0
    assert len(result.walker_trajectories) == 2
    assert result.metrics.total_cost == result.total_cost


def test_run_experiment_determinism_and_reproducibility():
    """Verify bit-for-bit repeatability of AESW across two independent executions."""
    exp1 = generate_experiment(
        graph_type="ba",
        num_nodes=30,
        m=2,
        walkers=3,
        time_horizon=40,
        seed=101,
    )
    res1 = run_experiment(exp1, "aesw")

    exp2 = generate_experiment(
        graph_type="ba",
        num_nodes=30,
        m=2,
        walkers=3,
        time_horizon=40,
        seed=101,
    )
    res2 = run_experiment(exp2, "aesw")

    assert res1.status == res2.status
    assert res1.search_time == res2.search_time
    assert res1.total_moves == res2.total_moves
    assert res1.total_messages == res2.total_messages
    assert res1.total_cost == res2.total_cost
    assert res1.nodes_visited == res2.nodes_visited
    assert res1.walker_trajectories == res2.walker_trajectories
    assert res1.target_trajectory == res2.target_trajectory


def test_environment_mediated_jump_execution():
    """Verify coordinator processes JUMP actions safely without graph leakage."""
    exp = generate_experiment(
        graph_type="er",
        num_nodes=20,
        p=0.3,
        walkers=1,
        time_horizon=25,
        seed=777,
    )
    coordinator = exp.create_coordinator()
    coordinator.get_observations()

    # Execute a deliberate JUMP action
    jump_action = SearchAction(action_type=ActionType.JUMP, destination=None)
    new_obs, terminated, status = coordinator.step({0: jump_action})

    # Walker must have moved or jumped to some valid node
    new_pos = coordinator._walker_states[0].current_node
    assert new_pos in exp.static_graph.nodes
    assert coordinator.total_moves == 1


def test_communication_cost_accounting():
    """Verify Q messages and total cost C = M + c_m * Q are computed accurately."""
    exp = generate_experiment(
        graph_type="er",
        num_nodes=20,
        p=0.3,
        walkers=2,
        time_horizon=20,
        c_m=0.5,
        seed=55,
    )
    result = run_experiment(exp, "aesw")
    expected_cost = result.total_moves + 0.5 * result.total_messages
    assert result.total_cost == pytest.approx(expected_cost, rel=1e-5)


@pytest.mark.parametrize("graph_family,extra_params", [
    ("er", {"p": 0.2}),
    ("ba", {"m": 2}),
    ("ws", {"k": 4, "p": 0.1}),
    ("grid", {"rows": 5, "cols": 5}),
    ("rgg", {"radius": 0.35}),
])
def test_cross_topology_execution(graph_family, extra_params):
    """Verify AESW executes across all 5 standard graph topologies without errors."""
    exp = generate_experiment(
        graph_type=graph_family,
        num_nodes=25,
        walkers=2,
        time_horizon=20,
        seed=123,
        **extra_params,
    )
    result = run_experiment(exp, "aesw")
    assert result.algorithm == "aesw"
    assert result.search_time > 0
    assert result.status in (
        SearchTerminationStatus.SUCCESS,
        SearchTerminationStatus.BUDGET_EXHAUSTED,
        SearchTerminationStatus.DISCONNECTED,
    )


def test_multi_algorithm_fairness_on_identical_instance():
    """Verify AESW and all 6 baselines run seamlessly on the exact same ExperimentInstance."""
    exp = generate_experiment(
        graph_type="er",
        num_nodes=20,
        p=0.3,
        walkers=2,
        time_horizon=25,
        seed=888,
    )

    algorithms = [
        "random_walk",
        "non_backtracking",
        "k_walk",
        "degree",
        "flooding",
        "ant_colony",
        "aesw",
    ]

    results = {}
    for algo in algorithms:
        res = run_experiment(exp, algo)
        results[algo] = res
        assert res.search_time > 0
        assert res.metrics.algorithm is not None

    # Verify that target initial position was identical across all algorithm runs
    initial_target_nodes = {res.target_trajectory[0] for res in results.values()}
    assert len(initial_target_nodes) == 1
    assert initial_target_nodes.pop() == exp.initial_target_node


def test_aesw_trace_and_explainability():
    """Verify that execution trace includes cognitive state attributes."""
    exp = generate_experiment(
        graph_type="er",
        num_nodes=20,
        p=0.3,
        walkers=2,
        time_horizon=15,
        seed=42,
    )
    result = run_experiment(exp, "aesw")
    assert result.trace is not None
    assert len(result.trace) > 0

    entry = result.trace[0]
    assert "time" in entry
    assert "walker_id" in entry
    assert "current_node" in entry
    assert "mode" in entry
    assert "lambda_hat" in entry
    assert "action_type" in entry

    # Verify serialization includes trace
    d = result.to_dict()
    assert "trace" in d
    assert len(d["trace"]) == len(result.trace)


def test_epistemic_firewall_integrity():
    """Verify that AESWAgent does not peek into hidden environment or ground truth attributes."""
    agent = AESWAgent(walker_id=0, seed=42)

    # Agent must not possess ground-truth environment attributes
    for forbidden in ("_problem_def", "_dynamic_graph", "_target_engine", "target_node", "p_on", "p_off"):
        assert not hasattr(agent, forbidden)

    # Step function requires only Observation snapshot
    obs = _make_obs(walker_id=0, current_node="A", time=0, neighbors=("B",), active_neighbors=("B",))
    action = agent.step(obs)
    assert action.action_type == ActionType.MOVE


def test_delay_compatibility():
    """Verify AESW operates properly under traversal delay regimes where delay > 1."""
    exp = generate_experiment(
        graph_type="er",
        num_nodes=20,
        p=0.3,
        walkers=2,
        time_horizon=25,
        min_delay=2,
        max_delay=4,
        seed=333,
    )
    result = run_experiment(exp, "aesw")
    assert result.search_time > 0
    assert result.total_moves > 0


def test_false_positive_evidence_decay():
    """Verify unreinforced false positive cues decay in weight as age advances."""
    agent = AESWAgent(walker_id=0, seed=42, default_lambda_hat=0.1)

    # False positive signal at time 0
    obs_fp = _make_obs(
        walker_id=0,
        current_node="fp_node",
        time=0,
        neighbors=("other",),
        active_neighbors=("other",),
        detected=True,
        signal_strength=1.0,
    )
    agent.step(obs_fp)

    # Weight at t=0
    w_0 = agent.cache.get("fp_node", current_time=0)
    assert w_0 is not None
    assert w_0.weight == pytest.approx(1.0, rel=1e-5)

    # At t=10 without reinforcement, weight must have decayed
    w_10 = agent.cache.get("fp_node", current_time=10)
    assert w_10 is not None
    assert w_10.weight < w_0.weight
    assert w_10.weight == pytest.approx(np.exp(-agent.cache.default_lambda_hat * 10), rel=1e-3)


def test_success_semantics_colocation_vs_signal():
    """Verify positive signal != success, while physical co-location == success."""
    g_spec = GraphSpecification(graph_type="er", num_nodes=20, seed=42, parameters={"p": 0.3})
    static_graph = generate_graph(g_spec, seed=42)
    p_def = ProblemDefinition(
        graph=g_spec,
        target=TargetSpecification(mode=TargetMode.STATIC, p_move=0.0),
        detection=DetectionSpecification(signal_radius=10, p_d=1.0, p_fa=0.0),
        walker=WalkerSpecification(count=1),
        simulation=SimulationSpecification(time_horizon=20, seed=42),
    )
    coord_sep = SimulationCoordinator(
        problem_def=p_def,
        static_graph=static_graph,
        initial_target_node=0,
        initial_walker_nodes={0: 1},
        seed_dynamics=1,
        seed_target=2,
        seed_detection=3,
        seed_observation=4,
    )

    obs = coord_sep.get_observations()[0]
    assert obs.detected is True
    assert coord_sep._termination_status is None
    assert not coord_sep.is_terminated

    # Co-location
    coord_coloc = SimulationCoordinator(
        problem_def=p_def,
        static_graph=static_graph,
        initial_target_node=0,
        initial_walker_nodes={0: 0},
        seed_dynamics=1,
        seed_target=2,
        seed_detection=3,
        seed_observation=4,
    )
    assert coord_coloc.is_terminated is True
    assert coord_coloc.termination_status == SearchTerminationStatus.SUCCESS
