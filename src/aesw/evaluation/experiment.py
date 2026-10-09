"""
Reproducible Experiment Engine and Harness
==========================================
Implements the experimental harness and execution pipeline for Adaptive Graph Search.
Separates experiment generation (environment realization) from algorithm execution.

Pipeline Architecture:
Configuration -> Master Seed -> Experiment Instance -> Graph Generation
  -> Initial Dynamic State -> Target Locomotion Init -> Environment Configuration
  -> Algorithm Execution -> Step Loop -> Metrics Evaluation -> Structured Result -> JSON Artifact
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np

from aesw.environment.types import (
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
    DelaySpecification,
    WalkerSpecification,
    SimulationSpecification,
    CostSpecification,
)
from aesw.environment.coordinator import SimulationCoordinator
from aesw.graph.base import GraphInstance
from aesw.graph.factory import generate_graph
from aesw.utils.config import load_config
from aesw.utils.reproducibility import create_rng
from aesw.baselines.types import BaselineType
from aesw.baselines.base import BaselinePolicy
from aesw.baselines.factory import create_baseline
from aesw.baselines.independent_walkers import IndependentRandomWalkers
from aesw.evaluation.metrics import RunMetrics, AggregatedMetrics, compute_aggregated_metrics


@dataclass(frozen=True)
class ExperimentInstance:
    """Immutable representation of a generated, reproducible search experiment environment.

    The fundamental research requirement is that every compared algorithm must be
    evaluated on the exact same underlying experimental realization.

    Attributes:
        problem_def: Complete immutable ProblemDefinition contract.
        master_seed: Master integer random seed used to generate this instance.
        sub_seeds: Mapping of derived deterministic seeds for each subsystem.
        static_graph: Generated permanent static topology GraphInstance.
        initial_target_node: Starting vertex occupied by target x_0.
        initial_walker_nodes: Mapping from walker ID to starting vertex.
    """
    problem_def: ProblemDefinition
    master_seed: int
    sub_seeds: Mapping[str, int]
    static_graph: GraphInstance
    initial_target_node: int | str
    initial_walker_nodes: Mapping[int, int | str]

    def create_coordinator(self) -> SimulationCoordinator:
        """Create a fresh SimulationCoordinator configured with this exact environment.

        Every call returns a brand new coordinator with identical initial state and
        identical subsystem seeds, ensuring exact multi-algorithm comparison fairness.
        """
        return SimulationCoordinator(
            problem_def=self.problem_def,
            static_graph=self.static_graph,
            initial_target_node=self.initial_target_node,
            initial_walker_nodes=self.initial_walker_nodes,
            seed_dynamics=self.sub_seeds["dynamics"],
            seed_target=self.sub_seeds["target"],
            seed_detection=self.sub_seeds["detection"],
            seed_observation=self.sub_seeds["observation"],
        )


@dataclass(frozen=True)
class ExperimentResult:
    """Structured result produced by executing an algorithm on an ExperimentInstance.

    Contains full evaluation metrics, execution trajectory data, and serialization methods.

    Attributes:
        experiment_id: Unique identifier for this experiment execution.
        algorithm: Name of the evaluated search algorithm.
        seed: Master random seed used for this experiment.
        status: Final termination state (SUCCESS, BUDGET_EXHAUSTED, DISCONNECTED).
        success: Boolean flag indicating target acquisition within budget.
        search_time: Total simulation steps elapsed until termination.
        total_moves: Total physical edge traversals M across all walkers.
        total_messages: Total messages exchanged Q across all walkers.
        total_cost: Evaluated total cost C_total = M + c_m * Q.
        nodes_visited: Count of unique vertices inspected by searchers.
        node_revisits: Total repeated visits to previously inspected nodes.
        metrics: Formal RunMetrics object.
        target_trajectory: Sequence of node coordinates occupied by target over time.
        walker_trajectories: Sequence of node coordinates occupied by each walker.
        parameters: Summary of environment, graph, dynamics, and walker specifications.
        timestamp: ISO 8601 UTC timestamp of execution.
    """
    experiment_id: str
    algorithm: str
    seed: int
    status: SearchTerminationStatus
    success: bool
    search_time: int
    total_moves: int
    total_messages: int
    total_cost: float
    nodes_visited: int
    node_revisits: int
    metrics: RunMetrics
    target_trajectory: tuple[int | str, ...]
    walker_trajectories: Mapping[int, tuple[int | str, ...]]
    parameters: Mapping[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert result into a JSON-serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "status": self.status.value,
            "success": self.success,
            "search_time": self.search_time,
            "total_moves": self.total_moves,
            "total_messages": self.total_messages,
            "total_cost": self.total_cost,
            "nodes_visited": self.nodes_visited,
            "node_revisits": self.node_revisits,
            "metrics": {
                "algorithm": self.metrics.algorithm,
                "seed": self.metrics.seed,
                "status": self.metrics.status.value,
                "success": self.metrics.success,
                "search_time": self.metrics.search_time,
                "nodes_visited": self.metrics.nodes_visited,
                "node_revisits": self.metrics.node_revisits,
                "total_moves": self.metrics.total_moves,
                "total_messages": self.metrics.total_messages,
                "total_cost": self.metrics.total_cost,
            },
            "target_trajectory": list(self.target_trajectory),
            "walker_trajectories": {str(k): list(v) for k, v in self.walker_trajectories.items()},
            "parameters": dict(self.parameters),
            "timestamp": self.timestamp,
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialize result to a formatted JSON string, optionally writing to a file."""
        data_str = json.dumps(self.to_dict(), indent=indent)
        if filepath is not None:
            path = Path(filepath).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(data_str)
        return data_str

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Save this experiment result as a JSON artifact into the specified directory."""
        base_dir = Path("results" if output_dir is None else output_dir).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.experiment_id}.json"
        path = base_dir / filename
        self.to_json(path)
        return path


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated results across multiple stochastic runs and comparative algorithms.

    Attributes:
        name: Name of the benchmark suite.
        num_runs_per_algorithm: Number of independent runs executed per algorithm.
        seeds: Tuple of master seeds evaluated.
        run_results: All individual ExperimentResult instances.
        aggregated_metrics: Mapping from algorithm name to AggregatedMetrics summary.
        timestamp: ISO 8601 UTC timestamp of benchmark completion.
    """
    name: str
    num_runs_per_algorithm: int
    seeds: tuple[int, ...]
    run_results: tuple[ExperimentResult, ...]
    aggregated_metrics: Mapping[str, AggregatedMetrics]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark result into a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "num_runs_per_algorithm": self.num_runs_per_algorithm,
            "seeds": list(self.seeds),
            "aggregated_metrics": {
                algo: {
                    "algorithm": agg.algorithm,
                    "num_runs": agg.num_runs,
                    "success_rate": agg.success_rate,
                    "mean_search_time": agg.mean_search_time,
                    "std_search_time": agg.std_search_time,
                    "mean_total_cost": agg.mean_total_cost,
                    "std_total_cost": agg.std_total_cost,
                    "ci_lower_cost": agg.ci_lower_cost,
                    "ci_upper_cost": agg.ci_upper_cost,
                }
                for algo, agg in self.aggregated_metrics.items()
            },
            "run_count": len(self.run_results),
            "timestamp": self.timestamp,
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialize benchmark summary to JSON."""
        data_str = json.dumps(self.to_dict(), indent=indent)
        if filepath is not None:
            path = Path(filepath).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(data_str)
        return data_str

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Save benchmark summary artifact to JSON."""
        base_dir = Path("results" if output_dir is None else output_dir).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.name}_summary.json"
        path = base_dir / filename
        self.to_json(path)
        return path


def _apply_overrides(base_problem: ProblemDefinition, overrides: Mapping[str, Any]) -> ProblemDefinition:
    """Helper to update a ProblemDefinition with parameter overrides."""
    if not overrides:
        return base_problem

    # Graph spec overrides
    g = base_problem.graph
    g_type = str(overrides.get("graph_type", g.graph_type))
    n_nodes = int(overrides.get("num_nodes", g.num_nodes))
    g_params = dict(g.parameters)
    if "graph_parameters" in overrides:
        g_params.update(overrides["graph_parameters"])
    # Support inline family parameters
    for k in ("p", "m", "k", "rows", "cols", "obstacle_ratio", "radius", "dim"):
        if k in overrides:
            g_params[k] = overrides[k]

    new_graph = GraphSpecification(
        graph_type=g_type,
        num_nodes=n_nodes,
        seed=g.seed,
        parameters=g_params,
    )

    # Dynamics spec overrides
    d = base_problem.dynamics
    regime = overrides.get("regime", d.regime)
    if isinstance(regime, str):
        regime = DynamicRegime(regime.upper())
    p_on = float(overrides.get("p_on", d.p_on))
    p_off = float(overrides.get("p_off", d.p_off))
    new_dynamics = DynamicsSpecification(
        regime=regime,
        p_on=p_on,
        p_off=p_off,
        edge_update_behavior=d.edge_update_behavior,
    )

    # Target spec overrides
    t = base_problem.target
    t_mode = overrides.get("target_mode", t.mode)
    if isinstance(t_mode, str):
        t_mode = TargetMode(t_mode.upper())
    p_move = float(overrides.get("p_move", t.p_move))
    init_target = overrides.get("initial_target_node", overrides.get("initial_node", t.initial_node))
    new_target = TargetSpecification(
        mode=t_mode,
        p_move=p_move,
        initial_node=init_target,
    )

    # Detection spec overrides
    det = base_problem.detection
    s_rad = int(overrides.get("signal_radius", det.signal_radius))
    p_d = float(overrides.get("p_d", det.p_d))
    p_fa = float(overrides.get("p_fa", det.p_fa))
    new_detection = DetectionSpecification(
        signal_radius=s_rad,
        p_d=p_d,
        p_fa=p_fa,
    )

    # Observation spec overrides
    obs = base_problem.observation
    b = int(overrides.get("neighbor_budget", obs.neighbor_budget))
    new_obs = ObservationSpecification(neighbor_budget=b)

    # Walker spec overrides
    w = base_problem.walker
    w_count = int(overrides.get("walker_count", overrides.get("walkers", w.count)))
    new_walker = WalkerSpecification(count=w_count)

    # Simulation spec overrides
    sim = base_problem.simulation
    t_horiz = int(overrides.get("time_horizon", sim.time_horizon))
    new_sim = SimulationSpecification(
        time_horizon=t_horiz,
        seed=sim.seed,
    )

    # Cost spec overrides
    c = base_problem.cost
    c_m = float(overrides.get("c_m", c.c_m))
    new_cost = CostSpecification(c_m=c_m)

    return ProblemDefinition(
        graph=new_graph,
        dynamics=new_dynamics,
        target=new_target,
        detection=new_detection,
        observation=new_obs,
        delay=base_problem.delay,
        walker=new_walker,
        simulation=new_sim,
        cost=new_cost,
    )


def generate_experiment(
    config: Optional[Any] = None,
    seed: Optional[int] = 42,
    **overrides: Any,
) -> ExperimentInstance:
    """Generate a reproducible, controlled search experiment environment instance.

    Core Principle (Separation of Generation and Execution):
    Generates the static graph topology, derives isolated subsystem PRNG streams,
    and places initial targets and walkers before any algorithm executes.
    Multiple algorithms can subsequently run on this single ExperimentInstance
    to guarantee exact multi-agent evaluation fairness.

    Args:
        config: ProblemDefinition, dict, YAML config path, or None (defaults loaded).
        seed: Master random seed (defaults to 42).
        **overrides: Parameter overrides (e.g., graph_type, num_nodes, p_on, p_off, p_move, p_d, p_fa, walker_count).

    Returns:
        Immutable ExperimentInstance ready for algorithm benchmarking.
    """
    master_seed = 42 if seed is None else int(seed)

    # 1. Resolve base ProblemDefinition
    if config is None:
        cfg_path = Path("configs")
        if (cfg_path / "default.yaml").is_file():
            d_cfg = load_config(cfg_path / "default.yaml")
            g_cfg = load_config(cfg_path / "graph.yaml")
            dyn_cfg = load_config(cfg_path / "dynamics.yaml")
            exp_cfg = load_config(cfg_path / "experiments.yaml")
            base_problem = ProblemDefinition.from_configs(d_cfg, g_cfg, dyn_cfg, exp_cfg)
        else:
            base_problem = ProblemDefinition()
    elif isinstance(config, ProblemDefinition):
        base_problem = config
    elif isinstance(config, (str, Path)):
        c_path = Path(config).resolve()
        cfg_data = load_config(c_path)
        # If user passed experiments.yaml or similar dict
        base_problem = ProblemDefinition()
    elif isinstance(config, dict):
        base_problem = ProblemDefinition()
    else:
        raise TypeError(f"Unsupported config type: {type(config).__name__}")

    # Apply overrides
    problem_def = _apply_overrides(base_problem, overrides)

    # 2. Derive isolated deterministic seeds using NumPy SeedSequence
    ss = np.random.SeedSequence(master_seed)
    spawned = ss.spawn(7)
    sub_seeds: dict[str, int] = {
        "graph": int(spawned[0].generate_state(1)[0]),
        "dynamics": int(spawned[1].generate_state(1)[0]),
        "target": int(spawned[2].generate_state(1)[0]),
        "detection": int(spawned[3].generate_state(1)[0]),
        "observation": int(spawned[4].generate_state(1)[0]),
        "init": int(spawned[5].generate_state(1)[0]),
        "algorithm": int(spawned[6].generate_state(1)[0]),
    }

    # 3. Generate static graph topology
    static_graph = generate_graph(problem_def.graph, seed=sub_seeds["graph"])

    # 4. Deterministic initial target and walker placement
    init_rng = create_rng(sub_seeds["init"])
    candidate_nodes = sorted(list(static_graph.nodes.keys()), key=str)

    # Initial target node
    if problem_def.target.initial_node is not None and problem_def.target.initial_node in static_graph.nodes:
        initial_target = problem_def.target.initial_node
    else:
        target_idx = int(init_rng.integers(0, len(candidate_nodes)))
        initial_target = candidate_nodes[target_idx]

    # Initial walker nodes: distribute across candidate nodes distinct from target if possible
    num_walkers = problem_def.walker.count
    non_target_candidates = [n for n in candidate_nodes if n != initial_target]

    if len(non_target_candidates) >= num_walkers:
        chosen_indices = init_rng.choice(len(non_target_candidates), size=num_walkers, replace=False)
        initial_walkers = {i: non_target_candidates[idx] for i, idx in enumerate(chosen_indices)}
    else:
        chosen_indices = init_rng.choice(len(candidate_nodes), size=num_walkers, replace=True)
        initial_walkers = {i: candidate_nodes[idx] for i, idx in enumerate(chosen_indices)}

    return ExperimentInstance(
        problem_def=problem_def,
        master_seed=master_seed,
        sub_seeds=sub_seeds,
        static_graph=static_graph,
        initial_target_node=initial_target,
        initial_walker_nodes=initial_walkers,
    )


def _resolve_baseline_type(name: str) -> BaselineType:
    """Normalize string algorithm identifier to BaselineType enum."""
    clean = name.lower().replace("-", "_").strip()
    alias_map = {
        "random_walk": BaselineType.RANDOM_WALK,
        "non_backtracking": BaselineType.NON_BACKTRACKING,
        "non_backtracking_walk": BaselineType.NON_BACKTRACKING,
        "independent_random_walkers": BaselineType.INDEPENDENT_RANDOM_WALKERS,
        "k_independent_random_walkers": BaselineType.INDEPENDENT_RANDOM_WALKERS,
        "degree_based": BaselineType.DEGREE_BASED,
        "degree_based_walk": BaselineType.DEGREE_BASED,
        "flooding": BaselineType.FLOODING,
        "ant_colony": BaselineType.ANT_COLONY,
        "ant_colony_walk": BaselineType.ANT_COLONY,
    }
    if clean in alias_map:
        return alias_map[clean]
    try:
        return BaselineType(clean)
    except ValueError:
        valid = list(alias_map.keys())
        raise ValueError(f"Unknown baseline algorithm '{name}'. Supported algorithms: {valid}")


def run_experiment(
    experiment: ExperimentInstance,
    algorithm: Union[str, BaselineType, BaselinePolicy, Sequence[BaselinePolicy]],
    output_dir: Optional[Union[str, Path]] = None,
    save_artifact: bool = False,
    **kwargs: Any,
) -> ExperimentResult:
    """Execute a search algorithm on the provided ExperimentInstance.

    Args:
        experiment: The generated ExperimentInstance.
        algorithm: Algorithm name (e.g., 'random_walk'), BaselineType enum, or instantiated policy.
        output_dir: Optional directory to save the JSON artifact.
        save_artifact: Whether to write the JSON result to disk (default False, True if output_dir set).
        **kwargs: Policy-specific hyperparameter overrides (e.g. epsilon, evaporation_rate).

    Returns:
        Structured ExperimentResult containing complete metrics and trajectories.
    """
    coordinator = experiment.create_coordinator()
    num_walkers = experiment.problem_def.walker.count
    base_algo_seed = experiment.sub_seeds["algorithm"]

    multi_walker_policy: Optional[IndependentRandomWalkers] = None
    walker_policies: dict[int, BaselinePolicy] = {}

    if isinstance(algorithm, str):
        b_type = _resolve_baseline_type(algorithm)
        algo_name = b_type.value

        if b_type == BaselineType.INDEPENDENT_RANDOM_WALKERS:
            multi_walker_policy = IndependentRandomWalkers(k=num_walkers, base_seed=base_algo_seed)
        else:
            # Instantiate policies for each walker with isolated seeds
            for w_id in range(num_walkers):
                w_seed = base_algo_seed + w_id * 1000
                walker_policies[w_id] = create_baseline(b_type, seed=w_seed, **kwargs)

    elif isinstance(algorithm, BaselineType):
        algo_name = algorithm.value
        if algorithm == BaselineType.INDEPENDENT_RANDOM_WALKERS:
            multi_walker_policy = IndependentRandomWalkers(k=num_walkers, base_seed=base_algo_seed)
        else:
            for w_id in range(num_walkers):
                w_seed = base_algo_seed + w_id * 1000
                walker_policies[w_id] = create_baseline(algorithm, seed=w_seed, **kwargs)

    elif isinstance(algorithm, IndependentRandomWalkers):
        algo_name = algorithm.metadata.name
        multi_walker_policy = algorithm
        multi_walker_policy.reset()

    elif isinstance(algorithm, BaselinePolicy):
        algo_name = algorithm.metadata.name
        algorithm.reset()
        for w_id in range(num_walkers):
            walker_policies[w_id] = algorithm

    elif isinstance(algorithm, (list, tuple)):
        if len(algorithm) != num_walkers:
            raise ValueError(f"Expected {num_walkers} policies for walkers, got {len(algorithm)}")
        algo_name = algorithm[0].metadata.name
        for w_id, pol in enumerate(algorithm):
            pol.reset()
            walker_policies[w_id] = pol
    else:
        raise TypeError(f"Unsupported algorithm type: {type(algorithm).__name__}")

    # Simulation execution loop
    while not coordinator.is_terminated:
        observations = coordinator.get_observations()
        actions: dict[int, Any] = {}

        if multi_walker_policy is not None:
            actions = multi_walker_policy.decide_all(observations)
        else:
            for w_id, obs in observations.items():
                if w_id in walker_policies:
                    actions[w_id] = walker_policies[w_id].decide(obs)

        coordinator.step(actions)

    # Collect metrics
    metrics = coordinator.get_run_metrics(
        algorithm_name=algo_name,
        seed=experiment.master_seed,
    )

    parameters = {
        "graph": experiment.problem_def.graph.parameters | {
            "type": experiment.problem_def.graph.graph_type,
            "num_nodes": experiment.problem_def.graph.num_nodes,
        },
        "dynamics": {
            "regime": experiment.problem_def.dynamics.regime.value,
            "p_on": experiment.problem_def.dynamics.p_on,
            "p_off": experiment.problem_def.dynamics.p_off,
        },
        "target": {
            "mode": experiment.problem_def.target.mode.value,
            "p_move": experiment.problem_def.target.p_move,
            "initial_node": experiment.initial_target_node,
        },
        "detection": {
            "signal_radius": experiment.problem_def.detection.signal_radius,
            "p_d": experiment.problem_def.detection.p_d,
            "p_fa": experiment.problem_def.detection.p_fa,
        },
        "walker": {
            "count": experiment.problem_def.walker.count,
            "neighbor_budget": experiment.problem_def.observation.neighbor_budget,
            "initial_nodes": dict(experiment.initial_walker_nodes),
        },
        "cost": {
            "c_m": experiment.problem_def.cost.c_m,
        },
    }

    exp_id = f"exp_{experiment.master_seed:04d}_{algo_name}"
    result = ExperimentResult(
        experiment_id=exp_id,
        algorithm=algo_name,
        seed=experiment.master_seed,
        status=metrics.status,
        success=metrics.success,
        search_time=metrics.search_time,
        total_moves=metrics.total_moves,
        total_messages=metrics.total_messages,
        total_cost=metrics.total_cost,
        nodes_visited=metrics.nodes_visited,
        node_revisits=metrics.node_revisits,
        metrics=metrics,
        target_trajectory=coordinator.target_trajectory,
        walker_trajectories=coordinator.walker_trajectories,
        parameters=parameters,
    )

    if save_artifact or (output_dir is not None):
        result.save(output_dir)

    return result


def run_benchmark(
    config: Optional[Any] = None,
    algorithms: Optional[Sequence[Union[str, BaselineType]]] = None,
    seeds: Optional[Sequence[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    suite_name: str = "baseline_benchmark",
    **overrides: Any,
) -> BenchmarkResult:
    """Run an automated benchmark suite across algorithms and stochastic seeds.

    Evaluates all algorithms on identical experiment instances for each seed,
    aggregates results, and writes structured JSON artifacts.

    Args:
        config: ProblemDefinition, dict, YAML config path, or None.
        algorithms: Sequence of algorithm names or BaselineType enums to evaluate.
        seeds: Sequence of master seeds (defaults to [42, 43, 44]).
        output_dir: Directory where JSON run and summary artifacts are saved.
        suite_name: Identifier for the benchmark suite.
        **overrides: Parameter overrides passed to generate_experiment.

    Returns:
        BenchmarkResult encapsulating all individual and aggregated metrics.
    """
    target_algos = algorithms or [
        "random_walk",
        "non_backtracking",
        "degree_based",
        "flooding",
        "ant_colony",
        "independent_random_walkers",
    ]
    target_seeds = tuple(seeds or [42, 43, 44])

    all_run_results: list[ExperimentResult] = []
    algo_runs: dict[str, list[RunMetrics]] = {}

    for s in target_seeds:
        # Generate the shared experimental environment for this seed
        exp_instance = generate_experiment(config=config, seed=s, **overrides)

        for algo in target_algos:
            res = run_experiment(
                experiment=exp_instance,
                algorithm=algo,
                output_dir=output_dir,
                save_artifact=(output_dir is not None),
            )
            all_run_results.append(res)
            algo_key = res.algorithm
            algo_runs.setdefault(algo_key, []).append(res.metrics)

    # Compute aggregated metrics per algorithm
    aggregated: dict[str, AggregatedMetrics] = {}
    for algo_key, runs in algo_runs.items():
        aggregated[algo_key] = compute_aggregated_metrics(runs, algorithm=algo_key)

    benchmark_res = BenchmarkResult(
        name=suite_name,
        num_runs_per_algorithm=len(target_seeds),
        seeds=target_seeds,
        run_results=tuple(all_run_results),
        aggregated_metrics=aggregated,
    )

    if output_dir is not None:
        benchmark_res.save(output_dir)

    return benchmark_res
