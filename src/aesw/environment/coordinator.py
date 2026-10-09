"""
Simulation Environment Coordinator
==================================
Central discrete-time simulation coordinator orchestrating dynamic edge transitions,
target locomotion, partial observation generation, walker action execution,
and evaluation metrics collection under strict epistemic boundaries.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence
import numpy as np

from aesw.environment.types import (
    EdgeState,
    SearchTerminationStatus,
)
from aesw.environment.models import Node, Edge, WalkerState, TargetState
from aesw.environment.state import GroundTruthState
from aesw.environment.problem import ProblemDefinition
from aesw.environment.target import TargetEngine
from aesw.environment.detection import DetectionEngine
from aesw.environment.observation import Observation, LocalObservationHistory
from aesw.environment.builder import ObservationBuilder
from aesw.dynamics.engine import DynamicGraphState
from aesw.dynamics.view import ActiveGraphView
from aesw.dynamics.factory import create_dynamic_graph_from_spec
from aesw.graph.base import GraphInstance
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aesw.baselines.actions import SearchAction
    from aesw.evaluation.metrics import RunMetrics






class SimulationCoordinator:
    """Central discrete-time coordinator managing the dynamic search simulation.

    Enforces the Controlled Observation Principle:
    - Ground-truth state (hidden target location, global topology, future edge transitions)
      is strictly encapsulated internally.
    - Walkers receive only partial, local, noisy Observation snapshots via `get_observations()`.
    - Physical actions (MOVE, STAY) are validated against the receiving walker's observation
      before mutating the environment state.
    """

    def __init__(
        self,
        problem_def: ProblemDefinition,
        static_graph: GraphInstance,
        initial_target_node: int | str,
        initial_walker_nodes: Mapping[int, int | str],
        seed_dynamics: Optional[int] = None,
        seed_target: Optional[int] = None,
        seed_detection: Optional[int] = None,
        seed_observation: Optional[int] = None,
        seed_jump: Optional[int] = None,
    ) -> None:
        """Initialize the simulation environment coordinator.

        Args:
            problem_def: Immutable problem definition specification.
            static_graph: Underlying permanent graph topology G = (V, E).
            initial_target_node: Initial vertex occupied by target x_0.
            initial_walker_nodes: Initial vertex positions for each walker ID.
            seed_dynamics: Seed for dynamic edge transitions.
            seed_target: Seed for target locomotion.
            seed_detection: Seed for detection noise.
            seed_observation: Seed for observation candidate sampling.
            seed_jump: Seed for environment-mediated long jumps.
        """
        self._problem_def = problem_def
        self._static_graph = static_graph
        self._time = 0
        self._time_horizon = problem_def.simulation.time_horizon
        self._jump_rng = np.random.default_rng(seed_jump if seed_jump is not None else 9999)

        # Initialize dynamic edge transition engine
        self._dynamic_graph: DynamicGraphState = create_dynamic_graph_from_spec(
            static_graph=self._static_graph,
            spec=self._problem_def.dynamics,
            seed=seed_dynamics,
        )
        self._active_view: ActiveGraphView = ActiveGraphView(self._dynamic_graph)

        # Initialize target locomotion engine
        self._target_engine: TargetEngine = TargetEngine(
            spec=self._problem_def.target,
            initial_node=initial_target_node,
            candidate_nodes=list(self._static_graph.nodes.keys()),
            seed=seed_target,
        )

        # Initialize detection engine
        self._detection_engine: DetectionEngine = DetectionEngine(
            spec=self._problem_def.detection,
            seed=seed_detection,
        )

        # Initialize observation builder
        self._observation_builder: ObservationBuilder = ObservationBuilder(
            observation_spec=self._problem_def.observation,
            detection_engine=self._detection_engine,
            seed=seed_observation,
        )

        # Initialize walkers
        self._walker_states: dict[int, WalkerState] = {}
        self._walker_histories: dict[int, LocalObservationHistory] = {}
        self._trajectories: dict[int, list[int | str]] = {}
        self._visited_nodes: set[int | str] = set()

        for w_id, start_node in initial_walker_nodes.items():
            if start_node not in self._static_graph.nodes:
                raise KeyError(f"Initial walker node '{start_node}' not found in graph topology")
            self._walker_states[w_id] = WalkerState(
                walker_id=w_id,
                current_node=start_node,
                step_count=0,
                message_count=0,
                busy_until_time=0,
            )
            self._walker_histories[w_id] = LocalObservationHistory()
            self._trajectories[w_id] = [start_node]
            self._visited_nodes.add(start_node)

        # Performance and tracking counters
        self._total_moves = 0
        self._total_messages = 0
        self._revisit_count = 0
        self._termination_status: Optional[SearchTerminationStatus] = None
        self._target_trajectory: list[int | str] = [self._target_engine.current_node]
        self._current_observations: Optional[dict[int, Observation]] = None


        # Check immediate initial acquisition
        for w_state in self._walker_states.values():
            if w_state.current_node == self._target_engine.current_node:
                self._termination_status = SearchTerminationStatus.SUCCESS
                break

    @property
    def time(self) -> int:
        """Current discrete simulation timestamp t."""
        return self._time

    @property
    def is_terminated(self) -> bool:
        """Whether the search episode has reached a terminal state."""
        return self._termination_status is not None

    @property
    def termination_status(self) -> Optional[SearchTerminationStatus]:
        """Final SearchTerminationStatus if terminated, else None."""
        return self._termination_status

    @property
    def walker_states(self) -> Mapping[int, WalkerState]:
        """Read-only mapping of current walker states."""
        return dict(self._walker_states)

    @property
    def target_state(self) -> TargetState:
        """Current ground-truth target state (internal verification only)."""
        return self._target_engine.state

    @property
    def visited_nodes(self) -> set[int | str]:
        """Set of unique vertices inspected or visited by walkers."""
        return set(self._visited_nodes)

    @property
    def revisit_count(self) -> int:
        """Total repeated visits to previously inspected nodes."""
        return self._revisit_count

    @property
    def total_moves(self) -> int:
        """Total physical moves M executed across all walkers."""
        return self._total_moves

    @property
    def total_messages(self) -> int:
        """Total messages Q exchanged across all walkers."""
        return self._total_messages

    def record_messages(self, count: int) -> None:
        """Record communication message overhead Q incurred by search policies.

        Args:
            count: Number of messages to add to cumulative Q.
        """
        if count < 0:
            raise ValueError(f"Message count must be non-negative, got {count}")
        self._total_messages += int(count)

    @property
    def walker_trajectories(self) -> Mapping[int, tuple[int | str, ...]]:
        """Complete trajectory history for each walker."""
        return {w_id: tuple(traj) for w_id, traj in self._trajectories.items()}

    @property
    def target_trajectory(self) -> tuple[int | str, ...]:
        """Complete ground-truth trajectory history of the target."""
        return tuple(self._target_trajectory)

    def get_ground_truth_state(self) -> GroundTruthState:
        """Construct an immutable GroundTruthState snapshot for auditing and analysis.

        CRITICAL: Never expose this object to search policies or walkers during execution.
        """
        edge_states = self._dynamic_graph.edge_states
        edges = tuple(
            Edge(
                source=e.source,
                target=e.target,
                state=edge_states.get(e.endpoints, EdgeState.ON),
                is_directed=e.is_directed,
            )
            for e in self._static_graph.edges
        )
        return GroundTruthState(
            time=self._time,
            nodes=self._static_graph.nodes,
            edges=edges,
            active_edge_states=edge_states,
            target_state=self._target_engine.state,
            walker_states=dict(self._walker_states),
        )

    def get_observations(self) -> dict[int, Observation]:
        """Generate or retrieve partial, local, noisy observations for all walkers at step t.

        Cached per discrete step t to ensure policy decisions and coordinator
        action validations inspect the exact same observation snapshot.
        """
        if self._current_observations is not None:
            return self._current_observations

        observations: dict[int, Observation] = {}
        target_node = self._target_engine.current_node

        for w_id, w_state in self._walker_states.items():
            history = self._walker_histories[w_id]
            obs = self._observation_builder.build_observation(
                walker=w_state,
                active_graph=self._active_view,
                target_node=target_node,
                time=self._time,
                history=history,
            )
            observations[w_id] = obs

        self._current_observations = observations
        return observations


    def step(
        self,
        actions: Mapping[int, SearchAction],
    ) -> tuple[dict[int, Observation], bool, Optional[SearchTerminationStatus]]:
        """Execute one discrete simulation step.

        Lifecycle:
        1. If already terminated, return immediately.
        2. Validate walker actions against current observations.
        3. Execute valid actions (MOVE / STAY) and update physical positions.
        4. Check for target acquisition at destination.
        5. Advance dynamic graph edge states G_t -> G_{t+1}.
        6. Advance target locomotion x_t -> x_{t+1}.
        7. Advance simulation time t -> t + 1.
        8. Evaluate termination (SUCCESS, BUDGET_EXHAUSTED, DISCONNECTED).
        9. Generate new observations for step t+1.

        Args:
            actions: Mapping from walker ID to proposed SearchAction.

        Returns:
            Tuple of (new_observations, is_terminated, termination_status).
        """
        from aesw.baselines.actions import validate_action
        from aesw.baselines.types import ActionType

        if self._termination_status is not None:

            return self.get_observations(), True, self._termination_status

        # 1. Obtain current observations for validation and history update

        current_obs = self.get_observations()

        # 2. Execute walker actions
        for w_id, w_state in self._walker_states.items():
            # If walker is currently busy traversing a delayed edge, skip
            if self._time < w_state.busy_until_time:
                continue

            if w_id not in actions:
                continue

            action = actions[w_id]
            obs = current_obs[w_id]

            # Enforce validation
            validate_action(action, obs)

            # Record observation into local history
            self._walker_histories[w_id] = self._walker_histories[w_id].add(obs)

            if action.action_type == ActionType.MOVE:
                dest = action.destination
                if dest is None:
                    continue

                # Traversal delay calculation (base model delay = 1)
                delay = int(self._problem_def.delay.min_delay)

                self._total_moves += 1
                new_step_count = w_state.step_count + 1

                if dest in self._visited_nodes:
                    self._revisit_count += 1
                else:
                    self._visited_nodes.add(dest)

                self._walker_states[w_id] = WalkerState(
                    walker_id=w_id,
                    current_node=dest,
                    step_count=new_step_count,
                    message_count=w_state.message_count,
                    busy_until_time=self._time + delay,
                )
                self._trajectories[w_id].append(dest)

                # Check immediate acquisition at destination
                if dest == self._target_engine.current_node:
                    self._termination_status = SearchTerminationStatus.SUCCESS

            elif action.action_type == ActionType.STAY:
                # Walker remains stationary
                pass

            elif action.action_type == ActionType.JUMP:
                # Environment-mediated long jump
                nodes = sorted(list(self._static_graph.nodes.keys()), key=str)
                if len(nodes) > 1 and w_state.current_node in nodes:
                    other_nodes = [n for n in nodes if n != w_state.current_node]
                    dest = other_nodes[int(self._jump_rng.integers(0, len(other_nodes)))]
                else:
                    dest = nodes[int(self._jump_rng.integers(0, len(nodes)))]

                delay = int(self._problem_def.delay.min_delay)
                self._total_moves += 1
                new_step_count = w_state.step_count + 1

                if dest in self._visited_nodes:
                    self._revisit_count += 1
                else:
                    self._visited_nodes.add(dest)

                self._walker_states[w_id] = WalkerState(
                    walker_id=w_id,
                    current_node=dest,
                    step_count=new_step_count,
                    message_count=w_state.message_count,
                    busy_until_time=self._time + delay,
                )
                self._trajectories[w_id].append(dest)

                # Check immediate acquisition at destination
                if dest == self._target_engine.current_node:
                    self._termination_status = SearchTerminationStatus.SUCCESS


        if self._termination_status == SearchTerminationStatus.SUCCESS:
            return self.get_observations(), True, SearchTerminationStatus.SUCCESS

        # 3. Advance dynamic environment
        self._dynamic_graph.advance()
        self._target_engine.step(self._active_view)
        self._time += 1
        self._current_observations = None
        self._target_trajectory.append(self._target_engine.current_node)


        # 4. Check if target moved onto any walker position
        for w_state in self._walker_states.values():
            if w_state.current_node == self._target_engine.current_node:
                self._termination_status = SearchTerminationStatus.SUCCESS
                return self.get_observations(), True, SearchTerminationStatus.SUCCESS

        # 5. Check time horizon budget exhaustion
        if self._time >= self._time_horizon:
            self._termination_status = SearchTerminationStatus.BUDGET_EXHAUSTED
            return self.get_observations(), True, SearchTerminationStatus.BUDGET_EXHAUSTED

        # 6. Check disconnection condition for static/degrading topologies
        if self._problem_def.dynamics.p_on == 0.0:
            target_curr = self._target_engine.current_node
            all_disconnected = True
            for w_state in self._walker_states.values():
                dist = self._detection_engine.compute_distance(
                    w_state.current_node, target_curr, self._active_view
                )
                if dist < float("inf"):
                    all_disconnected = False
                    break
            if all_disconnected:
                self._termination_status = SearchTerminationStatus.DISCONNECTED
                return self.get_observations(), True, SearchTerminationStatus.DISCONNECTED

        return self.get_observations(), False, None

    def get_run_metrics(self, algorithm_name: str, seed: int) -> RunMetrics:
        """Construct formal RunMetrics for this completed or ongoing simulation run.

        Args:
            algorithm_name: Algorithm identifier.
            seed: Master random seed for this execution.

        Returns:
            Validated RunMetrics instance.
        """
        from aesw.evaluation.metrics import RunMetrics

        status = self._termination_status or SearchTerminationStatus.BUDGET_EXHAUSTED

        success = (status == SearchTerminationStatus.SUCCESS)
        total_cost = self._problem_def.cost.compute_cost(
            num_moves=self._total_moves,
            num_messages=self._total_messages,
        )

        return RunMetrics(
            algorithm=algorithm_name,
            seed=seed,
            status=status,
            success=success,
            search_time=self._time,
            nodes_visited=len(self._visited_nodes),
            node_revisits=self._revisit_count,
            total_moves=self._total_moves,
            total_messages=self._total_messages,
            total_cost=total_cost,
        )
