"""
Observation Builder and Partial Visibility Engine
=================================================
Constructs partial, local, noisy observations for walkers on dynamic graphs.
Acts as a strict information firewall between the unobserved ground-truth
environment state and walker decision-making.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Sequence
import numpy as np

from aesw.utils.reproducibility import create_rng
from aesw.environment.types import EdgeState
from aesw.environment.models import Node, WalkerState
from aesw.environment.state import GroundTruthState
from aesw.environment.problem import ObservationSpecification
from aesw.environment.detection import DetectionEngine
from aesw.environment.observation import (
    Observation,
    ObservedEdgeInfo,
    TargetSignalObservation,
    LocalObservationHistory,
)

if TYPE_CHECKING:
    from aesw.dynamics.view import ActiveGraphView


class _ActiveGraphViewFromGroundTruth:
    """Lightweight adapter presenting an ActiveGraphView-compatible interface over GroundTruthState."""

    def __init__(self, ground_truth: GroundTruthState) -> None:
        self._ground_truth = ground_truth

    @property
    def time(self) -> int:
        return self._ground_truth.time

    @property
    def node_count(self) -> int:
        return self._ground_truth.num_nodes

    @property
    def nodes(self) -> Mapping[int | str, Node]:
        return self._ground_truth.nodes

    def active_neighbors(self, node_id: int | str) -> list[int | str]:
        neighbors: list[int | str] = []
        for edge in self._ground_truth.edges:
            if edge.source == node_id:
                nbr = edge.target
            elif edge.target == node_id and not edge.is_directed:
                nbr = edge.source
            else:
                continue
            pair = frozenset([node_id, nbr])
            if self._ground_truth.active_edge_states.get(pair, edge.state) == EdgeState.ON:
                neighbors.append(nbr)
        return neighbors

    def has_active_edge(self, u: int | str, v: int | str) -> bool:
        pair = frozenset([u, v])
        return self._ground_truth.active_edge_states.get(pair) == EdgeState.ON

    def is_edge_active(self, u: int | str, v: int | str) -> bool:
        return self.has_active_edge(u, v)


class ObservationBuilder:
    """Constructs partial, local, noisy observations for walkers.

    Acts as an information firewall between the ground-truth environment state
    and the walker's perception:
    - Bounded neighbor inspection budget B
    - Active edge visibility constraint (no multi-hop or global topology knowledge)
    - Binary target detection signal reduction (hides ground-truth classification,
      distance, and target position)
    - Fully decoupled, reproducible PRNG stream
    - Immutable snapshot generation
    """

    def __init__(
        self,
        observation_spec: Optional[ObservationSpecification] = None,
        neighbor_budget: Optional[int] = None,
        detection_engine: Optional[DetectionEngine] = None,
        seed: Optional[int] = None,
    ) -> None:
        if observation_spec is not None:
            self._neighbor_budget = observation_spec.neighbor_budget
        elif neighbor_budget is not None:
            if neighbor_budget <= 0:
                raise ValueError(f"neighbor_budget B must be strictly positive, got {neighbor_budget}")
            self._neighbor_budget = neighbor_budget
        else:
            self._neighbor_budget = 4

        self._detection_engine = detection_engine
        self._seed = seed
        self._rng: np.random.Generator = create_rng(seed)

    @property
    def neighbor_budget(self) -> int:
        """Maximum neighbor inspection budget B."""
        return self._neighbor_budget

    @property
    def seed(self) -> Optional[int]:
        """PRNG seed for observation sampling."""
        return self._seed

    def build_observation(
        self,
        walker: WalkerState,
        active_graph: ActiveGraphView | _ActiveGraphViewFromGroundTruth,
        target_node: int | str,
        time: Optional[int] = None,
        history: Optional[LocalObservationHistory] = None,
        candidate_neighbors: Optional[Sequence[int | str]] = None,
    ) -> Observation:
        """Build an immutable, partial observation for a walker.

        Args:
            walker: State of the observing walker.
            active_graph: Read-only active graph view G_t = (V, E_t).
            target_node: Ground-truth target location x_t (used ONLY internally for sensor probe).
            time: Simulation timestamp t. If None, defaults to walker.busy_until_time or 0.
            history: Optional local observation history prior to this observation.
            candidate_neighbors: Optional explicit candidate subset to inspect. If None,
                defaults to all active neighbors of walker.current_node in G_t.

        Returns:
            Immutable Observation snapshot conforming to the Controlled Observation Principle.
        """
        current_node = walker.current_node
        current_time = time if time is not None else getattr(walker, "busy_until_time", 0)

        # 1. Determine local visible candidates
        if candidate_neighbors is not None:
            raw_candidates = list(candidate_neighbors)
        else:
            raw_candidates = list(active_graph.active_neighbors(current_node))

        # Deterministic sorting prior to stochastic selection guarantees platform invariance
        candidates = sorted(raw_candidates, key=str)

        # 2. Apply neighbor checking budget B
        b = self._neighbor_budget
        if len(candidates) <= b:
            checked = tuple(candidates)
        else:
            # Uniform sampling without replacement
            indices = self._rng.choice(len(candidates), size=b, replace=False)
            checked = tuple(candidates[i] for i in sorted(indices))

        # 3. Observe local edge states for checked neighbors
        observed_edges: dict[int | str, ObservedEdgeInfo] = {}
        for v in checked:
            is_active = active_graph.has_active_edge(current_node, v)
            edge_state = EdgeState.ON if is_active else EdgeState.OFF
            observed_edges[v] = ObservedEdgeInfo(
                neighbor_id=v,
                state=edge_state,
                observed_at_time=current_time,
            )

        # 4. Probe target detection sensor if detection engine is provided
        if self._detection_engine is not None:
            det_res = self._detection_engine.detect(
                observer_node=current_node,
                target_node=target_node,
                active_graph=active_graph,  # type: ignore[arg-type]
                timestamp=current_time,
            )
            detected = bool(det_res.signal)
        else:
            detected = False

        # Strictly walker-visible binary signal: outcome_category is None!
        target_signal = TargetSignalObservation(
            detected=detected,
            signal_strength=1.0 if detected else 0.0,
            outcome_category=None,
            timestamp=current_time,
        )

        # 5. Construct and return immutable Observation
        obs = Observation(
            walker_id=walker.walker_id,
            current_node=current_node,
            time=current_time,
            checked_neighbors=checked,
            observed_edges=observed_edges,
            target_signal=target_signal,
            neighbor_budget_used=len(checked),
            visible_neighbors=checked,
            observed_signals={current_node: detected},
            local_history=history,
        )
        return obs

    def observe_and_update_history(
        self,
        walker: WalkerState,
        active_graph: ActiveGraphView | _ActiveGraphViewFromGroundTruth,
        target_node: int | str,
        time: Optional[int] = None,
        history: Optional[LocalObservationHistory] = None,
        candidate_neighbors: Optional[Sequence[int | str]] = None,
    ) -> tuple[Observation, LocalObservationHistory]:
        """Build observation and append it to local history."""
        obs = self.build_observation(
            walker=walker,
            active_graph=active_graph,
            target_node=target_node,
            time=time,
            history=history,
            candidate_neighbors=candidate_neighbors,
        )
        current_history = history if history is not None else LocalObservationHistory()
        new_history = current_history.add(obs)
        return obs, new_history

    def build_from_ground_truth(
        self,
        walker_id: int | str,
        ground_truth: GroundTruthState,
        active_graph: Optional[ActiveGraphView] = None,
        history: Optional[LocalObservationHistory] = None,
        candidate_neighbors: Optional[Sequence[int | str]] = None,
    ) -> Observation:
        """Construct an observation directly from GroundTruthState.

        Walkers must never access GroundTruthState directly. This method is called
        by the simulation coordinator / test harness to construct the projected
        Observation.
        """
        if walker_id not in ground_truth.walker_states:
            raise KeyError(f"Walker '{walker_id}' not found in GroundTruthState")
        walker = ground_truth.walker_states[walker_id]
        target_node = ground_truth.target_state.current_node

        graph_view = active_graph if active_graph is not None else _ActiveGraphViewFromGroundTruth(ground_truth)

        return self.build_observation(
            walker=walker,
            active_graph=graph_view,
            target_node=target_node,
            time=ground_truth.time,
            history=history,
            candidate_neighbors=candidate_neighbors,
        )
