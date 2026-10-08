"""
Dynamic Edge Transition Engine
==============================
Implements the stochastic discrete-time edge transition model G_t = (V, E_t).

Mathematical Model:
For every edge e in E independently:
- P(s_{t+1}(e) = OFF | s_t(e) = ON) = p_off
- P(s_{t+1}(e) = ON  | s_t(e) = OFF) = p_on

Underlying static topology G = (V, E) remains strictly immutable.
Edge transitions are executed synchronously without order dependence.
"""

from typing import Optional, Mapping
import numpy as np

from aesw.environment.types import EdgeState, DynamicRegime
from aesw.graph.base import GraphInstance
from aesw.dynamics.types import InitializationPolicy
from aesw.dynamics.models import TransitionStatistics, DynamicGraphSnapshot
from aesw.utils.reproducibility import create_rng


class DynamicGraphState:
    """Dynamic graph state and Markovian edge transition manager.

    Maintains time-varying edge availability G_t = (V, E_t) over an immutable
    underlying static GraphInstance G = (V, E).

    Attributes:
        static_graph: The underlying permanent GraphInstance (immutable).
        regime: Named churn regime (STATIC, SLOW, MEDIUM, FAST, VERY_FAST).
        p_on: Probability of transition from OFF to ON.
        p_off: Probability of transition from ON to OFF.
        seed: Random seed for this dynamic transition sequence.
    """

    def __init__(
        self,
        static_graph: GraphInstance,
        p_on: float = 0.05,
        p_off: float = 0.025,
        regime: DynamicRegime = DynamicRegime.MEDIUM,
        initialization_policy: InitializationPolicy = InitializationPolicy.ALL_ON,
        seed: Optional[int] = 42,
    ) -> None:
        if static_graph is None:
            raise ValueError("static_graph must not be None")
        if not (0.0 <= p_on <= 1.0):
            raise ValueError(f"p_on must be in [0.0, 1.0], got {p_on}")
        if not (0.0 <= p_off <= 1.0):
            raise ValueError(f"p_off must be in [0.0, 1.0], got {p_off}")
        if not isinstance(regime, DynamicRegime):
            raise TypeError(f"regime must be a DynamicRegime instance, got {type(regime).__name__}")
        if not isinstance(initialization_policy, InitializationPolicy):
            raise TypeError(f"initialization_policy must be an InitializationPolicy instance, got {type(initialization_policy).__name__}")

        self._static_graph: GraphInstance = static_graph
        self._regime: DynamicRegime = regime
        self._p_on: float = float(p_on)
        self._p_off: float = float(p_off)
        self._seed: int = 42 if seed is None else int(seed)
        self._rng: np.random.Generator = create_rng(self._seed)

        self._time: int = 0
        self._history: list[TransitionStatistics] = []

        # Build stable ordered list of edge endpoint keys
        # We preserve undirected frozenset representation for edge keys
        self._edge_keys: list[frozenset[int | str]] = [
            e.endpoints for e in self._static_graph.edges
        ]

        # Initialize edge states
        self._edge_states: dict[frozenset[int | str], EdgeState] = {}
        self._initialize_states(initialization_policy)

    def _initialize_states(self, policy: InitializationPolicy) -> None:
        """Initialize all edge states at time t = 0."""
        if policy == InitializationPolicy.ALL_ON:
            for k in self._edge_keys:
                self._edge_states[k] = EdgeState.ON

        elif policy == InitializationPolicy.ALL_OFF:
            for k in self._edge_keys:
                self._edge_states[k] = EdgeState.OFF

        elif policy == InitializationPolicy.STATIONARY:
            denom = self._p_on + self._p_off
            if denom <= 0.0:
                # Static regime with p_on = p_off = 0: default to ALL_ON
                pi_on = 1.0
            else:
                pi_on = self._p_on / denom

            for k in self._edge_keys:
                u = self._rng.random()
                self._edge_states[k] = EdgeState.ON if u < pi_on else EdgeState.OFF

    @property
    def static_graph(self) -> GraphInstance:
        """The permanent underlying static graph topology G = (V, E)."""
        return self._static_graph

    @property
    def time(self) -> int:
        """Current discrete simulation step t."""
        return self._time

    @property
    def regime(self) -> DynamicRegime:
        """Active dynamic regime."""
        return self._regime

    @property
    def p_on(self) -> float:
        """Activation transition probability p_on."""
        return self._p_on

    @property
    def p_off(self) -> float:
        """Deactivation transition probability p_off."""
        return self._p_off

    @property
    def seed(self) -> int:
        """RNG seed assigned to this dynamic process."""
        return self._seed

    @property
    def edge_states(self) -> Mapping[frozenset[int | str], EdgeState]:
        """Read-only mapping of edge states at current step t."""
        return dict(self._edge_states)

    def is_edge_active(self, u: int | str, v: int | str) -> bool:
        """Query whether edge (u, v) is active (ON) at current step t.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            True if edge exists in underlying graph and is ON; False otherwise.

        Raises:
            KeyError: If edge (u, v) does not exist in the underlying static graph.
        """
        key = frozenset([u, v])
        if key not in self._edge_states:
            raise KeyError(f"Edge ({u}, {v}) does not exist in underlying static graph")
        return self._edge_states[key] == EdgeState.ON

    def active_edges(self) -> list[frozenset[int | str]]:
        """Return list of endpoint frozensets for all currently ON edges."""
        return [k for k, state in self._edge_states.items() if state == EdgeState.ON]

    def inactive_edges(self) -> list[frozenset[int | str]]:
        """Return list of endpoint frozensets for all currently OFF edges."""
        return [k for k, state in self._edge_states.items() if state == EdgeState.OFF]

    def active_edge_count(self) -> int:
        """Total count of active (ON) edges at step t."""
        return sum(1 for state in self._edge_states.values() if state == EdgeState.ON)

    def inactive_edge_count(self) -> int:
        """Total count of inactive (OFF) edges at step t."""
        return sum(1 for state in self._edge_states.values() if state == EdgeState.OFF)

    def active_neighbors(self, node_id: int | str) -> list[int | str]:
        """Retrieve neighbors of node_id connected via currently active (ON) edges.

        Args:
            node_id: Identifier of the vertex in V.

        Returns:
            List of neighbor vertex identifiers with operational incident edges.

        Raises:
            KeyError: If node_id does not exist in the graph.
        """
        all_neighbors = self._static_graph.neighbors(node_id)
        result: list[int | str] = []
        for nbr in all_neighbors:
            key = frozenset([node_id, nbr])
            if self._edge_states[key] == EdgeState.ON:
                result.append(nbr)
        return result

    def advance(self) -> TransitionStatistics:
        """Advance simulation time from t to t+1 via synchronous edge state transitions.

        For each edge independently:
        - If state == ON:  transitions to OFF with probability p_off
        - If state == OFF: transitions to ON with probability p_on

        Returns:
            TransitionStatistics recording the transition event metrics.
        """
        next_states: dict[frozenset[int | str], EdgeState] = {}
        on_to_off = 0
        off_to_on = 0
        on_to_on = 0
        off_to_off = 0

        # Special fast path for STATIC regime (or p_on == p_off == 0)
        is_static = (self._regime == DynamicRegime.STATIC) or (self._p_on == 0.0 and self._p_off == 0.0)

        if is_static:
            for k in self._edge_keys:
                cur = self._edge_states[k]
                next_states[k] = cur
                if cur == EdgeState.ON:
                    on_to_on += 1
                else:
                    off_to_off += 1
        else:
            # Synchronous sampling: sample uniform random numbers for all edges
            num_edges = len(self._edge_keys)
            uniform_samples = self._rng.random(size=num_edges)

            for idx, k in enumerate(self._edge_keys):
                cur = self._edge_states[k]
                u = uniform_samples[idx]

                if cur == EdgeState.ON:
                    if u < self._p_off:
                        next_states[k] = EdgeState.OFF
                        on_to_off += 1
                    else:
                        next_states[k] = EdgeState.ON
                        on_to_on += 1
                else:  # cur == EdgeState.OFF
                    if u < self._p_on:
                        next_states[k] = EdgeState.ON
                        off_to_on += 1
                    else:
                        next_states[k] = EdgeState.OFF
                        off_to_off += 1

        self._edge_states = next_states
        self._time += 1

        active_cnt = on_to_on + off_to_on
        inactive_cnt = off_to_off + on_to_off

        stats = TransitionStatistics(
            time=self._time,
            on_to_off=on_to_off,
            off_to_on=off_to_on,
            on_to_on=on_to_on,
            off_to_off=off_to_off,
            active_edge_count=active_cnt,
            inactive_edge_count=inactive_cnt,
        )
        self._history.append(stats)
        return stats

    def snapshot(self) -> DynamicGraphSnapshot:
        """Create an immutable snapshot of the dynamic state at step t."""
        return DynamicGraphSnapshot(
            time=self._time,
            edge_states=dict(self._edge_states),
            active_edge_count=self.active_edge_count(),
            inactive_edge_count=self.inactive_edge_count(),
            regime=self._regime,
        )

    def transition_history(self) -> list[TransitionStatistics]:
        """Return list of transition statistics recorded up to step t."""
        return list(self._history)
