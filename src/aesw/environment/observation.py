"""
Walker Observation Model
========================
Formal data models representing partial, local, noisy observations emitted to walkers.
Strictly implements the Controlled Observation Principle.
"""

from dataclasses import dataclass, field
from typing import Mapping, Optional
from aesw.environment.types import EdgeState, DetectionOutcome


@dataclass(frozen=True)
class ObservedEdgeInfo:
    """Locally observable information regarding an incident edge.

    Attributes:
        neighbor_id: The adjacent node identifier v.
        state: The observed operational state (ON/OFF).
        observed_at_time: Simulation timestamp when this observation occurred.
    """
    neighbor_id: int | str
    state: EdgeState
    observed_at_time: int

    def __post_init__(self) -> None:
        if self.neighbor_id is None:
            raise ValueError("neighbor_id must not be None")
        if not isinstance(self.state, EdgeState):
            raise TypeError(f"state must be an EdgeState instance, got {type(self.state).__name__}")
        if self.observed_at_time < 0:
            raise ValueError(f"observed_at_time must be non-negative, got {self.observed_at_time}")


@dataclass(frozen=True)
class TargetSignalObservation:
    """Noisy target sensor reading at the walker's current node.

    Subject to detection probability p_d (within radius s) and false alarm probability p_fa.
    Does NOT contain the true location of the target x_t.

    Attributes:
        detected: Boolean flag indicating whether the sensor triggered.
        signal_strength: Observed scalar strength (e.g., in [0, 1]).
        outcome_category: Theoretical classification (optional; strictly None when emitted
            to walkers by ObservationBuilder to prevent ground-truth leakage).
        timestamp: Simulation timestamp of the observation.
    """
    detected: bool
    signal_strength: float = 1.0
    outcome_category: Optional[DetectionOutcome] = None
    timestamp: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.signal_strength <= 1.0):
            raise ValueError(f"signal_strength must be in [0.0, 1.0], got {self.signal_strength}")
        if self.outcome_category is not None and not isinstance(self.outcome_category, DetectionOutcome):
            raise TypeError(f"outcome_category must be a DetectionOutcome or None, got {type(self.outcome_category).__name__}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")


@dataclass(frozen=True)
class LocalObservationHistory:
    """Immutable sequence of local observations accumulated by a single walker.

    Enforces snapshot semantics: subsequent mutations to the environment
    or future steps never alter past entries in the history.
    """
    entries: tuple["Observation", ...] = ()

    def add(self, observation: "Observation") -> "LocalObservationHistory":
        """Return a new history instance with the observation appended."""
        if not isinstance(observation, Observation):
            raise TypeError(f"Expected Observation instance, got {type(observation).__name__}")
        return LocalObservationHistory(entries=self.entries + (observation,))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> "Observation":
        return self.entries[index]

    def __iter__(self):
        return iter(self.entries)

    @property
    def latest(self) -> Optional["Observation"]:
        """Most recent observation snapshot, or None if history is empty."""
        return self.entries[-1] if self.entries else None

    @property
    def visited_nodes(self) -> tuple[int | str, ...]:
        """Sequence of nodes occupied across observation history."""
        return tuple(obs.current_node for obs in self.entries)

    def get_latest_edge_status(self, neighbor: int | str) -> Optional[EdgeState]:
        """Return most recent observed state of an edge to neighbor, or None if never observed."""
        for obs in reversed(self.entries):
            if neighbor in obs.observed_edges:
                return obs.observed_edges[neighbor].state
        return None

    def is_edge_known_active(self, neighbor: int | str) -> bool:
        """Whether the edge to neighbor was observed active in the most recent inspection."""
        status = self.get_latest_edge_status(neighbor)
        return status == EdgeState.ON

    def is_edge_known_inactive(self, neighbor: int | str) -> bool:
        """Whether the edge to neighbor was observed inactive in the most recent inspection."""
        status = self.get_latest_edge_status(neighbor)
        return status == EdgeState.OFF

    def is_edge_unknown(self, neighbor: int | str) -> bool:
        """Return True if neighbor has never been inspected in any historical observation."""
        return all(neighbor not in obs.observed_edges for obs in self.entries)


@dataclass(frozen=True)
class Observation:
    """Partial, local observation emitted to a single walker at simulation step t.

    Controlled Observation Principle Invariant:
    This object encapsulates strictly local, bounded information:
    - Does NOT contain the true target location x_t.
    - Does NOT contain global graph topology V or E.
    - Does NOT contain future edge transition states.
    - Does NOT contain true future traversal delays.
    - Does NOT contain ground-truth detection classification or distance.

    Attributes:
        walker_id: Identifier of the walker receiving this observation.
        current_node: Identifier of the node currently occupied by the walker.
        time: Current simulation step t.
        checked_neighbors: Tuple of neighbor identifiers inspected within budget B.
        observed_edges: Mapping from neighbor_id to observed edge state information.
        target_signal: Noisy target detection sensor reading at current_node.
        neighbor_budget_used: Number of neighbor checks performed (must be <= B).
        visible_neighbors: Tuple of neighbors actually visible/checked (bounded by B).
        observed_signals: Mapping from node identifier to observed binary signal.
        local_history: Optional immutable history of past observations.
    """
    walker_id: int | str
    current_node: int | str
    time: int
    checked_neighbors: tuple[int | str, ...]
    observed_edges: Mapping[int | str, ObservedEdgeInfo]
    target_signal: TargetSignalObservation
    neighbor_budget_used: int
    visible_neighbors: tuple[int | str, ...] = ()
    observed_signals: Mapping[int | str, bool] = field(default_factory=dict)
    local_history: Optional[LocalObservationHistory] = None

    def __post_init__(self) -> None:
        if self.walker_id is None:
            raise ValueError("walker_id must not be None")
        if self.current_node is None:
            raise ValueError("current_node must not be None")
        if self.time < 0:
            raise ValueError(f"time must be non-negative, got {self.time}")
        if self.neighbor_budget_used < 0:
            raise ValueError(f"neighbor_budget_used must be non-negative, got {self.neighbor_budget_used}")
        if len(self.checked_neighbors) != self.neighbor_budget_used:
            raise ValueError(
                f"Number of checked_neighbors ({len(self.checked_neighbors)}) must equal "
                f"neighbor_budget_used ({self.neighbor_budget_used})"
            )

        # Enforce tuple for checked_neighbors
        if not isinstance(self.checked_neighbors, tuple):
            object.__setattr__(self, "checked_neighbors", tuple(self.checked_neighbors))

        # Synchronize visible_neighbors to checked_neighbors if omitted
        if not self.visible_neighbors:
            object.__setattr__(self, "visible_neighbors", self.checked_neighbors)
        elif not isinstance(self.visible_neighbors, tuple):
            object.__setattr__(self, "visible_neighbors", tuple(self.visible_neighbors))

        # Synchronize observed_signals if omitted
        if not self.observed_signals:
            signals = {self.current_node: self.target_signal.detected} if self.target_signal else {}
            object.__setattr__(self, "observed_signals", signals)

    @property
    def detected(self) -> bool:
        """Convenience property for binary detection signal at current node."""
        return self.target_signal.detected

    def is_edge_known_active(self, neighbor: int | str) -> bool:
        """Whether the edge to neighbor was inspected and observed active (ON)."""
        edge = self.observed_edges.get(neighbor)
        return edge is not None and edge.state == EdgeState.ON

    def is_edge_known_inactive(self, neighbor: int | str) -> bool:
        """Whether the edge to neighbor was inspected and observed inactive (OFF)."""
        edge = self.observed_edges.get(neighbor)
        return edge is not None and edge.state == EdgeState.OFF

    def is_edge_unknown(self, neighbor: int | str) -> bool:
        """Whether the edge to neighbor was not inspected in this observation."""
        return neighbor not in self.observed_edges

    def get_edge_status(self, neighbor: int | str) -> Optional[EdgeState]:
        """Return observed EdgeState if inspected, otherwise None (unknown)."""
        edge = self.observed_edges.get(neighbor)
        return edge.state if edge is not None else None
