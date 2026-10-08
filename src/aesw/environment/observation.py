"""
Walker Observation Model
========================
Formal data models representing partial, local, noisy observations emitted to walkers.
Strictly implements the Controlled Observation Principle.
"""

from dataclasses import dataclass
from typing import Mapping
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
        outcome_category: Theoretical classification (POSITIVE, NEGATIVE, MISSED, FALSE_POSITIVE).
        timestamp: Simulation timestamp of the observation.
    """
    detected: bool
    signal_strength: float
    outcome_category: DetectionOutcome
    timestamp: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.signal_strength <= 1.0):
            raise ValueError(f"signal_strength must be in [0.0, 1.0], got {self.signal_strength}")
        if not isinstance(self.outcome_category, DetectionOutcome):
            raise TypeError(f"outcome_category must be a DetectionOutcome, got {type(self.outcome_category).__name__}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")


@dataclass(frozen=True)
class Observation:
    """Partial, local observation emitted to a single walker at simulation step t.

    Controlled Observation Principle Invariant:
    This object encapsulates strictly local, bounded information:
    - Does NOT contain the true target location x_t.
    - Does NOT contain global graph topology V or E.
    - Does NOT contain future edge transition states.
    - Does NOT contain true future traversal delays.

    Attributes:
        walker_id: Identifier of the walker receiving this observation.
        current_node: Identifier of the node currently occupied by the walker.
        time: Current simulation step t.
        checked_neighbors: Tuple of neighbor identifiers inspected within budget B.
        observed_edges: Mapping from neighbor_id to observed edge state information.
        target_signal: Noisy target detection sensor reading at current_node.
        neighbor_budget_used: Number of neighbor checks performed (must be <= B).
    """
    walker_id: int | str
    current_node: int | str
    time: int
    checked_neighbors: tuple[int | str, ...]
    observed_edges: Mapping[int | str, ObservedEdgeInfo]
    target_signal: TargetSignalObservation
    neighbor_budget_used: int

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
