"""
Formal Data Models for Dynamic Graph Search
===========================================
Defines immutable representations of nodes, edges, delay specifications, targets, and walkers.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
from aesw.environment.types import EdgeState, TargetMode


@dataclass(frozen=True)
class Node:
    """Formal representation of a graph vertex v ∈ V.

    Attributes:
        node_id: Stable identifier for the node (integer or string).
        metadata: Optional topology/domain metadata (e.g., spatial coordinates, cluster label).
    """
    node_id: int | str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_id is None or (isinstance(self.node_id, str) and not self.node_id.strip()):
            raise ValueError("node_id must not be None or empty string")


@dataclass(frozen=True)
class Edge:
    """Formal representation of an edge e = (u, v) in the dynamic graph G_t = (V, E_t).

    Supports undirected graphs where (u, v) == (v, u) by default, while retaining
    source/destination orientation if directed extensions are modeled.

    Attributes:
        source: Identifier of the source node u.
        target: Identifier of the target node v.
        state: Current operational state (EdgeState.ON or EdgeState.OFF).
        is_directed: Whether the edge is strictly directed (False for undirected).
        metadata: Optional edge attributes (e.g., base weight, capacity).
    """
    source: int | str
    target: int | str
    state: EdgeState = EdgeState.ON
    is_directed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source is None or self.target is None:
            raise ValueError("Edge endpoints source and target must not be None")
        if self.source == self.target:
            raise ValueError(f"Self-loops are not permitted in base model: ({self.source}, {self.target})")
        if not isinstance(self.state, EdgeState):
            raise TypeError(f"state must be an EdgeState instance, got {type(self.state).__name__}")

    @property
    def endpoints(self) -> frozenset[int | str]:
        """Unordered pair of endpoints for undirected comparison."""
        return frozenset([self.source, self.target])


@dataclass(frozen=True)
class DelaySpecification:
    """Formal specification for traversal delays along active edges.

    Movement along an edge takes random discrete or continuous duration d >= 1 step.
    The searcher does NOT know the true delay beforehand.

    Attributes:
        distribution_type: Name of the distribution (e.g., 'constant', 'uniform', 'geometric').
        min_delay: Minimum traversal duration (must be >= 1).
        max_delay: Maximum traversal duration (must be >= min_delay).
        mean_delay: Expected traversal duration.
    """
    distribution_type: str = "constant"
    min_delay: int = 1
    max_delay: int = 1
    mean_delay: float = 1.0

    def __post_init__(self) -> None:
        if self.min_delay < 1:
            raise ValueError(f"min_delay must be at least 1 step, got {self.min_delay}")
        if self.max_delay < self.min_delay:
            raise ValueError(f"max_delay ({self.max_delay}) cannot be less than min_delay ({self.min_delay})")
        if self.mean_delay < self.min_delay or self.mean_delay > self.max_delay:
            raise ValueError(f"mean_delay ({self.mean_delay}) must be within [{self.min_delay}, {self.max_delay}]")


@dataclass(frozen=True)
class TargetState:
    """Formal ground-truth state representation of the target x_t ∈ V.

    INVARIANT: This ground-truth object must NEVER be directly exposed to walkers.

    Attributes:
        current_node: Current vertex occupied by the target x_t.
        mode: Locomotion mode (STATIC or MOVING).
        p_move: Transition probability of moving to an adjacent active node at time t.
        previous_node: Vertex occupied at step t-1 (None if at t=0).
        time: Current simulation timestamp t.
        move_attempted: Whether a movement was attempted at the most recent step.
        move_succeeded: Whether the movement attempt succeeded.
    """
    current_node: int | str
    mode: TargetMode = TargetMode.STATIC
    p_move: float = 0.0
    previous_node: Optional[int | str] = None
    time: int = 0
    move_attempted: bool = False
    move_succeeded: bool = False

    def __post_init__(self) -> None:
        if self.current_node is None:
            raise ValueError("Target current_node must not be None")
        if not (0.0 <= self.p_move <= 1.0):
            raise ValueError(f"p_move must be in [0, 1], got {self.p_move}")
        if self.time < 0:
            raise ValueError(f"time must be non-negative, got {self.time}")


@dataclass(frozen=True)
class WalkerState:
    """Generic formal state representation of a search agent / walker.

    Contains identity, physical position, and tracking counters.
    Does NOT contain algorithm-specific decision heuristics, scoring functions,
    or evidence caches.

    Attributes:
        walker_id: Stable identifier for the walker (e.g., 0, 1, ...).
        current_node: Identifier of the node currently occupied by the walker.
        step_count: Number of physical moves executed by this walker.
        message_count: Number of messages emitted/exchanged by this walker.
        busy_until_time: Simulation timestamp until which the walker is occupied traversing a delayed edge.
    """
    walker_id: int | str
    current_node: int | str
    step_count: int = 0
    message_count: int = 0
    busy_until_time: int = 0

    def __post_init__(self) -> None:
        if self.walker_id is None:
            raise ValueError("walker_id must not be None")
        if self.current_node is None:
            raise ValueError("current_node must not be None")
        if self.step_count < 0:
            raise ValueError(f"step_count must be non-negative, got {self.step_count}")
        if self.message_count < 0:
            raise ValueError(f"message_count must be non-negative, got {self.message_count}")
        if self.busy_until_time < 0:
            raise ValueError(f"busy_until_time must be non-negative, got {self.busy_until_time}")
