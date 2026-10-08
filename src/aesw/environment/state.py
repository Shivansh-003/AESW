"""
Ground Truth Environment State
==============================
Represents the complete unobserved ground-truth state of the dynamic search system.
Strictly separated from walker observations (Controlled Observation Principle).
"""

from dataclasses import dataclass, field
from typing import Mapping
from aesw.environment.models import Node, Edge, TargetState, WalkerState
from aesw.environment.types import EdgeState


@dataclass(frozen=True)
class GroundTruthState:
    """Full unobserved ground-truth state of the environment at simulation time t.

    INVARIANT: Search algorithms and walkers must NEVER have direct access to this object.
    It is maintained solely by the simulation environment coordinator.

    Attributes:
        time: Current discrete simulation step t (t ∈ {0, 1, 2, ...}).
        nodes: Mapping of all vertex identifiers to Node objects V.
        edges: Tuple of all underlying edges E in the graph.
        active_edge_states: Mapping of edge pairs (frozenset[u, v]) to their active state (ON/OFF).
        target_state: True unobserved target position and locomotion state x_t.
        walker_states: Mapping of walker identifiers to their ground-truth state.
    """
    time: int
    nodes: Mapping[int | str, Node]
    edges: tuple[Edge, ...]
    active_edge_states: Mapping[frozenset[int | str], EdgeState]
    target_state: TargetState
    walker_states: Mapping[int | str, WalkerState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"Simulation time t must be non-negative, got {self.time}")
        if not self.nodes:
            raise ValueError("Environment must have at least one node")
        if self.target_state.current_node not in self.nodes:
            raise ValueError(
                f"Target location '{self.target_state.current_node}' does not exist in graph node set V"
            )

    @property
    def num_nodes(self) -> int:
        """Total number of nodes n = |V|."""
        return len(self.nodes)

    @property
    def active_edges(self) -> tuple[Edge, ...]:
        """Currently active edges E_t ⊆ E where edge state == ON."""
        return tuple(
            e for e in self.edges
            if self.active_edge_states.get(e.endpoints, e.state) == EdgeState.ON
        )
