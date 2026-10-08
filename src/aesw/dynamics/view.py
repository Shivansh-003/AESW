"""
Dynamic Graph View and Representation
=====================================
Provides a read-only active topology view G_t = (V, E_t) wrapping the underlying
GraphInstance and dynamic edge states without mutating the static graph.
"""

from typing import Mapping
from aesw.graph.base import GraphInstance
from aesw.environment.models import Node, Edge
from aesw.environment.types import EdgeState
from aesw.dynamics.engine import DynamicGraphState


class ActiveGraphView:
    """Read-only view of the active sub-network G_t = (V, E_t) at time step t.

    Exposes only currently active edges without modifying or copying the permanent
    underlying GraphInstance topology.

    Attributes:
        dynamic_state: The underlying DynamicGraphState instance.
    """

    def __init__(self, dynamic_state: DynamicGraphState) -> None:
        self._dynamic_state = dynamic_state

    @property
    def time(self) -> int:
        """Current discrete simulation step t."""
        return self._dynamic_state.time

    @property
    def node_count(self) -> int:
        """Total vertices |V| (all vertices remain in G_t)."""
        return self._dynamic_state.static_graph.node_count

    @property
    def active_edge_count(self) -> int:
        """Total number of active edges |E_t|."""
        return self._dynamic_state.active_edge_count()

    @property
    def nodes(self) -> Mapping[int | str, Node]:
        """All nodes V from the underlying static topology."""
        return self._dynamic_state.static_graph.nodes

    def active_edges(self) -> tuple[Edge, ...]:
        """Return tuple of active Edge instances at step t."""
        active_keys = set(self._dynamic_state.active_edges())
        return tuple(
            e for e in self._dynamic_state.static_graph.edges
            if e.endpoints in active_keys
        )

    def active_neighbors(self, node_id: int | str) -> list[int | str]:
        """Retrieve neighbors of node_id connected via currently active (ON) edges."""
        return self._dynamic_state.active_neighbors(node_id)

    def has_active_edge(self, u: int | str, v: int | str) -> bool:
        """Check if edge (u, v) is currently active (ON)."""
        try:
            return self._dynamic_state.is_edge_active(u, v)
        except KeyError:
            return False

    def is_edge_active(self, u: int | str, v: int | str) -> bool:
        """Check if edge (u, v) is currently active (ON). Alias for has_active_edge."""
        return self.has_active_edge(u, v)

    def active_degree(self, node_id: int | str) -> int:
        """Degree of node_id in active sub-network G_t."""
        return len(self.active_neighbors(node_id))
