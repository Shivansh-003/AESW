"""
Core Graph Instance Abstraction
================================
Provides the project-level GraphInstance abstraction wrapping NetworkX internally.
Decouples research algorithms and simulations from NetworkX implementation details.
"""

from typing import Any, Mapping
import networkx as nx

from aesw.graph.types import GraphType
from aesw.graph.metadata import GraphMetadata, TopologyStatistics
from aesw.environment.models import Node, Edge
from aesw.environment.types import EdgeState


class GraphInstance:
    """Project-level immutable/encapsulated graph abstraction for static topology G = (V, E).

    Attributes:
        graph_type: The family/type of the graph (ER, BA, WS, Grid, RGG, etc.).
        metadata: Generation metadata, parameter provenance, and random seed.
    """

    def __init__(
        self,
        nx_graph: nx.Graph,
        graph_type: GraphType,
        metadata: GraphMetadata,
    ) -> None:
        self._nx_graph: nx.Graph = nx_graph.copy()
        self.graph_type: GraphType = graph_type
        self.metadata: GraphMetadata = metadata

        # Cache nodes and edges as immutable models
        node_dict: dict[int | str, Node] = {}
        for node_id, data in self._nx_graph.nodes(data=True):
            node_dict[node_id] = Node(node_id=node_id, metadata=dict(data))
        self._nodes: dict[int | str, Node] = node_dict

        edges_list: list[Edge] = []
        for u, v, data in self._nx_graph.edges(data=True):
            edges_list.append(
                Edge(
                    source=u,
                    target=v,
                    state=EdgeState.ON,
                    is_directed=self._nx_graph.is_directed(),
                    metadata=dict(data),
                )
            )
        self._edges: tuple[Edge, ...] = tuple(edges_list)

    @property
    def node_count(self) -> int:
        """Total number of vertices |V|."""
        return self._nx_graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Total number of edges |E|."""
        return self._nx_graph.number_of_edges()

    @property
    def nodes(self) -> Mapping[int | str, Node]:
        """Mapping from node identifier to Node object."""
        return self._nodes

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Tuple of all Edge objects in the topology."""
        return self._edges

    @property
    def node_ids(self) -> list[int | str]:
        """List of node identifiers."""
        return list(self._nodes.keys())

    def neighbors(self, node_id: int | str) -> list[int | str]:
        """Retrieve list of adjacent neighbor node identifiers."""
        if node_id not in self._nx_graph:
            raise KeyError(f"Node '{node_id}' not found in graph")
        return list(self._nx_graph.neighbors(node_id))

    def degree(self, node_id: int | str) -> int:
        """Retrieve the degree of a specific node."""
        if node_id not in self._nx_graph:
            raise KeyError(f"Node '{node_id}' not found in graph")
        return int(self._nx_graph.degree[node_id])

    def has_edge(self, u: int | str, v: int | str) -> bool:
        """Check if an edge exists between nodes u and v."""
        return self._nx_graph.has_edge(u, v)

    def is_connected(self) -> bool:
        """Check if the graph is connected (single connected component)."""
        if self.node_count == 0:
            return True
        return nx.is_connected(self._nx_graph)

    def number_of_connected_components(self) -> int:
        """Count the number of connected components."""
        if self.node_count == 0:
            return 0
        return nx.number_connected_components(self._nx_graph)

    def compute_statistics(self) -> TopologyStatistics:
        """Calculate and return lightweight topology statistics."""
        n = self.node_count
        m = self.edge_count
        if n == 0:
            return TopologyStatistics(
                node_count=0,
                edge_count=0,
                average_degree=0.0,
                min_degree=0,
                max_degree=0,
                number_of_connected_components=0,
                is_connected=True,
                density=0.0,
            )

        degrees = [deg for _, deg in self._nx_graph.degree()]
        avg_deg = sum(degrees) / n if degrees else 0.0
        min_deg = min(degrees) if degrees else 0
        max_deg = max(degrees) if degrees else 0
        density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
        num_cc = nx.number_connected_components(self._nx_graph)
        conn = num_cc == 1

        return TopologyStatistics(
            node_count=n,
            edge_count=m,
            average_degree=float(avg_deg),
            min_degree=int(min_deg),
            max_degree=int(max_deg),
            number_of_connected_components=num_cc,
            is_connected=conn,
            density=float(density),
        )

    def to_networkx(self) -> nx.Graph:
        """Return a copy of the underlying NetworkX graph (for analysis/testing)."""
        return self._nx_graph.copy()
