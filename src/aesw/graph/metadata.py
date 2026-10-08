"""
Graph Metadata and Topology Statistics
======================================
Immutable data structures for topology metrics and generation provenance.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping
from aesw.graph.types import GraphType


@dataclass(frozen=True)
class TopologyStatistics:
    """Lightweight structural statistics calculated over a static graph G = (V, E).

    Attributes:
        node_count: Total number of nodes |V|.
        edge_count: Total number of edges |E|.
        average_degree: Mean vertex degree.
        min_degree: Minimum vertex degree.
        max_degree: Maximum vertex degree.
        number_of_connected_components: Number of connected components.
        is_connected: Whether the graph forms a single connected component.
        density: Graph density 2|E| / (|V|(|V|-1)).
    """
    node_count: int
    edge_count: int
    average_degree: float
    min_degree: int
    max_degree: int
    number_of_connected_components: int
    is_connected: bool
    density: float


@dataclass(frozen=True)
class GraphMetadata:
    """Provenance and generation parameters for a GraphInstance.

    Attributes:
        graph_type: The GraphType family used to generate the graph.
        seed: The integer random seed used for generation.
        generation_parameters: Mapping of specific generator arguments (e.g. p, m, k, radius, dimensions).
        is_directed: Whether the graph is directed.
        has_self_loops: Whether self-loops are present (strictly False in base models).
        spatial: Flag indicating whether vertices retain spatial (x, y) coordinates.
        extra: Additional family-specific metadata (e.g., obstacle_ratio, obstacle_count).
    """
    graph_type: GraphType
    seed: int
    generation_parameters: Mapping[str, Any] = field(default_factory=dict)
    is_directed: bool = False
    has_self_loops: bool = False
    spatial: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)
