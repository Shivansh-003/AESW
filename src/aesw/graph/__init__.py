"""
Graph Module
============
Research-grade graph generation subsystem for Adaptive Graph Search (AESW).

Supported Graph Topologies:
- Erdős–Rényi G(n, p)
- Barabási–Albert BA(n, m)
- Watts–Strogatz WS(n, k, p)
- 2D Grid with Obstacles
- Random Geometric Graphs (RGG)

Extension Point:
- AS-733 real-world temporal network snapshots (via TemporalGraphSource)
"""

from aesw.graph.types import GraphType
from aesw.graph.metadata import GraphMetadata, TopologyStatistics
from aesw.graph.base import GraphInstance
from aesw.graph.validation import (
    validate_erdos_renyi_params,
    validate_barabasi_albert_params,
    validate_watts_strogatz_params,
    validate_grid_params,
    validate_random_geometric_params,
    validate_graph_invariants,
)
from aesw.graph.generators import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_grid_with_obstacles,
    generate_random_geometric,
)
from aesw.graph.factory import generate_graph, generate_graph_by_family
from aesw.graph.adapters import BaseGraphGenerator, TemporalGraphSource

__all__ = [
    # Types & Metadata
    "GraphType",
    "GraphMetadata",
    "TopologyStatistics",
    # Abstraction
    "GraphInstance",
    # Generators
    "generate_erdos_renyi",
    "generate_barabasi_albert",
    "generate_watts_strogatz",
    "generate_grid_with_obstacles",
    "generate_random_geometric",
    # Factory
    "generate_graph",
    "generate_graph_by_family",
    # Interfaces
    "BaseGraphGenerator",
    "TemporalGraphSource",
    # Validation
    "validate_erdos_renyi_params",
    "validate_barabasi_albert_params",
    "validate_watts_strogatz_params",
    "validate_grid_params",
    "validate_random_geometric_params",
    "validate_graph_invariants",
]
