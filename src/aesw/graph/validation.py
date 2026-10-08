"""
Graph Validation Utilities
==========================
Reusable invariant verifiers for graph instances and generation parameter constraints.
"""

from typing import Any, Mapping
import networkx as nx
from aesw.graph.types import GraphType


def validate_erdos_renyi_params(n: int, p: float) -> None:
    """Validate Erdős–Rényi G(n, p) generation parameters."""
    if n <= 0:
        raise ValueError(f"Number of nodes n must be strictly positive, got {n}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Edge probability p must be in [0.0, 1.0], got {p}")


def validate_barabasi_albert_params(n: int, m: int) -> None:
    """Validate Barabási–Albert BA(n, m) generation parameters."""
    if n <= 0:
        raise ValueError(f"Number of nodes n must be strictly positive, got {n}")
    if m < 1:
        raise ValueError(f"Attachment parameter m must be at least 1, got {m}")
    if m >= n:
        raise ValueError(f"Attachment parameter m ({m}) must be strictly less than node count n ({n})")


def validate_watts_strogatz_params(n: int, k: int, p: float) -> None:
    """Validate Watts–Strogatz WS(n, k, p) generation parameters."""
    if n <= 0:
        raise ValueError(f"Number of nodes n must be strictly positive, got {n}")
    if k < 2:
        raise ValueError(f"Initial degree k must be at least 2, got {k}")
    if k >= n:
        raise ValueError(f"Initial degree k ({k}) must be strictly less than node count n ({n})")
    if k % 2 != 0:
        raise ValueError(f"Initial degree k must be an even integer, got {k}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Rewiring probability p must be in [0.0, 1.0], got {p}")


def validate_grid_params(rows: int, cols: int, obstacle_ratio: float) -> None:
    """Validate 2D Grid with obstacles parameters."""
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Grid rows and cols must be strictly positive, got ({rows}, {cols})")
    if not (0.0 <= obstacle_ratio < 1.0):
        raise ValueError(f"obstacle_ratio must be in [0.0, 1.0), got {obstacle_ratio}")


def validate_random_geometric_params(n: int, radius: float, dim: int = 2) -> None:
    """Validate Random Geometric Graph parameters."""
    if n <= 0:
        raise ValueError(f"Number of nodes n must be strictly positive, got {n}")
    if radius < 0.0:
        raise ValueError(f"Connection radius must be non-negative, got {radius}")
    if dim <= 0:
        raise ValueError(f"Spatial dimension must be positive, got {dim}")


def validate_graph_invariants(
    g: nx.Graph,
    expected_type: GraphType,
    allow_self_loops: bool = False,
    require_connected: bool = False,
) -> None:
    """Verify structural invariants on generated NetworkX graph.

    Args:
        g: NetworkX graph instance.
        expected_type: Expected graph family.
        allow_self_loops: Whether self-loops are permitted (strictly False for research models).
        require_connected: If True, asserts the graph has exactly 1 connected component.

    Raises:
        ValueError: If any invariant is violated.
    """
    if g is None:
        raise ValueError("Graph instance must not be None")

    if not allow_self_loops:
        self_loops = list(nx.nodes_with_selfloops(g))
        if self_loops:
            raise ValueError(f"Graph contains {len(self_loops)} self-loops, which are not permitted: {self_loops}")

    if g.is_directed():
        raise ValueError("Expected undirected graph for baseline research topologies")

    if require_connected and g.number_of_nodes() > 0:
        if not nx.is_connected(g):
            num_cc = nx.number_connected_components(g)
            raise ValueError(f"Graph is required to be connected, but contains {num_cc} components")
