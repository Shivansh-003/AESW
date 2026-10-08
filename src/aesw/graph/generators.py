"""
Concrete Graph Generators
=========================
Deterministic generators for ER, BA, WS, 2D Grid with obstacles, and RGG topologies.
"""

from typing import Optional, Sequence
import math
import numpy as np
import networkx as nx

from aesw.graph.types import GraphType
from aesw.graph.base import GraphInstance
from aesw.graph.metadata import GraphMetadata
from aesw.graph.validation import (
    validate_erdos_renyi_params,
    validate_barabasi_albert_params,
    validate_watts_strogatz_params,
    validate_grid_params,
    validate_random_geometric_params,
    validate_graph_invariants,
)
from aesw.utils.reproducibility import create_rng


def generate_erdos_renyi(
    n: int,
    p: float,
    seed: Optional[int] = 42,
) -> GraphInstance:
    """Generate an Erdős–Rényi G(n, p) random graph.

    Args:
        n: Number of nodes |V| > 0.
        p: Independent edge probability p ∈ [0, 1].
        seed: Explicit deterministic integer seed.

    Returns:
        GraphInstance encapsulating the generated topology and metadata.
    """
    validate_erdos_renyi_params(n, p)
    actual_seed = 42 if seed is None else int(seed)
    rng = create_rng(actual_seed)

    # Use NetworkX G(n, p) with isolated numpy RNG
    nx_g = nx.erdos_renyi_graph(n=n, p=p, seed=rng)

    validate_graph_invariants(nx_g, expected_type=GraphType.ERDOS_RENYI)

    metadata = GraphMetadata(
        graph_type=GraphType.ERDOS_RENYI,
        seed=actual_seed,
        generation_parameters={"n": n, "p": p},
        is_directed=False,
        has_self_loops=False,
        spatial=False,
    )
    return GraphInstance(nx_graph=nx_g, graph_type=GraphType.ERDOS_RENYI, metadata=metadata)


def generate_barabasi_albert(
    n: int,
    m: int,
    seed: Optional[int] = 42,
) -> GraphInstance:
    """Generate a Barabási–Albert BA(n, m) scale-free network using preferential attachment.

    Args:
        n: Number of nodes |V| > 0.
        m: Number of edges to attach from a new node to existing nodes (1 <= m < n).
        seed: Explicit deterministic integer seed.

    Returns:
        GraphInstance encapsulating the generated topology and metadata.
    """
    validate_barabasi_albert_params(n, m)
    actual_seed = 42 if seed is None else int(seed)
    rng = create_rng(actual_seed)

    nx_g = nx.barabasi_albert_graph(n=n, m=m, seed=rng)

    validate_graph_invariants(nx_g, expected_type=GraphType.BARABASI_ALBERT)

    metadata = GraphMetadata(
        graph_type=GraphType.BARABASI_ALBERT,
        seed=actual_seed,
        generation_parameters={"n": n, "m": m},
        is_directed=False,
        has_self_loops=False,
        spatial=False,
    )
    return GraphInstance(nx_graph=nx_g, graph_type=GraphType.BARABASI_ALBERT, metadata=metadata)


def generate_watts_strogatz(
    n: int,
    k: int,
    p: float,
    seed: Optional[int] = 42,
) -> GraphInstance:
    """Generate a Watts–Strogatz WS(n, k, p) small-world network.

    Args:
        n: Number of nodes |V| > 0.
        k: Initial degree of each node in the ring lattice (k must be even, 2 <= k < n).
        p: Edge rewiring probability p ∈ [0, 1].
        seed: Explicit deterministic integer seed.

    Returns:
        GraphInstance encapsulating the generated topology and metadata.
    """
    validate_watts_strogatz_params(n, k, p)
    actual_seed = 42 if seed is None else int(seed)
    rng = create_rng(actual_seed)

    nx_g = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=rng)

    validate_graph_invariants(nx_g, expected_type=GraphType.WATTS_STROGATZ)

    metadata = GraphMetadata(
        graph_type=GraphType.WATTS_STROGATZ,
        seed=actual_seed,
        generation_parameters={"n": n, "k": k, "p": p},
        is_directed=False,
        has_self_loops=False,
        spatial=False,
    )
    return GraphInstance(nx_graph=nx_g, graph_type=GraphType.WATTS_STROGATZ, metadata=metadata)


def generate_grid_with_obstacles(
    rows: int,
    cols: int,
    obstacle_ratio: float = 0.0,
    obstacle_mask: Optional[Sequence[tuple[int, int]]] = None,
    allow_diagonal: bool = False,
    seed: Optional[int] = 42,
) -> GraphInstance:
    """Generate a 2D spatial grid network with obstacles and orthogonal connectivity.

    Nodes represent free cells with stable string/tuple IDs "(r, c)" and spatial coords (r, c).
    Obstacle cells are excluded from the active node set V.
    Edges connect strictly orthogonal neighbors: up, down, left, right (no diagonals by default).

    Args:
        rows: Number of grid rows > 0.
        cols: Number of grid columns > 0.
        obstacle_ratio: Probability of each cell being an obstacle (in [0.0, 1.0)).
        obstacle_mask: Optional explicit sequence of (r, c) obstacle coordinates.
        allow_diagonal: Whether 8-neighbor connectivity is allowed (default False, 4-neighbor only).
        seed: Explicit deterministic integer seed.

    Returns:
        GraphInstance with spatial coordinates and obstacle metadata.
    """
    validate_grid_params(rows, cols, obstacle_ratio)
    actual_seed = 42 if seed is None else int(seed)
    rng = create_rng(actual_seed)

    # Determine obstacle set
    obstacles: set[tuple[int, int]] = set()
    if obstacle_mask is not None:
        for r, c in obstacle_mask:
            if 0 <= r < rows and 0 <= c < cols:
                obstacles.add((r, c))
    elif obstacle_ratio > 0.0:
        for r in range(rows):
            for c in range(cols):
                if rng.random() < obstacle_ratio:
                    obstacles.add((r, c))

    nx_g = nx.Graph()

    # Add free cells as nodes with coordinates metadata
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in obstacles:
                node_id = f"({r},{c})"
                nx_g.add_node(
                    node_id,
                    row=r,
                    col=c,
                    pos=(float(c), float(r)),  # (x, y) standard convention
                )

    # Add orthogonal edges (4-neighbor)
    orthogonal_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    offsets = orthogonal_offsets + (diagonal_offsets if allow_diagonal else [])

    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                continue
            u_id = f"({r},{c})"
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in obstacles:
                        v_id = f"({nr},{nc})"
                        nx_g.add_edge(u_id, v_id)

    validate_graph_invariants(nx_g, expected_type=GraphType.GRID_OBSTACLE)

    metadata = GraphMetadata(
        graph_type=GraphType.GRID_OBSTACLE,
        seed=actual_seed,
        generation_parameters={
            "rows": rows,
            "cols": cols,
            "obstacle_ratio": obstacle_ratio,
            "allow_diagonal": allow_diagonal,
        },
        is_directed=False,
        has_self_loops=False,
        spatial=True,
        extra={
            "num_obstacles": len(obstacles),
            "obstacle_cells": sorted(list(obstacles)),
        },
    )
    return GraphInstance(nx_graph=nx_g, graph_type=GraphType.GRID_OBSTACLE, metadata=metadata)


def generate_random_geometric(
    n: int,
    radius: float,
    dim: int = 2,
    seed: Optional[int] = 42,
) -> GraphInstance:
    """Generate a Random Geometric Graph (RGG) in unit [0, 1]^dim space.

    Nodes are placed uniformly at random. Edges connect pairs within Euclidean distance <= radius.
    Coordinates are preserved on nodes under attribute 'pos'.

    Args:
        n: Number of nodes > 0.
        radius: Euclidean connection radius >= 0.0.
        dim: Spatial dimension (default 2).
        seed: Explicit deterministic integer seed.

    Returns:
        GraphInstance with preserved spatial coordinates.
    """
    validate_random_geometric_params(n, radius, dim)
    actual_seed = 42 if seed is None else int(seed)
    rng = create_rng(actual_seed)

    nx_g = nx.random_geometric_graph(n=n, radius=radius, dim=dim, seed=rng)

    validate_graph_invariants(nx_g, expected_type=GraphType.RANDOM_GEOMETRIC)

    metadata = GraphMetadata(
        graph_type=GraphType.RANDOM_GEOMETRIC,
        seed=actual_seed,
        generation_parameters={"n": n, "radius": radius, "dim": dim},
        is_directed=False,
        has_self_loops=False,
        spatial=True,
    )
    return GraphInstance(nx_graph=nx_g, graph_type=GraphType.RANDOM_GEOMETRIC, metadata=metadata)
