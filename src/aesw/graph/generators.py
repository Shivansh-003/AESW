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
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    obstacle_ratio: float = 0.0,
    obstacle_mask: Optional[Sequence[tuple[int, int]]] = None,
    allow_diagonal: bool = False,
    seed: Optional[int] = 42,
    n: Optional[int] = None,
) -> GraphInstance:
    """Generate a 2D spatial grid network with obstacles and orthogonal connectivity.

    Nodes represent free cells with stable string/tuple IDs "(r, c)" and spatial coords (r, c).
    Obstacle cells are excluded from the active node set V.
    Edges connect strictly orthogonal neighbors: up, down, left, right (no diagonals by default).

    Args:
        rows: Optional number of grid rows > 0.
        cols: Optional number of grid columns > 0.
        obstacle_ratio: Probability of each cell being an obstacle (in [0.0, 1.0)).
        obstacle_mask: Optional explicit sequence of (r, c) obstacle coordinates.
        allow_diagonal: Whether 8-neighbor connectivity is allowed (default False, 4-neighbor only).
        seed: Explicit deterministic integer seed.
        n: Optional exact target node count |V|. When provided, grid dimensions are
           derived (if rows/cols not specified) and exactly (rows * cols - n) obstacle cells
           are placed, ensuring len(V) == n.

    Returns:
        GraphInstance with spatial coordinates and obstacle metadata.
    """
    actual_seed = 42 if seed is None else int(seed)
    rng = create_rng(actual_seed)

    if n is not None:
        if n <= 0:
            raise ValueError(f"Target node count n must be positive, got {n}")
        if not (0.0 <= obstacle_ratio < 1.0):
            raise ValueError(f"obstacle_ratio must be in [0.0, 1.0), got {obstacle_ratio}")

        if rows is None or cols is None:
            total_cells = math.ceil(n / (1.0 - obstacle_ratio)) if obstacle_ratio > 0.0 else n
            side = max(1, int(math.isqrt(total_cells)))
            actual_rows = side
            actual_cols = math.ceil(total_cells / actual_rows)
            while actual_rows * actual_cols < total_cells:
                actual_cols += 1
        else:
            actual_rows = rows
            actual_cols = cols
            if actual_rows * actual_cols < n:
                raise ValueError(
                    f"Grid dimensions ({actual_rows}x{actual_cols}={actual_rows * actual_cols}) "
                    f"cannot accommodate requested node count n={n}"
                )

        validate_grid_params(actual_rows, actual_cols, obstacle_ratio)

        num_obstacles = (actual_rows * actual_cols) - n
        obstacles: set[tuple[int, int]] = set()
        if obstacle_mask is not None:
            for r, c in obstacle_mask:
                if 0 <= r < actual_rows and 0 <= c < actual_cols:
                    obstacles.add((r, c))
        elif num_obstacles > 0:
            all_cells = [(r, c) for r in range(actual_rows) for c in range(actual_cols)]
            obs_indices = rng.choice(len(all_cells), size=num_obstacles, replace=False)
            obstacles = {all_cells[int(idx)] for idx in obs_indices}

        actual_rows_final = actual_rows
        actual_cols_final = actual_cols
    else:
        actual_rows_final = 10 if rows is None else rows
        actual_cols_final = 10 if cols is None else cols
        validate_grid_params(actual_rows_final, actual_cols_final, obstacle_ratio)

        # Determine obstacle set using Bernoulli trials
        obstacles = set()
        if obstacle_mask is not None:
            for r, c in obstacle_mask:
                if 0 <= r < actual_rows_final and 0 <= c < actual_cols_final:
                    obstacles.add((r, c))
        elif obstacle_ratio > 0.0:
            for r in range(actual_rows_final):
                for c in range(actual_cols_final):
                    if rng.random() < obstacle_ratio:
                        obstacles.add((r, c))

    nx_g = nx.Graph()

    # Add free cells as nodes with coordinates metadata
    for r in range(actual_rows_final):
        for c in range(actual_cols_final):
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

    for r in range(actual_rows_final):
        for c in range(actual_cols_final):
            if (r, c) in obstacles:
                continue
            u_id = f"({r},{c})"
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < actual_rows_final and 0 <= nc < actual_cols_final:
                    if (nr, nc) not in obstacles:
                        v_id = f"({nr},{nc})"
                        nx_g.add_edge(u_id, v_id)

    validate_graph_invariants(nx_g, expected_type=GraphType.GRID_OBSTACLE)

    metadata = GraphMetadata(
        graph_type=GraphType.GRID_OBSTACLE,
        seed=actual_seed,
        generation_parameters={
            "rows": actual_rows_final,
            "cols": actual_cols_final,
            "obstacle_ratio": obstacle_ratio,
            "allow_diagonal": allow_diagonal,
            **({"n": n} if n is not None else {}),
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
