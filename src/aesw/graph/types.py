"""
Graph Types and Enumerations
============================
Defines strongly typed graph families and topology categories.
"""

from enum import Enum


class GraphType(str, Enum):
    """Supported graph families for topological generation.

    - ERDOS_RENYI: Erdős–Rényi random graph G(n, p).
    - BARABASI_ALBERT: Barabási–Albert scale-free preferential attachment network BA(n, m).
    - WATTS_STROGATZ: Watts–Strogatz small-world network WS(n, k, p).
    - GRID_OBSTACLE: 2D spatial grid with optional obstacle cells and orthogonal connectivity.
    - RANDOM_GEOMETRIC: Random Geometric Graph in unit square with Euclidean connection radius r.
    - AS_733: Real-world Autonomous Systems temporal network snapshots (temporal network validation).
    """
    ERDOS_RENYI = "er"
    BARABASI_ALBERT = "ba"
    WATTS_STROGATZ = "watts_strogatz"
    GRID_OBSTACLE = "grid"
    RANDOM_GEOMETRIC = "random_geometric"
    AS_733 = "as733"
