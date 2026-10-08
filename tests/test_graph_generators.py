"""
Comprehensive Test Suite for Graph Generation Subsystem.
Verifies:
- Erdős–Rényi G(n, p)
- Barabási–Albert BA(n, m)
- Watts–Strogatz WS(n, k, p)
- 2D Spatial Grid with Obstacles
- Random Geometric Graphs (RGG)
- Invariant testing, validation rules, statistics, and deterministic reproducibility.
"""

import math
import pytest
import networkx as nx

from aesw.graph.types import GraphType
from aesw.graph.base import GraphInstance
from aesw.graph.generators import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_grid_with_obstacles,
    generate_random_geometric,
)
from aesw.graph.factory import generate_graph, generate_graph_by_family
from aesw.graph.adapters import TemporalGraphSource
from aesw.environment.problem import GraphSpecification


# ==========================================
# A. Erdős–Rényi Tests
# ==========================================

def test_erdos_renyi_valid_generation():
    """Verify standard ER graph construction."""
    g = generate_erdos_renyi(n=50, p=0.1, seed=42)
    assert g.graph_type == GraphType.ERDOS_RENYI
    assert g.node_count == 50
    assert g.metadata.seed == 42
    assert g.metadata.generation_parameters["n"] == 50
    assert g.metadata.generation_parameters["p"] == 0.1
    assert not g.metadata.is_directed
    assert not g.metadata.has_self_loops


def test_erdos_renyi_invariants_p0_and_p1():
    """Verify ER boundary conditions: p=0 has 0 edges; p=1 is complete graph."""
    # p = 0
    g_empty = generate_erdos_renyi(n=20, p=0.0, seed=42)
    assert g_empty.node_count == 20
    assert g_empty.edge_count == 0

    # p = 1 (complete graph: n*(n-1)/2 edges)
    g_complete = generate_erdos_renyi(n=10, p=1.0, seed=42)
    assert g_complete.node_count == 10
    assert g_complete.edge_count == (10 * 9) // 2


def test_erdos_renyi_validation_errors():
    """Verify ER validation rejects invalid parameters."""
    with pytest.raises(ValueError, match="n must be strictly positive"):
        generate_erdos_renyi(n=0, p=0.1)

    with pytest.raises(ValueError, match="p must be in \\[0.0, 1.0\\]"):
        generate_erdos_renyi(n=10, p=-0.1)

    with pytest.raises(ValueError, match="p must be in \\[0.0, 1.0\\]"):
        generate_erdos_renyi(n=10, p=1.5)


def test_erdos_renyi_determinism():
    """Verify that identical seeds produce identical edge sets, while different seeds differ."""
    g1 = generate_erdos_renyi(n=30, p=0.2, seed=123)
    g2 = generate_erdos_renyi(n=30, p=0.2, seed=123)
    g3 = generate_erdos_renyi(n=30, p=0.2, seed=999)

    e1 = {e.endpoints for e in g1.edges}
    e2 = {e.endpoints for e in g2.edges}
    e3 = {e.endpoints for e in g3.edges}

    assert e1 == e2
    assert e1 != e3


# ==========================================
# B. Barabási–Albert Tests
# ==========================================

def test_barabasi_albert_valid_generation():
    """Verify standard BA scale-free network construction."""
    g = generate_barabasi_albert(n=40, m=3, seed=42)
    assert g.graph_type == GraphType.BARABASI_ALBERT
    assert g.node_count == 40
    # In BA(n, m), number of edges is exactly m*(n - m) + initial_edges
    assert g.edge_count > 0
    assert not g.metadata.is_directed
    assert not g.metadata.has_self_loops


def test_barabasi_albert_validation_errors():
    """Verify BA validation rejects invalid m attachment values."""
    with pytest.raises(ValueError, match="m must be at least 1"):
        generate_barabasi_albert(n=20, m=0)

    with pytest.raises(ValueError, match="m .* must be strictly less than node count n"):
        generate_barabasi_albert(n=20, m=20)

    with pytest.raises(ValueError, match="m .* must be strictly less than node count n"):
        generate_barabasi_albert(n=20, m=25)


def test_barabasi_albert_determinism():
    """Verify that identical seeds produce identical BA networks."""
    g1 = generate_barabasi_albert(n=30, m=2, seed=42)
    g2 = generate_barabasi_albert(n=30, m=2, seed=42)
    g3 = generate_barabasi_albert(n=30, m=2, seed=77)

    e1 = {e.endpoints for e in g1.edges}
    e2 = {e.endpoints for e in g2.edges}
    e3 = {e.endpoints for e in g3.edges}

    assert e1 == e2
    assert e1 != e3


# ==========================================
# C. Watts–Strogatz Tests
# ==========================================

def test_watts_strogatz_valid_generation():
    """Verify standard WS small-world network construction."""
    g = generate_watts_strogatz(n=50, k=4, p=0.1, seed=42)
    assert g.graph_type == GraphType.WATTS_STROGATZ
    assert g.node_count == 50
    # In WS(n, k, p), edge count is exactly n*k / 2
    assert g.edge_count == (50 * 4) // 2


def test_watts_strogatz_validation_errors():
    """Verify WS validation enforces k even, 2 <= k < n, and p in [0, 1]."""
    with pytest.raises(ValueError, match="k must be an even integer"):
        generate_watts_strogatz(n=30, k=3, p=0.1)

    with pytest.raises(ValueError, match="k must be at least 2"):
        generate_watts_strogatz(n=30, k=0, p=0.1)

    with pytest.raises(ValueError, match="k .* must be strictly less than node count n"):
        generate_watts_strogatz(n=10, k=10, p=0.1)

    with pytest.raises(ValueError, match="p must be in \\[0.0, 1.0\\]"):
        generate_watts_strogatz(n=20, k=4, p=1.2)


def test_watts_strogatz_determinism():
    """Verify WS determinism with identical seeds."""
    g1 = generate_watts_strogatz(n=25, k=4, p=0.3, seed=10)
    g2 = generate_watts_strogatz(n=25, k=4, p=0.3, seed=10)
    g3 = generate_watts_strogatz(n=25, k=4, p=0.3, seed=99)

    e1 = {e.endpoints for e in g1.edges}
    e2 = {e.endpoints for e in g2.edges}
    e3 = {e.endpoints for e in g3.edges}

    assert e1 == e2
    assert e1 != e3


# ==========================================
# D. 2D Spatial Grid with Obstacles Tests
# ==========================================

def test_grid_no_obstacles():
    """Verify grid construction without obstacles yields rows*cols nodes and 4-neighbor edges."""
    rows, cols = 5, 6
    g = generate_grid_with_obstacles(rows=rows, cols=cols, obstacle_ratio=0.0, seed=42)
    assert g.graph_type == GraphType.GRID_OBSTACLE
    assert g.node_count == rows * cols
    # For rows*cols grid with 4-connectivity:
    expected_edges = (rows - 1) * cols + rows * (cols - 1)
    assert g.edge_count == expected_edges
    assert g.metadata.spatial is True
    assert g.is_connected() is True


def test_grid_with_obstacle_mask():
    """Verify grid obstacle masking explicitly removes designated nodes."""
    rows, cols = 4, 4
    mask = [(1, 1), (2, 2)]
    g = generate_grid_with_obstacles(rows=rows, cols=cols, obstacle_mask=mask, seed=42)
    assert g.node_count == (16 - 2)
    assert "(1,1)" not in g.nodes
    assert "(2,2)" not in g.nodes
    assert "(0,0)" in g.nodes

    # Verify spatial coordinates preserved
    node_00 = g.nodes["(0,0)"]
    assert node_00.metadata["row"] == 0
    assert node_00.metadata["col"] == 0
    assert node_00.metadata["pos"] == (0.0, 0.0)


def test_grid_orthogonal_connectivity_invariant():
    """Verify that all grid edges connect strictly orthogonal neighbors (Manhattan dist == 1)."""
    g = generate_grid_with_obstacles(rows=6, cols=6, obstacle_ratio=0.15, seed=42)
    for edge in g.edges:
        u_node = g.nodes[edge.source]
        v_node = g.nodes[edge.target]
        r1, c1 = u_node.metadata["row"], u_node.metadata["col"]
        r2, c2 = v_node.metadata["row"], v_node.metadata["col"]
        manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
        assert manhattan_dist == 1, f"Non-orthogonal edge detected between ({r1},{c1}) and ({r2},{c2})"


def test_grid_determinism():
    """Verify that identical seeds produce identical obstacle sets and graphs."""
    g1 = generate_grid_with_obstacles(rows=10, cols=10, obstacle_ratio=0.2, seed=88)
    g2 = generate_grid_with_obstacles(rows=10, cols=10, obstacle_ratio=0.2, seed=88)
    g3 = generate_grid_with_obstacles(rows=10, cols=10, obstacle_ratio=0.2, seed=44)

    assert set(g1.node_ids) == set(g2.node_ids)
    assert {e.endpoints for e in g1.edges} == {e.endpoints for e in g2.edges}
    assert set(g1.node_ids) != set(g3.node_ids)


# ==========================================
# E. Random Geometric Graph Tests
# ==========================================

def test_random_geometric_valid_generation():
    """Verify RGG construction, node count, and spatial coordinates."""
    g = generate_random_geometric(n=30, radius=0.35, seed=42)
    assert g.graph_type == GraphType.RANDOM_GEOMETRIC
    assert g.node_count == 30
    assert g.metadata.spatial is True

    # Verify positions exist on all nodes
    for node in g.nodes.values():
        assert "pos" in node.metadata
        pos = node.metadata["pos"]
        assert len(pos) == 2
        assert 0.0 <= pos[0] <= 1.0
        assert 0.0 <= pos[1] <= 1.0


def test_random_geometric_euclidean_distance_invariant():
    """Verify invariant: every edge in RGG must have Euclidean distance <= radius + eps."""
    radius = 0.25
    g = generate_random_geometric(n=40, radius=radius, seed=42)
    for edge in g.edges:
        pos_u = g.nodes[edge.source].metadata["pos"]
        pos_v = g.nodes[edge.target].metadata["pos"]
        dist = math.dist(pos_u, pos_v)
        assert dist <= radius + 1e-6, f"Edge ({edge.source}, {edge.target}) exceeds radius: {dist} > {radius}"


def test_random_geometric_determinism():
    """Verify identical seeds produce identical node positions and edges."""
    g1 = generate_random_geometric(n=25, radius=0.3, seed=55)
    g2 = generate_random_geometric(n=25, radius=0.3, seed=55)
    g3 = generate_random_geometric(n=25, radius=0.3, seed=77)

    # Coordinates match
    pos1 = [g1.nodes[i].metadata["pos"] for i in range(25)]
    pos2 = [g2.nodes[i].metadata["pos"] for i in range(25)]
    assert pos1 == pos2

    e1 = {e.endpoints for e in g1.edges}
    e2 = {e.endpoints for e in g2.edges}
    e3 = {e.endpoints for e in g3.edges}
    assert e1 == e2
    assert e1 != e3


# ==========================================
# F. Factory & Dispatcher Tests
# ==========================================

def test_generate_graph_factory():
    """Verify generation through GraphSpecification and factory function."""
    spec_er = GraphSpecification(graph_type="er", num_nodes=20, seed=42, parameters={"p": 0.15})
    g_er = generate_graph(spec_er)
    assert g_er.node_count == 20
    assert g_er.graph_type == GraphType.ERDOS_RENYI

    spec_ba = GraphSpecification(graph_type="ba", num_nodes=25, seed=42, parameters={"m": 2})
    g_ba = generate_graph(spec_ba)
    assert g_ba.node_count == 25
    assert g_ba.graph_type == GraphType.BARABASI_ALBERT

    spec_ws = GraphSpecification(graph_type="watts_strogatz", num_nodes=20, seed=42, parameters={"k": 4, "p": 0.1})
    g_ws = generate_graph(spec_ws)
    assert g_ws.node_count == 20
    assert g_ws.graph_type == GraphType.WATTS_STROGATZ

    spec_grid = GraphSpecification(graph_type="grid", num_nodes=100, seed=42, parameters={"rows": 5, "cols": 5, "obstacle_ratio": 0.0})
    g_grid = generate_graph(spec_grid)
    assert g_grid.node_count == 25
    assert g_grid.graph_type == GraphType.GRID_OBSTACLE

    spec_rgg = GraphSpecification(graph_type="random_geometric", num_nodes=15, seed=42, parameters={"radius": 0.3})
    g_rgg = generate_graph(spec_rgg)
    assert g_rgg.node_count == 15
    assert g_rgg.graph_type == GraphType.RANDOM_GEOMETRIC


def test_generate_graph_by_family_convenience():
    """Verify convenience function generate_graph_by_family."""
    g = generate_graph_by_family(GraphType.ERDOS_RENYI, num_nodes=15, seed=42, parameters={"p": 0.2})
    assert g.node_count == 15
    assert g.graph_type == GraphType.ERDOS_RENYI


def test_as733_extension_point_error():
    """Verify AS-733 raises clear NotImplementedError pointing to temporal validation."""
    spec_as = GraphSpecification(graph_type="as733", num_nodes=100)
    with pytest.raises(NotImplementedError, match="temporal validation"):
        generate_graph(spec_as)


# ==========================================
# G. Topology Statistics & Invariants
# ==========================================

def test_topology_statistics():
    """Verify lightweight topology statistics calculation."""
    g = generate_erdos_renyi(n=30, p=0.2, seed=42)
    stats = g.compute_statistics()
    assert stats.node_count == 30
    assert stats.edge_count == g.edge_count
    assert stats.average_degree >= 0.0
    assert stats.min_degree <= stats.max_degree
    assert 0.0 <= stats.density <= 1.0
    assert stats.number_of_connected_components >= 1
    assert stats.is_connected == (stats.number_of_connected_components == 1)


def test_graph_instance_methods():
    """Verify GraphInstance neighborhood and degree queries."""
    g = generate_barabasi_albert(n=20, m=2, seed=42)
    node_0 = list(g.nodes.keys())[0]
    nbrs = g.neighbors(node_0)
    assert isinstance(nbrs, list)
    assert g.degree(node_0) == len(nbrs)

    with pytest.raises(KeyError):
        g.neighbors("non_existent_node")

    with pytest.raises(KeyError):
        g.degree("non_existent_node")
