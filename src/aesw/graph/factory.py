"""
Graph Generation Dispatcher API
===============================
Provides the primary factory entry points for generating static graph topologies
from GraphSpecification objects or family-specific parameters.
"""

from typing import Optional, Any, Mapping
from aesw.graph.types import GraphType
from aesw.graph.base import GraphInstance
from aesw.graph.generators import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_grid_with_obstacles,
    generate_random_geometric,
)
from aesw.environment.problem import GraphSpecification


def generate_graph(
    spec: GraphSpecification,
    seed: Optional[int] = None,
) -> GraphInstance:
    """Generate a static graph topology instance from a GraphSpecification.

    Args:
        spec: Validated GraphSpecification defining graph type and parameters.
        seed: Optional seed override; if None, uses spec.seed.

    Returns:
        GraphInstance encapsulating the generated topology and metadata.

    Raises:
        ValueError: If graph_type is unsupported or parameters are invalid.
        NotImplementedError: If AS-733 real-world trace loader is not yet available.
    """
    actual_seed = spec.seed if seed is None else int(seed)
    params = spec.parameters or {}
    raw_type = spec.graph_type.lower()

    if raw_type in ("er", "erdos_renyi"):
        p = float(params.get("p", 0.05))
        return generate_erdos_renyi(n=spec.num_nodes, p=p, seed=actual_seed)

    elif raw_type in ("ba", "barabasi_albert"):
        m = int(params.get("m", 2))
        return generate_barabasi_albert(n=spec.num_nodes, m=m, seed=actual_seed)

    elif raw_type in ("watts_strogatz", "ws"):
        k = int(params.get("k", 4))
        p = float(params.get("p", 0.1))
        return generate_watts_strogatz(n=spec.num_nodes, k=k, p=p, seed=actual_seed)

    elif raw_type in ("grid", "grid_obstacle"):
        rows = int(params.get("rows", 10))
        cols = int(params.get("cols", 10))
        obstacle_ratio = float(params.get("obstacle_ratio", 0.1))
        allow_diagonal = bool(params.get("allow_diagonal", False))
        obstacle_mask = params.get("obstacle_mask", None)
        return generate_grid_with_obstacles(
            rows=rows,
            cols=cols,
            obstacle_ratio=obstacle_ratio,
            obstacle_mask=obstacle_mask,
            allow_diagonal=allow_diagonal,
            seed=actual_seed,
        )

    elif raw_type in ("random_geometric", "rgg"):
        radius = float(params.get("radius", 0.2))
        dim = int(params.get("dim", 2))
        return generate_random_geometric(
            n=spec.num_nodes,
            radius=radius,
            dim=dim,
            seed=actual_seed,
        )

    elif raw_type in ("as733", "as_733"):
        raise NotImplementedError(
            "AS-733 real-world temporal network snapshot loader is planned for temporal validation. "
            "Please use synthetic generators (er, ba, watts_strogatz, grid, random_geometric)."
        )

    else:
        raise ValueError(
            f"Unsupported graph family: '{spec.graph_type}'. Supported types: "
            f"{[gt.value for gt in GraphType]}"
        )


def generate_graph_by_family(
    graph_type: GraphType | str,
    num_nodes: int = 100,
    seed: int = 42,
    parameters: Optional[Mapping[str, Any]] = None,
) -> GraphInstance:
    """Convenience factory function generating a graph by family name and parameters."""
    type_str = graph_type.value if isinstance(graph_type, GraphType) else str(graph_type)
    spec = GraphSpecification(
        graph_type=type_str,
        num_nodes=num_nodes,
        seed=seed,
        parameters=dict(parameters or {}),
    )
    return generate_graph(spec=spec, seed=seed)
