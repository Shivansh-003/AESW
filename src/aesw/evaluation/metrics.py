"""
Formal Metrics and Evaluation Contract
======================================
Defines data structures and contracts for per-run and aggregated evaluation metrics.
Does NOT implement metric computation or experiment runners.
"""

from dataclasses import dataclass
from aesw.environment.types import SearchTerminationStatus


@dataclass(frozen=True)
class RunMetrics:
    """Metrics produced by a single search execution run.

    Attributes:
        algorithm: Name of the evaluated search algorithm.
        seed: Random seed utilized for this execution.
        status: Final termination state (SUCCESS, BUDGET_EXHAUSTED, DISCONNECTED).
        success: Boolean flag indicating target acquisition within budget.
        search_time: Total simulation steps elapsed until termination.
        nodes_visited: Count of unique vertices inspected by searchers.
        node_revisits: Total number of repeated visits to previously inspected nodes.
        total_moves: Total physical edge traversals M across all walkers.
        total_messages: Total messages exchanged Q across all walkers.
        total_cost: Evaluated total cost C_total = M + c_m * Q.
    """
    algorithm: str
    seed: int
    status: SearchTerminationStatus
    success: bool
    search_time: int
    nodes_visited: int
    node_revisits: int
    total_moves: int
    total_messages: int
    total_cost: float

    def __post_init__(self) -> None:
        if not self.algorithm:
            raise ValueError("algorithm name must not be empty")
        if not isinstance(self.status, SearchTerminationStatus):
            raise TypeError(f"status must be a SearchTerminationStatus, got {type(self.status).__name__}")
        if self.search_time < 0:
            raise ValueError(f"search_time must be non-negative, got {self.search_time}")
        if self.nodes_visited < 0:
            raise ValueError(f"nodes_visited must be non-negative, got {self.nodes_visited}")
        if self.node_revisits < 0:
            raise ValueError(f"node_revisits must be non-negative, got {self.node_revisits}")
        if self.total_moves < 0:
            raise ValueError(f"total_moves must be non-negative, got {self.total_moves}")
        if self.total_messages < 0:
            raise ValueError(f"total_messages must be non-negative, got {self.total_messages}")
        if self.total_cost < 0.0:
            raise ValueError(f"total_cost must be non-negative, got {self.total_cost}")


@dataclass(frozen=True)
class AggregatedMetrics:
    """Statistical summary across multiple stochastic runs (N >= 50).

    Attributes:
        algorithm: Name of the evaluated algorithm.
        num_runs: Total number of independent runs executed.
        success_rate: Fraction of runs that succeeded (in [0, 1]).
        mean_search_time: Average search duration.
        std_search_time: Standard deviation of search duration.
        mean_total_cost: Average total cost C_total.
        std_total_cost: Standard deviation of total cost.
        ci_lower_cost: Lower bound of 95% confidence interval for total cost.
        ci_upper_cost: Upper bound of 95% confidence interval for total cost.
    """
    algorithm: str
    num_runs: int
    success_rate: float
    mean_search_time: float
    std_search_time: float
    mean_total_cost: float
    std_total_cost: float
    ci_lower_cost: float
    ci_upper_cost: float

    def __post_init__(self) -> None:
        if self.num_runs <= 0:
            raise ValueError(f"num_runs must be strictly positive, got {self.num_runs}")
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError(f"success_rate must be in [0.0, 1.0], got {self.success_rate}")
