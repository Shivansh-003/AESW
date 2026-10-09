"""
Evaluation Module
=================
Formal evaluation metrics contracts, experimental harness, and reproducible benchmark interfaces.
"""

from aesw.evaluation.metrics import (
    RunMetrics,
    AggregatedMetrics,
    compute_aggregated_metrics,
)
from aesw.evaluation.experiment import (
    ExperimentInstance,
    ExperimentResult,
    BenchmarkResult,
    generate_experiment,
    run_experiment,
    run_benchmark,
)

__all__ = [
    "RunMetrics",
    "AggregatedMetrics",
    "compute_aggregated_metrics",
    "ExperimentInstance",
    "ExperimentResult",
    "BenchmarkResult",
    "generate_experiment",
    "run_experiment",
    "run_benchmark",
]
