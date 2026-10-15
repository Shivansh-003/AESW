"""
Quantum Integration Package
===========================
Provides candidate registration, marking rules, quantum oracle operators,
and Grover search statevector simulation for hybrid classical-quantum search.
"""

from aesw.quantum.grover import GroverSearchEngine, compute_optimal_grover_iterations
from aesw.quantum.mapping import CandidateRegister
from aesw.quantum.marking import (
    MarkingRule,
    PredicateMarkingRule,
    TargetIdMarkingRule,
    ThresholdMarkingRule,
)
from aesw.quantum.oracle import QuantumOracle
from aesw.quantum.types import (
    NodeId,
    QuantumCandidate,
    QuantumSearchConfig,
    QuantumSearchResult,
    QuantumSearchStatus,
)

__all__ = [
    "CandidateRegister",
    "GroverSearchEngine",
    "MarkingRule",
    "NodeId",
    "PredicateMarkingRule",
    "QuantumCandidate",
    "QuantumOracle",
    "QuantumSearchConfig",
    "QuantumSearchResult",
    "QuantumSearchStatus",
    "TargetIdMarkingRule",
    "ThresholdMarkingRule",
    "compute_optimal_grover_iterations",
]
