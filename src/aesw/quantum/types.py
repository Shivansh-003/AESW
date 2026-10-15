"""
Quantum Search Types and Data Models
====================================
Defines strongly typed data contracts for quantum candidate representation,
search configurations, marking rules, status enumerations, and structured results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aesw.aesw.next_hop import CandidateScore
from aesw.memory.models import NodeEvidence, WeightedEvidence

NodeId = int | str


class QuantumSearchStatus(str, Enum):
    """Categorical outcome status for a quantum search invocation.

    - SUCCESS: A valid, marked candidate satisfying the eligibility predicate was measured.
    - NO_MARKED_CANDIDATES: No candidate in the provided set satisfied the marking rule.
    - EMPTY_CANDIDATES: The candidate register was empty; search could not proceed.
    - MEASUREMENT_NON_TARGET: A valid candidate index was measured, but it did not satisfy the target rule.
    - MEASUREMENT_PADDED_INDEX: A non-physical padded register index (>= K) was sampled upon measurement.
    - ERROR: An invalid configuration or computational error occurred.
    """
    SUCCESS = "SUCCESS"
    NO_MARKED_CANDIDATES = "NO_MARKED_CANDIDATES"
    EMPTY_CANDIDATES = "EMPTY_CANDIDATES"
    MEASUREMENT_NON_TARGET = "MEASUREMENT_NON_TARGET"
    MEASUREMENT_PADDED_INDEX = "MEASUREMENT_PADDED_INDEX"
    ERROR = "ERROR"

    @property
    def is_success(self) -> bool:
        """Whether status indicates a successful target discovery."""
        return self == QuantumSearchStatus.SUCCESS


@dataclass(frozen=True)
class QuantumCandidate:
    """Typed representation of a candidate vertex passed to the quantum module.

    Decouples raw graph structures from the quantum search register. Only carries
    the node identifier and relevant evidence score / metadata necessary for oracle marking.

    Attributes:
        node_id: Original graph vertex identifier (int or str).
        score: Primary numerical evidence score in [0.0, 1.0], if available.
        evidence: Optional provenance reference (NodeEvidence, WeightedEvidence, or CandidateScore).
        metadata: Optional auxiliary diagnostics or scoring breakdown.
    """
    node_id: NodeId
    score: float | None = None
    evidence: NodeEvidence | WeightedEvidence | CandidateScore | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_id is None:
            raise ValueError("node_id must not be None")
        if isinstance(self.node_id, str) and not self.node_id.strip():
            raise ValueError("node_id must not be an empty string")
        if self.score is not None and (math.isnan(self.score) or math.isinf(self.score)):
            raise ValueError(f"score must be a finite float, got {self.score}")

    @classmethod
    def from_node_id(cls, node_id: NodeId, **kwargs: Any) -> QuantumCandidate:
        """Create a candidate from a simple node identifier."""
        return cls(node_id=node_id, **kwargs)

    @classmethod
    def from_candidate_score(cls, candidate: CandidateScore) -> QuantumCandidate:
        """Construct a QuantumCandidate from an AESW CandidateScore breakdown."""
        return cls(
            node_id=candidate.node_id,
            score=float(candidate.positive_evidence),
            evidence=candidate,
            metadata={
                "positive_evidence": candidate.positive_evidence,
                "novelty": candidate.novelty,
                "revisit_penalty": candidate.revisit_penalty,
                "delay": candidate.delay,
                "total_score": candidate.total_score,
            },
        )

    @classmethod
    def from_weighted_evidence(cls, evidence: WeightedEvidence) -> QuantumCandidate:
        """Construct a QuantumCandidate from a time-decayed WeightedEvidence record."""
        return cls(
            node_id=evidence.node_id,
            score=float(evidence.effective_confidence),
            evidence=evidence,
            metadata={
                "effective_confidence": evidence.effective_confidence,
                "weight": evidence.weight,
                "age": evidence.age,
                "lambda_hat": evidence.lambda_hat,
                "is_positive": evidence.is_positive,
            },
        )

    @classmethod
    def from_node_evidence(cls, evidence: NodeEvidence) -> QuantumCandidate:
        """Construct a QuantumCandidate from a raw NodeEvidence record."""
        return cls(
            node_id=evidence.node_id,
            score=float(evidence.signal_strength if evidence.is_positive else 0.0),
            evidence=evidence,
            metadata={
                "confidence": evidence.confidence,
                "signal_strength": evidence.signal_strength,
                "target_found": evidence.target_found,
                "is_positive": evidence.is_positive,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate to a serializable dictionary."""
        return {
            "node_id": self.node_id,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class QuantumSearchConfig:
    """Configuration parameters for the quantum search simulation.

    Attributes:
        seed: Optional RNG seed for deterministic quantum projective measurement sampling.
        max_iterations: Optional upper bound or override for Grover iterations. If None,
            the theoretically optimal count R = floor(pi / 4 * sqrt(N / |M|)) is computed.
        target_threshold: Minimum evidence score required for marking under threshold rules.
        score_field: Field name inside candidate metadata or primary score to evaluate.
        allow_padding_retry: Whether to resample if a measurement lands on a padded state index.
        max_measurement_retries: Maximum number of measurement resamples before failing safely.
    """
    seed: int | None = None
    max_iterations: int | None = None
    target_threshold: float | None = None
    score_field: str = "score"
    allow_padding_retry: bool = True
    max_measurement_retries: int = 10

    def __post_init__(self) -> None:
        if self.max_iterations is not None and self.max_iterations < 0:
            raise ValueError(f"max_iterations must be non-negative, got {self.max_iterations}")
        if self.target_threshold is not None and not (0.0 <= self.target_threshold <= 1.0):
            raise ValueError(f"target_threshold must be in [0.0, 1.0], got {self.target_threshold}")
        if self.max_measurement_retries < 1:
            raise ValueError("max_measurement_retries must be at least 1")


@dataclass(frozen=True)
class QuantumSearchResult:
    """Structured and explainable result emitted by quantum search operations.

    Attributes:
        selected_candidate: Original graph node identifier of the chosen candidate, or None.
        selected_index: Index in [0, K-1] of the chosen candidate, or None.
        marked_candidates: Tuple of original node identifiers that satisfied the marking rule.
        marked_indices: Tuple of candidate indices that satisfied the marking rule.
        iteration_count: Total Grover iterations executed during statevector evolution.
        oracle_call_count: Total queries evaluated by the quantum oracle.
        success: True if and only if a valid marked candidate was chosen.
        status: Categorical QuantumSearchStatus code.
        probabilities: Mapping from state index to probability |psi(x)|^2 after evolution.
        diagnostic_metadata: Detailed quantum register diagnostics and parameters.
    """
    selected_candidate: NodeId | None
    selected_index: int | None
    marked_candidates: tuple[NodeId, ...]
    marked_indices: tuple[int, ...]
    iteration_count: int
    oracle_call_count: int
    success: bool
    status: QuantumSearchStatus
    probabilities: Mapping[int, float] = field(default_factory=dict)
    diagnostic_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert search result to a serializable dictionary."""
        return {
            "selected_candidate": self.selected_candidate,
            "selected_index": self.selected_index,
            "marked_candidates": list(self.marked_candidates),
            "marked_indices": list(self.marked_indices),
            "iteration_count": self.iteration_count,
            "oracle_call_count": self.oracle_call_count,
            "success": self.success,
            "status": self.status.value,
            "probabilities": {str(k): v for k, v in self.probabilities.items()},
            "diagnostic_metadata": dict(self.diagnostic_metadata),
        }
