"""
Quantum Target-Marking Rules and Eligibility Predicates
======================================================
Defines explicit and testable candidate-marking rules for the quantum oracle.

Architectural Principles:
1. Strict Separation: Eligibility determination is fully decoupled from the quantum
   evolution and Grover circuit implementation.
2. Epistemic Humility: A candidate meeting an evidence threshold is an "eligible target
   candidate" for Grover amplification, NOT a guarantee of true target presence.
3. No Phantom Fields: Reuses existing AESW evidence fields (effective_confidence,
   signal_strength, positive_evidence, total_score).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence

from aesw.quantum.types import NodeId, QuantumCandidate


class MarkingRule(ABC):
    """Abstract base class defining candidate eligibility for quantum marking.

    An oracle queries this rule to determine whether a given quantum basis state
    corresponding to a candidate should receive a phase inversion (-1).
    """

    @abstractmethod
    def is_marked(self, candidate: QuantumCandidate) -> bool:
        """Evaluate whether a candidate satisfies the marking condition.

        Args:
            candidate: Strongly typed QuantumCandidate instance.

        Returns:
            True if candidate is marked (phase-inverted), False otherwise.
        """

    def filter_marked(self, candidates: Sequence[QuantumCandidate]) -> list[QuantumCandidate]:
        """Filter a collection of candidates to return only those marked by this rule.

        Args:
            candidates: Sequence of candidates.

        Returns:
            List of candidates where is_marked(c) is True.
        """
        return [c for c in candidates if self.is_marked(c)]


class ThresholdMarkingRule(MarkingRule):
    """Marks candidates whose evidence score satisfies a numerical threshold.

    Attributes:
        threshold: Minimum score required for marking (must be in [0.0, 1.0]).
        score_field: Target field name to inspect ('score', 'effective_confidence',
            'positive_evidence', 'signal_strength', 'total_score').
        strict: If True, missing scores raise ValueError. If False, treated as unmarked.
    """

    def __init__(
        self,
        threshold: float,
        score_field: str = "score",
        strict: bool = True,
    ) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        self._threshold = float(threshold)
        self._score_field = str(score_field)
        self._strict = bool(strict)

    @property
    def threshold(self) -> float:
        """Minimum threshold value."""
        return self._threshold

    @property
    def score_field(self) -> str:
        """Target score attribute name."""
        return self._score_field

    def is_marked(self, candidate: QuantumCandidate) -> bool:
        """Check if candidate score meets or exceeds the configured threshold."""
        score_val: float | None = None

        if self._score_field == "score":
            score_val = candidate.score
        elif self._score_field in candidate.metadata:
            raw = candidate.metadata[self._score_field]
            score_val = float(raw) if raw is not None else None
        elif candidate.score is not None and self._score_field in ("positive_evidence", "effective_confidence"):
            # Fall back to primary score if standard evidence field is requested
            score_val = candidate.score

        if score_val is None:
            if self._strict:
                raise ValueError(
                    f"Candidate '{candidate.node_id}' is missing required score field '{self._score_field}'"
                )
            return False

        return score_val >= self._threshold

    def __repr__(self) -> str:
        return f"ThresholdMarkingRule(threshold={self._threshold}, field='{self._score_field}')"


class PredicateMarkingRule(MarkingRule):
    """Marks candidates according to an arbitrary user-defined callable predicate.

    Attributes:
        predicate: Callable taking a QuantumCandidate and returning a boolean.
        name: Human-readable diagnostic description of the predicate.
    """

    def __init__(
        self,
        predicate: Callable[[QuantumCandidate], bool],
        name: str = "custom_predicate",
    ) -> None:
        if not callable(predicate):
            raise TypeError("predicate must be a callable accepting QuantumCandidate")
        self._predicate = predicate
        self._name = str(name)

    @property
    def name(self) -> str:
        """Name of the predicate."""
        return self._name

    def is_marked(self, candidate: QuantumCandidate) -> bool:
        """Evaluate the underlying predicate on candidate."""
        return bool(self._predicate(candidate))

    def __repr__(self) -> str:
        return f"PredicateMarkingRule(name='{self._name}')"


class TargetIdMarkingRule(MarkingRule):
    """Marks candidates whose node identifiers belong to a specified target set.

    Primarily used for oracle verification, ground-truth tests, and diagnostic benchmarks.

    Attributes:
        target_ids: Set of designated target node identifiers.
    """

    def __init__(self, target_ids: Iterable[NodeId]) -> None:
        self._target_ids: set[NodeId] = set(target_ids)

    @property
    def target_ids(self) -> frozenset[NodeId]:
        """Immutable set of target node IDs."""
        return frozenset(self._target_ids)

    def is_marked(self, candidate: QuantumCandidate) -> bool:
        """Check if candidate node ID is in the target set."""
        return candidate.node_id in self._target_ids

    def __repr__(self) -> str:
        return f"TargetIdMarkingRule(target_ids={sorted(self._target_ids, key=str)})"
