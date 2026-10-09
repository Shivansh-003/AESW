"""
Evidence Data Models
====================
Provides formal immutable representations of node-level evidence records
and evaluated weighted evidence snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Optional, Union

from aesw.memory.types import EvidencePolarity, EvidenceSource
from aesw.memory.decay import exponential_decay


@dataclass(frozen=True)
class NodeEvidence:
    """Formal structured record of evidence collected at or regarding a graph vertex.

    Attributes:
        node_id: Identifier of the vertex v ∈ V.
        visited: Boolean indicating whether the walker physically occupied this vertex.
        target_found: Boolean indicating verified target acquisition at this vertex.
        signal_strength: Observed sensor reading in [0.0, 1.0].
        timestamp: Simulation time step t when this evidence was gathered (t >= 0).
        walker_id: Identifier of the walker agent that recorded this evidence.
        confidence: Baseline reliability / sensor confidence c_0 ∈ [0.0, 1.0].
        polarity: EvidencePolarity (POSITIVE or NEGATIVE). Inferred if omitted.
        source: Provenance source (DIRECT_VISIT, REMOTE_SENSOR, RECEIVED_EXCHANGE).
        metadata: Optional arbitrary metadata attributes.
    """
    node_id: Union[int, str]
    visited: bool
    target_found: bool
    signal_strength: float
    timestamp: int
    walker_id: Union[int, str]
    confidence: float
    polarity: EvidencePolarity = field(default=None)  # type: ignore[assignment]
    source: EvidenceSource = EvidenceSource.DIRECT_VISIT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_id is None:
            raise ValueError("node_id must not be None")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if not (0.0 <= self.signal_strength <= 1.0):
            raise ValueError(f"signal_strength must be in [0.0, 1.0], got {self.signal_strength}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.walker_id is None:
            raise ValueError("walker_id must not be None")

        # Automatically infer polarity if omitted
        if self.polarity is None:
            inferred = (
                EvidencePolarity.POSITIVE
                if (self.target_found or self.signal_strength > 0.0)
                else EvidencePolarity.NEGATIVE
            )
            object.__setattr__(self, "polarity", inferred)
        elif not isinstance(self.polarity, EvidencePolarity):
            object.__setattr__(self, "polarity", EvidencePolarity(self.polarity))

        if not isinstance(self.source, EvidenceSource):
            object.__setattr__(self, "source", EvidenceSource(self.source))

    @property
    def is_positive(self) -> bool:
        """Whether this record represents positive evidence (target indication)."""
        return self.polarity == EvidencePolarity.POSITIVE

    @property
    def is_negative(self) -> bool:
        """Whether this record represents negative evidence (absence confirmation)."""
        return self.polarity == EvidencePolarity.NEGATIVE

    def age(self, current_time: int) -> int:
        """Compute the elapsed simulation time since this evidence was recorded.

        Args:
            current_time: Current simulation timestamp t.

        Returns:
            Non-negative age: current_time - timestamp.

        Raises:
            ValueError: If current_time is earlier than the evidence timestamp.
        """
        if current_time < self.timestamp:
            raise ValueError(
                f"current_time ({current_time}) cannot be earlier than evidence timestamp ({self.timestamp})"
            )
        return int(current_time - self.timestamp)

    def weight(self, current_time: int, lambda_hat: float) -> float:
        """Compute the current time-dependent evidence weight w = exp(-lambda_hat * age).

        Args:
            current_time: Current simulation timestamp t.
            lambda_hat: Estimated or configured churn rate (must be >= 0.0).

        Returns:
            Evidence weight w in [0.0, 1.0].
        """
        return exponential_decay(age=self.age(current_time), lambda_hat=lambda_hat)

    def effective_confidence(self, current_time: int, lambda_hat: float) -> float:
        """Compute decayed effective confidence: confidence * w.

        Args:
            current_time: Current simulation timestamp t.
            lambda_hat: Estimated or configured churn rate.

        Returns:
            Decayed confidence scalar in [0.0, 1.0].
        """
        return float(self.confidence * self.weight(current_time, lambda_hat))

    def evaluate(self, current_time: int, lambda_hat: float) -> WeightedEvidence:
        """Construct an immutable WeightedEvidence snapshot evaluated at current_time.

        Args:
            current_time: Current simulation timestamp t.
            lambda_hat: Estimated churn rate.

        Returns:
            WeightedEvidence snapshot.
        """
        elapsed_age = self.age(current_time)
        w = exponential_decay(age=elapsed_age, lambda_hat=lambda_hat)
        eff_conf = float(self.confidence * w)

        return WeightedEvidence(
            evidence=self,
            current_time=current_time,
            lambda_hat=lambda_hat,
            age=elapsed_age,
            weight=w,
            effective_confidence=eff_conf,
        )

    def as_received(self) -> NodeEvidence:
        """Return an immutable copy marked as RECEIVED_EXCHANGE, preserving original creator walker_id."""
        return NodeEvidence(
            node_id=self.node_id,
            visited=self.visited,
            target_found=self.target_found,
            signal_strength=self.signal_strength,
            timestamp=self.timestamp,
            walker_id=self.walker_id,
            confidence=self.confidence,
            polarity=self.polarity,
            source=EvidenceSource.RECEIVED_EXCHANGE,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence record to a dictionary."""
        return {
            "node_id": self.node_id,
            "visited": self.visited,
            "target_found": self.target_found,
            "signal_strength": self.signal_strength,
            "timestamp": self.timestamp,
            "walker_id": self.walker_id,
            "confidence": self.confidence,
            "polarity": self.polarity.value,
            "source": self.source.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NodeEvidence:
        """Construct a NodeEvidence instance from a dictionary."""
        return cls(
            node_id=data["node_id"],
            visited=bool(data.get("visited", True)),
            target_found=bool(data.get("target_found", False)),
            signal_strength=float(data.get("signal_strength", 0.0)),
            timestamp=int(data["timestamp"]),
            walker_id=data["walker_id"],
            confidence=float(data.get("confidence", 1.0)),
            polarity=EvidencePolarity(data["polarity"]) if "polarity" in data else None,
            source=EvidenceSource(data.get("source", EvidenceSource.DIRECT_VISIT)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class WeightedEvidence:
    """Evaluated snapshot of a NodeEvidence record with time-decayed metrics.

    Attributes:
        evidence: Underlying NodeEvidence record.
        current_time: Simulation timestamp at which this evaluation occurred.
        lambda_hat: Churn rate used for exponential decay evaluation.
        age: Elapsed simulation steps (current_time - evidence.timestamp).
        weight: Decay weight w = exp(-lambda_hat * age) in [0.0, 1.0].
        effective_confidence: Decayed confidence = evidence.confidence * weight.
    """
    evidence: NodeEvidence
    current_time: int
    lambda_hat: float
    age: int
    weight: float
    effective_confidence: float

    @property
    def node_id(self) -> Union[int, str]:
        return self.evidence.node_id

    @property
    def visited(self) -> bool:
        return self.evidence.visited

    @property
    def target_found(self) -> bool:
        return self.evidence.target_found

    @property
    def signal_strength(self) -> float:
        return self.evidence.signal_strength

    @property
    def timestamp(self) -> int:
        return self.evidence.timestamp

    @property
    def walker_id(self) -> Union[int, str]:
        return self.evidence.walker_id

    @property
    def confidence(self) -> float:
        return self.evidence.confidence

    @property
    def polarity(self) -> EvidencePolarity:
        return self.evidence.polarity

    @property
    def source(self) -> EvidenceSource:
        return self.evidence.source

    @property
    def is_positive(self) -> bool:
        return self.evidence.is_positive

    @property
    def is_negative(self) -> bool:
        return self.evidence.is_negative

    def to_dict(self) -> dict[str, Any]:
        """Convert weighted evidence snapshot to a dictionary."""
        return {
            "evidence": self.evidence.to_dict(),
            "current_time": self.current_time,
            "lambda_hat": self.lambda_hat,
            "age": self.age,
            "weight": self.weight,
            "effective_confidence": self.effective_confidence,
        }
