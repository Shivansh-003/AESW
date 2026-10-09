"""
Local Node-Level Evidence Memory and Cache
==========================================
Implements the local evidence memory container managing evidence storage,
retrieval, update rules, and time-dependent exponential decay weighting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from aesw.memory.types import EvidencePolarity, EvidenceSource
from aesw.memory.models import NodeEvidence, WeightedEvidence
from aesw.memory.decay import exponential_decay

if TYPE_CHECKING:
    from aesw.environment.observation import Observation


class EvidenceCache:
    """Local node-level evidence memory cache.

    Maintains historical search evidence collected by a walker (or resident at a node)
    and evaluates time-decayed weights and effective confidences using:
        w = exp(-lambda_hat * age)

    Attributes:
        default_lambda_hat: Fallback graph churn rate for decay evaluations.
        max_capacity: Optional upper bound on stored entries (FIFO / least-recent eviction).
    """

    def __init__(
        self,
        default_lambda_hat: float = 0.05,
        max_capacity: Optional[int] = None,
    ) -> None:
        if default_lambda_hat < 0.0:
            raise ValueError(f"default_lambda_hat must be non-negative, got {default_lambda_hat}")
        if max_capacity is not None and max_capacity <= 0:
            raise ValueError(f"max_capacity must be strictly positive, got {max_capacity}")

        self._default_lambda_hat = float(default_lambda_hat)
        self._max_capacity = max_capacity
        self._entries: dict[Union[int, str], NodeEvidence] = {}
        self._insertion_order: list[Union[int, str]] = []

    @property
    def default_lambda_hat(self) -> float:
        """Default churn rate lambda_hat applied when not explicitly overridden."""
        return self._default_lambda_hat

    @default_lambda_hat.setter
    def default_lambda_hat(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"default_lambda_hat must be non-negative, got {value}")
        self._default_lambda_hat = float(value)

    @property
    def max_capacity(self) -> Optional[int]:
        """Maximum capacity of the evidence cache."""
        return self._max_capacity

    @property
    def size(self) -> int:
        """Number of distinct vertices with stored evidence."""
        return len(self._entries)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, node_id: Union[int, str]) -> bool:
        return node_id in self._entries

    @property
    def tracked_nodes(self) -> tuple[Union[int, str], ...]:
        """Tuple of all node identifiers currently tracked in memory."""
        return tuple(self._entries.keys())

    def store(self, evidence: NodeEvidence) -> bool:
        """Store or update evidence for a node.

        Update Rule:
        - If node is not yet in cache: insert new record.
        - If node exists:
          * If new evidence is strictly newer (timestamp > existing): update.
          * If timestamp is identical and new confidence >= existing: update.
          * If new evidence is older (timestamp < existing): ignore (keep newer evidence).

        Args:
            evidence: The NodeEvidence record to store.

        Returns:
            True if the cache was updated/inserted; False if rejected as older.
        """
        if not isinstance(evidence, NodeEvidence):
            raise TypeError(f"Expected NodeEvidence instance, got {type(evidence).__name__}")

        node = evidence.node_id
        if node in self._entries:
            existing = self._entries[node]
            if evidence.timestamp > existing.timestamp:
                self._entries[node] = evidence
                return True
            elif evidence.timestamp == existing.timestamp:
                if evidence.confidence >= existing.confidence:
                    self._entries[node] = evidence
                    return True
                return False
            else:
                # Strictly older evidence does not overwrite newer observation
                return False
        else:
            # Capacity check: evict oldest entry if at capacity
            if self._max_capacity is not None and len(self._entries) >= self._max_capacity:
                evicted = self._insertion_order.pop(0)
                self._entries.pop(evicted, None)

            self._entries[node] = evidence
            self._insertion_order.append(node)
            return True

    def store_from_observation(
        self,
        observation: Optional[Observation] = None,
        *,
        obs: Optional[Observation] = None,
        current_time: Optional[int] = None,
        visited: bool = True,
        target_found: bool = False,
        confidence: float = 1.0,
        walker_id: Optional[Union[int, str]] = None,
        source: EvidenceSource = EvidenceSource.DIRECT_VISIT,
    ) -> NodeEvidence:
        """Construct and store a NodeEvidence record directly from an Observation.

        Args:
            observation: The partial Observation snapshot (can also be passed as obs).
            obs: Keyword alias for observation.
            current_time: Optional timestamp override (defaults to observation.time).
            visited: Whether the walker physically visited this vertex.
            target_found: Whether the target was physically acquired.
            confidence: Base confidence of the observation.
            walker_id: Optional walker identifier override.
            source: Acquisition mechanism.

        Returns:
            The created and stored NodeEvidence record.
        """
        resolved_obs = observation if observation is not None else obs
        if resolved_obs is None:
            raise ValueError("Must provide an Observation instance via observation or obs")

        t = resolved_obs.time if current_time is None else int(current_time)
        w_id = resolved_obs.walker_id if walker_id is None else walker_id
        node = resolved_obs.current_node

        detected = bool(resolved_obs.detected)
        signal_val = float(resolved_obs.target_signal.signal_strength) if detected else 0.0


        polarity = (
            EvidencePolarity.POSITIVE
            if (target_found or detected)
            else EvidencePolarity.NEGATIVE
        )

        evidence = NodeEvidence(
            node_id=node,
            visited=visited,
            target_found=target_found,
            signal_strength=signal_val,
            timestamp=t,
            walker_id=w_id,
            confidence=confidence,
            polarity=polarity,
            source=source,
        )

        self.store(evidence)
        return evidence

    def get_raw(self, node_id: Union[int, str]) -> Optional[NodeEvidence]:
        """Retrieve the unweighted raw NodeEvidence record for node_id."""
        return self._entries.get(node_id)

    def get(
        self,
        node_id: Union[int, str],
        current_time: int,
        lambda_hat: Optional[float] = None,
    ) -> Optional[WeightedEvidence]:
        """Retrieve the time-decayed weighted evidence for node_id at current_time.

        Args:
            node_id: Vertex identifier.
            current_time: Current simulation time step t.
            lambda_hat: Optional churn rate override. Defaults to default_lambda_hat.

        Returns:
            WeightedEvidence snapshot, or None if node is not in memory.
        """
        raw = self._entries.get(node_id)
        if raw is None:
            return None

        rate = self._default_lambda_hat if lambda_hat is None else float(lambda_hat)
        return raw.evaluate(current_time=current_time, lambda_hat=rate)

    def all_entries(
        self,
        current_time: int,
        lambda_hat: Optional[float] = None,
        min_weight: float = 0.0,
    ) -> dict[Union[int, str], WeightedEvidence]:
        """Retrieve all memory records evaluated at current_time with weight >= min_weight.

        Args:
            current_time: Current simulation timestamp t.
            lambda_hat: Optional churn rate override.
            min_weight: Minimum decay weight threshold for inclusion.

        Returns:
            Mapping from node_id to WeightedEvidence.
        """
        rate = self._default_lambda_hat if lambda_hat is None else float(lambda_hat)
        result: dict[Union[int, str], WeightedEvidence] = {}

        for node_id, raw in self._entries.items():
            evaluated = raw.evaluate(current_time=current_time, lambda_hat=rate)
            if evaluated.weight >= min_weight:
                result[node_id] = evaluated

        return result

    def get_positive(
        self,
        current_time: int,
        lambda_hat: Optional[float] = None,
        min_confidence: float = 0.0,
    ) -> list[WeightedEvidence]:
        """Retrieve positive evidence entries sorted by effective confidence descending.

        Args:
            current_time: Current simulation timestamp t.
            lambda_hat: Churn rate override.
            min_confidence: Minimum decayed effective confidence threshold.

        Returns:
            List of WeightedEvidence instances where is_positive is True.
        """
        entries = self.all_entries(current_time=current_time, lambda_hat=lambda_hat)
        positive_list = [
            w for w in entries.values()
            if w.is_positive and w.effective_confidence >= min_confidence
        ]
        return sorted(positive_list, key=lambda w: w.effective_confidence, reverse=True)

    def get_negative(
        self,
        current_time: int,
        lambda_hat: Optional[float] = None,
        min_confidence: float = 0.0,
    ) -> list[WeightedEvidence]:
        """Retrieve negative evidence entries sorted by effective confidence descending.

        Args:
            current_time: Current simulation timestamp t.
            lambda_hat: Churn rate override.
            min_confidence: Minimum decayed effective confidence threshold.

        Returns:
            List of WeightedEvidence instances where is_negative is True.
        """
        entries = self.all_entries(current_time=current_time, lambda_hat=lambda_hat)
        negative_list = [
            w for w in entries.values()
            if w.is_negative and w.effective_confidence >= min_confidence
        ]
        return sorted(negative_list, key=lambda w: w.effective_confidence, reverse=True)

    def prune_stale(
        self,
        current_time: int,
        min_weight: float = 1e-4,
        lambda_hat: Optional[float] = None,
    ) -> int:
        """Remove stale memory entries whose decay weight has fallen below min_weight.

        Args:
            current_time: Current simulation timestamp t.
            min_weight: Weight cutoff threshold.
            lambda_hat: Churn rate override.

        Returns:
            Count of removed entries.
        """
        rate = self._default_lambda_hat if lambda_hat is None else float(lambda_hat)
        stale_nodes: list[Union[int, str]] = []

        for node_id, raw in self._entries.items():
            w = raw.weight(current_time=current_time, lambda_hat=rate)
            if w < min_weight:
                stale_nodes.append(node_id)

        for node_id in stale_nodes:
            self._entries.pop(node_id, None)
            if node_id in self._insertion_order:
                self._insertion_order.remove(node_id)

        return len(stale_nodes)

    def merge(self, other: EvidenceCache) -> int:
        """Merge records from another EvidenceCache into this cache.

        Uses the standard update rule: strictly newer timestamps or higher confidences overwrite.

        Args:
            other: EvidenceCache to merge from.

        Returns:
            Number of entries newly inserted or updated.
        """
        if not isinstance(other, EvidenceCache):
            raise TypeError(f"Expected EvidenceCache instance, got {type(other).__name__}")

        updated_count = 0
        for node_id in other.tracked_nodes:
            raw = other.get_raw(node_id)
            if raw is not None:
                if self.store(raw):
                    updated_count += 1

        return updated_count

    def clear(self) -> None:
        """Clear all stored evidence records from memory."""
        self._entries.clear()
        self._insertion_order.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize cache state to a dictionary."""
        return {
            "default_lambda_hat": self._default_lambda_hat,
            "max_capacity": self._max_capacity,
            "entries": {str(k): v.to_dict() for k, v in self._entries.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceCache:
        """Construct an EvidenceCache from a serialized dictionary."""
        cache = cls(
            default_lambda_hat=float(data.get("default_lambda_hat", 0.05)),
            max_capacity=data.get("max_capacity"),
        )
        for raw_dict in data.get("entries", {}).values():
            evidence = NodeEvidence.from_dict(raw_dict)
            cache.store(evidence)
        return cache
