"""
Node-Mediated Information Sharing and Communication
===================================================
Implements the decentralized, node-mediated communication architecture for
Adaptive Evidence-Sharing Walkers (AESW).

Fundamental Communication Topology:
    Walker A
       │
       ▼ (PUSH)
   Node Cache
       │
       ├──────────► (PULL) Walker B
       │
       └──────────► (PULL) Walker C

Architectural Invariants:
1. Walkers MUST NOT communicate directly with one another (no direct A -> B link).
2. All information exchange is mediated strictly through node-local shared caches.
3. Evidence records retain their original creator identity (walker_id == A).
4. Evidence arriving via node cache is stamped with EvidenceSource.RECEIVED_EXCHANGE.
5. Communication is strictly measured according to research cost model C = M + c_m * Q.
6. Epistemic firewall: Zero leakage of hidden environment state, p_on/p_off, or true target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Union

from aesw.memory.cache import EvidenceCache
from aesw.memory.models import NodeEvidence
from aesw.memory.types import EvidenceSource


class CommunicationMode(str, Enum):
    """Categorical inter-walker communication modes.

    - NO_SHARING: No evidence exchange across walkers; Q = 0; strict epistemic isolation.
    - PUSH: Walkers deposit locally acquired evidence into the shared node cache.
    - PULL: Walkers request and retrieve available shared evidence from the node cache.
    - PUSH_PULL: Walkers both deposit local evidence and retrieve available shared evidence.
    """
    NO_SHARING = "NO_SHARING"
    PUSH = "PUSH"
    PULL = "PULL"
    PUSH_PULL = "PUSH_PULL"

    @property
    def can_push(self) -> bool:
        """Whether this communication mode permits publishing to node caches."""
        return self in (CommunicationMode.PUSH, CommunicationMode.PUSH_PULL)

    @property
    def can_pull(self) -> bool:
        """Whether this communication mode permits retrieving from node caches."""
        return self in (CommunicationMode.PULL, CommunicationMode.PUSH_PULL)


def validate_communication_mode(mode: Union[str, CommunicationMode]) -> CommunicationMode:
    """Validate and convert an input to a CommunicationMode enum instance.

    Args:
        mode: String or enum representation of communication mode.

    Returns:
        Validated CommunicationMode instance.

    Raises:
        ValueError: If mode is not a valid CommunicationMode.
    """
    if isinstance(mode, CommunicationMode):
        return mode
    try:
        return CommunicationMode(str(mode).upper())
    except (ValueError, KeyError):
        valid = [m.value for m in CommunicationMode]
        raise ValueError(f"Invalid communication mode: {mode}. Must be one of {valid}")


@dataclass(frozen=True)
class CommunicationEvent:
    """Immutable audit record representing a discrete node-mediated communication transaction.

    Attributes:
        action: Transaction action identifier ("PUSH" or "PULL").
        walker_id: Identifier of the walker participating in this transaction.
        node_id: Identifier of the mediating graph node.
        mode: Communication mode under which the transaction took place.
        timestamp: Simulation time step t at which communication occurred.
        evidence_count: Number of evidence records bundled in this transaction.
        message_cost: Incremental communication message units Q added (default 1).
    """
    action: str
    walker_id: Union[int, str]
    node_id: Union[int, str]
    mode: CommunicationMode
    timestamp: int
    evidence_count: int
    message_cost: int = 1

    def __post_init__(self) -> None:
        if self.action not in ("PUSH", "PULL"):
            raise ValueError(f"action must be 'PUSH' or 'PULL', got {self.action}")
        if self.walker_id is None:
            raise ValueError("walker_id must not be None")
        if self.node_id is None:
            raise ValueError("node_id must not be None")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if self.evidence_count < 0:
            raise ValueError(f"evidence_count must be non-negative, got {self.evidence_count}")
        if self.message_cost < 0:
            raise ValueError(f"message_cost must be non-negative, got {self.message_cost}")


@dataclass
class CommunicationMetrics:
    """Tracks cumulative communication message accounting Q and diagnostic metrics.

    Message Accounting Convention:
    - 1 push transaction (depositing 1+ evidence items to node cache) = 1 message.
    - 1 pull transaction (querying node cache for available evidence) = 1 message.
    - NO_SHARING = 0 messages.
    - Total messages Q = push_messages + pull_messages.

    Attributes:
        total_messages: Total communication message count Q (used in C = M + c_m * Q).
        messages_sent: Total messages dispatched by walkers to node stores.
        messages_received: Total messages processed by node stores or walkers.
        push_messages: Total push transactions executed.
        pull_messages: Total pull transactions executed.
        evidence_records_pushed: Total evidence items deposited into shared stores.
        evidence_records_pulled: Total evidence items retrieved from shared stores.
        events: Chronological sequence of communication event records.
    """
    total_messages: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    push_messages: int = 0
    pull_messages: int = 0
    evidence_records_pushed: int = 0
    evidence_records_pulled: int = 0
    events: list[CommunicationEvent] = field(default_factory=list)

    def record_event(self, event: CommunicationEvent) -> None:
        """Record an immutable communication event and update cumulative message counters."""
        self.events.append(event)
        self.total_messages += event.message_cost
        self.messages_sent += event.message_cost
        self.messages_received += event.message_cost

        if event.action == "PUSH":
            self.push_messages += event.message_cost
            self.evidence_records_pushed += event.evidence_count
        elif event.action == "PULL":
            self.pull_messages += event.message_cost
            self.evidence_records_pulled += event.evidence_count

    def reset(self) -> None:
        """Reset all communication metrics back to zero."""
        self.total_messages = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.push_messages = 0
        self.pull_messages = 0
        self.evidence_records_pushed = 0
        self.evidence_records_pulled = 0
        self.events.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a diagnostic dictionary."""
        return {
            "total_messages": self.total_messages,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "push_messages": self.push_messages,
            "pull_messages": self.pull_messages,
            "evidence_records_pushed": self.evidence_records_pushed,
            "evidence_records_pulled": self.evidence_records_pulled,
            "num_events": len(self.events),
        }


class SharedNodeCache:
    """Node-mediated shared evidence cache located at a specific graph vertex.

    Encapsulates a vertex-anchored EvidenceCache that holds evidence deposited
    by passing searchers. Walkers do not communicate directly; they interact
    with this localized repository.

    Attributes:
        node_id: Identifier of the graph vertex hosting this cache.
        max_capacity: Optional upper bound on the number of stored evidence items.
    """

    def __init__(
        self,
        node_id: Union[int, str],
        max_capacity: Optional[int] = None,
    ) -> None:
        if node_id is None:
            raise ValueError("node_id must not be None")
        self._node_id: Union[int, str] = node_id
        self._cache: EvidenceCache = EvidenceCache(max_capacity=max_capacity)

    @property
    def node_id(self) -> Union[int, str]:
        """Identifier of the hosting vertex."""
        return self._node_id

    @property
    def size(self) -> int:
        """Number of distinct vertex evidence records stored at this node."""
        return self._cache.size

    @property
    def tracked_nodes(self) -> list[Union[int, str]]:
        """List of target vertex IDs with stored evidence in this node cache."""
        return self._cache.tracked_nodes

    def store(self, evidence: NodeEvidence) -> bool:
        """Store or update an evidence record using EvidenceCache's precedence rules."""
        return self._cache.store(evidence)

    def store_many(self, evidence_items: Iterable[NodeEvidence]) -> int:
        """Store multiple evidence records, returning the count of new/updated entries."""
        count = 0
        for ev in evidence_items:
            if self.store(ev):
                count += 1
        return count

    def get_all(self) -> list[NodeEvidence]:
        """Retrieve all raw NodeEvidence records currently stored at this node."""
        records: list[NodeEvidence] = []
        for nid in self._cache.tracked_nodes:
            raw = self._cache.get_raw(nid)
            if raw is not None:
                records.append(raw)
        return records

    def get_raw(self, target_node: Union[int, str]) -> Optional[NodeEvidence]:
        """Retrieve the raw NodeEvidence for a specific target node, if present."""
        return self._cache.get_raw(target_node)

    def clear(self) -> None:
        """Clear all stored evidence from this node cache."""
        self._cache.clear()

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"SharedNodeCache(node_id={self._node_id!r}, size={self.size})"


class NodeMediatedExchange:
    """Mediates inter-walker evidence exchange strictly through localized node caches.

    This class serves as the communication substrate for AESW walkers. It manages
    the collection of vertex-anchored SharedNodeCache instances, coordinates
    evidence publishing (push) and retrieval (pull), enforces communication modes,
    and accurately tracks communication costs Q.

    Guarantees:
    - Zero direct walker-to-walker messaging paths.
    - Zero communication overhead in NO_SHARING mode (Q = 0).
    - Source creator identity preserved on exchanged records.
    - Exchanged records tagged with EvidenceSource.RECEIVED_EXCHANGE.
    """

    def __init__(
        self,
        mode: Union[str, CommunicationMode] = CommunicationMode.NO_SHARING,
        node_capacity: Optional[int] = None,
    ) -> None:
        """Initialize the node-mediated communication exchange.

        Args:
            mode: Operating communication mode (NO_SHARING, PUSH, PULL, PUSH_PULL).
            node_capacity: Optional upper bound on per-node cache entries.
        """
        self._mode: CommunicationMode = validate_communication_mode(mode)
        self._node_capacity: Optional[int] = node_capacity
        self._node_caches: dict[Union[int, str], SharedNodeCache] = {}
        self._metrics: CommunicationMetrics = CommunicationMetrics()

    @property
    def mode(self) -> CommunicationMode:
        """Currently configured communication mode."""
        return self._mode

    @mode.setter
    def mode(self, new_mode: Union[str, CommunicationMode]) -> None:
        """Update the active communication mode."""
        self._mode = validate_communication_mode(new_mode)

    @property
    def metrics(self) -> CommunicationMetrics:
        """Cumulative communication metrics container."""
        return self._metrics

    @property
    def total_messages(self) -> int:
        """Total communication messages Q exchanged."""
        return self._metrics.total_messages

    def get_node_cache(self, node_id: Union[int, str]) -> SharedNodeCache:
        """Retrieve or lazily initialize the shared cache for a specific vertex.

        Args:
            node_id: Vertex identifier.

        Returns:
            SharedNodeCache instance anchored at node_id.
        """
        if node_id not in self._node_caches:
            self._node_caches[node_id] = SharedNodeCache(
                node_id=node_id,
                max_capacity=self._node_capacity,
            )
        return self._node_caches[node_id]

    def has_node_cache(self, node_id: Union[int, str]) -> bool:
        """Return True if a shared cache has been initialized at node_id."""
        return node_id in self._node_caches

    def push(
        self,
        walker_id: Union[int, str],
        at_node: Union[int, str],
        evidence: Union[NodeEvidence, Iterable[NodeEvidence]],
        timestamp: int,
    ) -> int:
        """Deposit evidence from a walker into the shared cache at at_node.

        Behavior:
        - If mode does not allow push (NO_SHARING or PULL), returns 0; Q unchanged.
        - If evidence is empty, returns 0; Q unchanged.
        - Otherwise, deposits records into the node cache and counts 1 push message (Q += 1).

        Args:
            walker_id: Identifier of the publishing walker.
            at_node: Vertex where the walker is physically depositing evidence.
            evidence: Single NodeEvidence or iterable of NodeEvidence to publish.
            timestamp: Simulation time step t.

        Returns:
            Number of evidence records successfully stored or updated.
        """
        if not self._mode.can_push:
            return 0

        if isinstance(evidence, NodeEvidence):
            items = [evidence]
        else:
            items = list(evidence)

        if not items:
            return 0

        node_cache = self.get_node_cache(at_node)
        stored_count = node_cache.store_many(items)

        event = CommunicationEvent(
            action="PUSH",
            walker_id=walker_id,
            node_id=at_node,
            mode=self._mode,
            timestamp=timestamp,
            evidence_count=len(items),
            message_cost=1,
        )
        self._metrics.record_event(event)
        return stored_count

    def push_from_cache(
        self,
        walker_id: Union[int, str],
        at_node: Union[int, str],
        local_cache: EvidenceCache,
        timestamp: int,
    ) -> int:
        """Convenience method to deposit all entries from a walker's local cache.

        Args:
            walker_id: Identifier of the publishing walker.
            at_node: Vertex where the walker is physically publishing.
            local_cache: Walker's private local EvidenceCache.
            timestamp: Simulation time step t.

        Returns:
            Number of evidence records successfully stored or updated.
        """
        if not self._mode.can_push:
            return 0

        entries = [
            local_cache.get_raw(nid)
            for nid in local_cache.tracked_nodes
            if local_cache.get_raw(nid) is not None
        ]
        return self.push(walker_id=walker_id, at_node=at_node, evidence=entries, timestamp=timestamp)

    def pull(
        self,
        walker_id: Union[int, str],
        at_node: Union[int, str],
        timestamp: int,
        target_cache: Optional[EvidenceCache] = None,
        exclude_self: bool = False,
    ) -> list[NodeEvidence]:
        """Request and retrieve available shared evidence from the cache at at_node.

        Behavior:
        - If mode does not allow pull (NO_SHARING or PUSH), returns []; Q unchanged.
        - Otherwise, retrieves records from at_node, transforms each to RECEIVED_EXCHANGE
          (preserving original creator walker_id), merges into target_cache (if provided),
          and counts 1 pull message (Q += 1).

        Args:
            walker_id: Identifier of the requesting walker.
            at_node: Vertex where the walker is querying for shared evidence.
            timestamp: Simulation time step t.
            target_cache: Optional local EvidenceCache into which received records are merged.
            exclude_self: If True, filters out evidence originally created by walker_id.

        Returns:
            List of received NodeEvidence records (with source=RECEIVED_EXCHANGE).
        """
        if not self._mode.can_pull:
            return []

        node_cache = self.get_node_cache(at_node)
        raw_items = node_cache.get_all()

        if exclude_self:
            eligible = [ev for ev in raw_items if ev.walker_id != walker_id]
        else:
            eligible = raw_items

        received_items: list[NodeEvidence] = []
        for ev in eligible:
            # Stamp with RECEIVED_EXCHANGE while strictly preserving original creator walker_id
            rec = ev.as_received()
            received_items.append(rec)
            if target_cache is not None:
                target_cache.store(rec)

        event = CommunicationEvent(
            action="PULL",
            walker_id=walker_id,
            node_id=at_node,
            mode=self._mode,
            timestamp=timestamp,
            evidence_count=len(received_items),
            message_cost=1,
        )
        self._metrics.record_event(event)
        return received_items

    def exchange(
        self,
        walker_id: Union[int, str],
        at_node: Union[int, str],
        local_cache: EvidenceCache,
        timestamp: int,
        exclude_self: bool = False,
    ) -> tuple[int, int]:
        """Perform unified node-mediated communication exchange at at_node based on active mode.

        In PUSH: deposits local_cache contents.
        In PULL: retrieves shared evidence into local_cache.
        In PUSH_PULL: executes push followed by pull.
        In NO_SHARING: executes neither; returns (0, 0); Q remains 0.

        Args:
            walker_id: Identifier of the participating walker.
            at_node: Vertex where the walker is physically positioned.
            local_cache: Walker's private local EvidenceCache.
            timestamp: Simulation time step t.
            exclude_self: Whether to exclude self-authored records during pull.

        Returns:
            Tuple (pushed_record_count, pulled_record_count).
        """
        pushed_count = 0
        pulled_count = 0

        if self._mode.can_push:
            pushed_count = self.push_from_cache(
                walker_id=walker_id,
                at_node=at_node,
                local_cache=local_cache,
                timestamp=timestamp,
            )

        if self._mode.can_pull:
            pulled_items = self.pull(
                walker_id=walker_id,
                at_node=at_node,
                timestamp=timestamp,
                target_cache=local_cache,
                exclude_self=exclude_self,
            )
            pulled_count = len(pulled_items)

        return pushed_count, pulled_count

    def reset(self) -> None:
        """Reset all node stores and communication metrics."""
        self._node_caches.clear()
        self._metrics.reset()

    def __repr__(self) -> str:
        return (
            f"NodeMediatedExchange(mode={self._mode.value}, "
            f"nodes_with_cache={len(self._node_caches)}, total_messages={self.total_messages})"
        )
