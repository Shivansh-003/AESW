"""
Quantum Candidate Preparation and Index Mapping Layer
=====================================================
Manages deterministic bidirectional mapping between classical graph node identifiers
and discrete quantum basis state indices in the search register.

Handles register sizing, non-power-of-two padding, input validation, and boundary enforcement.

Mathematical Formulation:
    For K classical candidates:
        n = ceil(log2(K))  (Number of address qubits, with n >= 1 for K >= 1)
        N = 2^n            (Hilbert space basis dimension)

    State Allocation:
        Indices 0 <= i < K:     Valid physical candidates mapped to graph nodes.
        Indices K <= i < N:     Padded ancillary basis states (dummy non-targets).

Epistemic & Safety Invariants:
1. Padded states (i >= K) are NEVER valid graph nodes and are NEVER marked by the oracle.
2. If measurement yields a padded index i in [K, N-1], reverse mapping safely detects it
   and prevents phantom node identifiers from entering the classical pipeline.
3. Node identifier mappings are strictly deterministic and reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from aesw.quantum.types import NodeId, QuantumCandidate


class CandidateRegister:
    """Manages the mapping between classical graph nodes and quantum register indices.

    Attributes:
        candidates: Immutable tuple of validated QuantumCandidate instances.
        num_candidates: Physical candidate count K.
        num_qubits: Address register qubit count n = ceil(log2(K)).
        dimension: Total basis state count N = 2^n.
        num_padded: Number of unused basis states N - K.
    """

    def __init__(
        self,
        candidates: Sequence[QuantumCandidate | NodeId],
        sort_by_id: bool = False,
    ) -> None:
        """Initialize the CandidateRegister with candidate nodes.

        Args:
            candidates: Sequence of QuantumCandidate objects or raw node IDs.
            sort_by_id: Whether to sort candidates alphabetically/numerically by node_id
                for canonical ordering. Defaults to False (preserves input sequence order).

        Raises:
            ValueError: If duplicate candidate node IDs are provided.
        """
        # Normalize to QuantumCandidate instances
        normalized: list[QuantumCandidate] = []
        seen_ids: set[NodeId] = set()

        for c in candidates:
            cand = c if isinstance(c, QuantumCandidate) else QuantumCandidate.from_node_id(c)
            if cand.node_id in seen_ids:
                raise ValueError(
                    f"Duplicate candidate node ID '{cand.node_id}' detected. "
                    f"Candidate register requires unique candidate identifiers."
                )
            seen_ids.add(cand.node_id)
            normalized.append(cand)

        if sort_by_id:
            normalized.sort(key=lambda x: str(x.node_id))

        self._candidates: tuple[QuantumCandidate, ...] = tuple(normalized)
        self._k = len(self._candidates)

        # Build bidirectional maps
        self._index_to_candidate: dict[int, QuantumCandidate] = {
            i: c for i, c in enumerate(self._candidates)
        }
        self._node_id_to_index: dict[NodeId, int] = {
            c.node_id: i for i, c in enumerate(self._candidates)
        }

        # Quantum register sizing
        if self._k == 0:
            self._num_qubits = 0
            self._dimension = 0
            self._num_padded = 0
        else:
            self._num_qubits = max(1, math.ceil(math.log2(self._k)))
            self._dimension = 1 << self._num_qubits
            self._num_padded = self._dimension - self._k

    @property
    def candidates(self) -> tuple[QuantumCandidate, ...]:
        """All valid physical candidates in register index order."""
        return self._candidates

    @property
    def num_candidates(self) -> int:
        """Physical candidate count K."""
        return self._k

    @property
    def num_qubits(self) -> int:
        """Number of address qubits n."""
        return self._num_qubits

    @property
    def dimension(self) -> int:
        """Full Hilbert space basis dimension N = 2^n."""
        return self._dimension

    @property
    def num_padded(self) -> int:
        """Count of padded basis states N - K."""
        return self._num_padded

    @property
    def is_empty(self) -> bool:
        """Whether candidate register contains 0 candidates."""
        return self._k == 0

    @property
    def is_power_of_two(self) -> bool:
        """Whether physical candidate count K is an exact power of two."""
        return self._k > 0 and (self._k & (self._k - 1)) == 0

    def is_valid_index(self, index: int) -> bool:
        """Check if an index corresponds to a real physical candidate (0 <= index < K)."""
        return 0 <= index < self._k

    def is_padded_index(self, index: int) -> bool:
        """Check if an index represents a padded ancillary state (K <= index < N)."""
        return self._k <= index < self._dimension

    def get_candidate(self, index: int) -> QuantumCandidate:
        """Retrieve candidate for index in [0, K-1].

        Raises:
            IndexError: If index is out of bounds or represents a padded state.
        """
        if not self.is_valid_index(index):
            if self.is_padded_index(index):
                raise IndexError(
                    f"Index {index} is a padded ancillary basis state, not a valid graph candidate"
                )
            raise IndexError(f"Candidate index {index} out of range [0, {self._k - 1}]")
        return self._index_to_candidate[index]

    def get_node_id(self, index: int) -> NodeId:
        """Retrieve original node identifier for index in [0, K-1].

        Raises:
            IndexError: If index is out of bounds or represents a padded state.
        """
        return self.get_candidate(index).node_id

    def get_index(self, node_id: NodeId) -> int:
        """Retrieve register index for a node ID.

        Raises:
            KeyError: If node_id is not in register.
        """
        if node_id not in self._node_id_to_index:
            raise KeyError(f"Node ID '{node_id}' not found in candidate register")
        return self._node_id_to_index[node_id]

    def reverse_map(self, index: int) -> NodeId | None:
        """Safely map a measured basis index back to a node ID.

        Returns:
            The original NodeId if 0 <= index < K, or None if index is padded or invalid.
        """
        if self.is_valid_index(index):
            return self._index_to_candidate[index].node_id
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize register metadata and index mappings for diagnostics."""
        return {
            "num_candidates": self._k,
            "num_qubits": self._num_qubits,
            "dimension": self._dimension,
            "num_padded": self._num_padded,
            "is_power_of_two": self.is_power_of_two,
            "index_mapping": {i: c.node_id for i, c in self._index_to_candidate.items()},
        }
