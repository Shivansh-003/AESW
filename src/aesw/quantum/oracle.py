"""
Quantum Phase-Flip Oracle Interface
===================================
Implements the quantum oracle operator O for Grover's search algorithm.

Mathematical Formulation:
    The standard Grover oracle acts on computational basis states |x> as:
        O |x> = (-1)^{f(x)} |x>
    where:
        f(x) = 1  if x corresponds to a valid candidate satisfying the marking rule.
        f(x) = 0  if x is an unmarked candidate OR an ancillary padded basis state.

Architectural Guarantees:
1. Strict Predicate Fidelity: Exactly candidates meeting marking_rule.is_marked() receive
   the phase inversion -1.
2. Safe Padding Invariant: Padded basis states (i >= K) ALWAYS evaluate to f(x) = 0 (+1 phase).
3. Query Auditability: Every oracle invocation on a register or individual state is tracked.
4. Independent Testability: The oracle can be verified on individual basis states and
   state vectors completely decoupled from Grover search iterations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aesw.quantum.mapping import CandidateRegister
from aesw.quantum.marking import MarkingRule
from aesw.quantum.types import NodeId


class QuantumOracle:
    """Quantum oracle evaluating candidate marking rules and applying phase flips.

    Attributes:
        register: CandidateRegister managing index to node ID mapping.
        marking_rule: MarkingRule defining candidate eligibility.
        marked_indices: Tuple of candidate indices satisfying the marking condition.
        marked_node_ids: Tuple of original node IDs satisfying the marking condition.
        call_count: Number of oracle operations executed.
    """

    def __init__(
        self,
        register: CandidateRegister,
        marking_rule: MarkingRule,
    ) -> None:
        """Initialize the QuantumOracle.

        Args:
            register: CandidateRegister defining candidate space.
            marking_rule: Configured marking rule.
        """
        if not isinstance(register, CandidateRegister):
            raise TypeError(f"register must be a CandidateRegister, got {type(register).__name__}")
        if not isinstance(marking_rule, MarkingRule):
            raise TypeError(f"marking_rule must be a MarkingRule, got {type(marking_rule).__name__}")

        self._register = register
        self._marking_rule = marking_rule
        self._call_count = 0

        # Precompute marked set for fast, deterministic evaluation
        marked_idx: list[int] = []
        marked_nodes: list[NodeId] = []

        for i in range(self._register.num_candidates):
            candidate = self._register.get_candidate(i)
            if self._marking_rule.is_marked(candidate):
                marked_idx.append(i)
                marked_nodes.append(candidate.node_id)

        self._marked_indices: tuple[int, ...] = tuple(marked_idx)
        self._marked_node_ids: tuple[NodeId, ...] = tuple(marked_nodes)
        self._marked_set: set[int] = set(marked_idx)

    @property
    def register(self) -> CandidateRegister:
        """Candidate register."""
        return self._register

    @property
    def marking_rule(self) -> MarkingRule:
        """Marking rule."""
        return self._marking_rule

    @property
    def call_count(self) -> int:
        """Cumulative oracle call count."""
        return self._call_count

    @property
    def marked_indices(self) -> tuple[int, ...]:
        """Indices of all marked physical candidates."""
        return self._marked_indices

    @property
    def marked_node_ids(self) -> tuple[NodeId, ...]:
        """Node IDs of all marked physical candidates."""
        return self._marked_node_ids

    @property
    def num_marked(self) -> int:
        """Count of marked candidates |M|."""
        return len(self._marked_indices)

    @property
    def has_marked(self) -> bool:
        """Whether at least one candidate is marked."""
        return len(self._marked_indices) > 0

    def is_marked(self, index: int) -> bool:
        """Query whether basis state index satisfies the marking predicate.

        Args:
            index: Computational basis state index in [0, dimension - 1].

        Returns:
            True if candidate at index is marked, False otherwise.

        Raises:
            IndexError: If index is outside [0, dimension - 1].
        """
        if self._register.dimension == 0:
            raise IndexError("Cannot query oracle on an empty candidate register")
        if not (0 <= index < self._register.dimension):
            raise IndexError(
                f"Oracle query index {index} out of basis bounds [0, {self._register.dimension - 1}]"
            )

        self._call_count += 1

        # Padded states (K <= index < dimension) are never marked
        if self._register.is_padded_index(index):
            return False

        return index in self._marked_set

    def evaluate_phase(self, index: int) -> float:
        """Return the phase multiplier for a basis state (-1.0 if marked, +1.0 if unmarked).

        Args:
            index: Basis state index.

        Returns:
            -1.0 if marked, +1.0 otherwise.
        """
        return -1.0 if self.is_marked(index) else 1.0

    def apply_phase_flip(self, state_vector: np.ndarray) -> np.ndarray:
        """Apply the quantum oracle phase-flip operator O to a statevector.

        Computes: O |psi> = sum_{x} (-1)^{f(x)} psi(x) |x>.

        Args:
            state_vector: 1D complex or real NumPy array of shape (dimension,).

        Returns:
            Phase-flipped state vector of identical dimension.

        Raises:
            ValueError: If state_vector dimension does not match register dimension.
        """
        dim = self._register.dimension
        if len(state_vector) != dim:
            raise ValueError(
                f"State vector dimension {len(state_vector)} does not match oracle dimension {dim}"
            )

        # Counting one quantum oracle gate query over the superposition
        self._call_count += 1

        if not self._marked_indices:
            # If no states are marked, O = I (identity operator)
            return state_vector.copy()

        # Invert amplitude signs for all marked states
        flipped = state_vector.copy()
        for idx in self._marked_indices:
            flipped[idx] = -flipped[idx]

        return flipped

    def reset_call_count(self) -> None:
        """Reset the oracle query counter to 0."""
        self._call_count = 0

    def to_dict(self) -> dict[str, Any]:
        """Diagnostic summary of the oracle."""
        return {
            "num_marked": self.num_marked,
            "marked_indices": list(self._marked_indices),
            "marked_node_ids": list(self._marked_node_ids),
            "call_count": self._call_count,
            "register_dimension": self._register.dimension,
            "num_candidates": self._register.num_candidates,
        }
