"""
Grover Search Engine and Statevector Simulator
==============================================
Implements exact statevector evolution for Grover's quantum search algorithm.

Mathematical Formulation:
1. Register Dimension:
   For K physical candidates, n = ceil(log2(K)), N = 2^n.
2. State Preparation (Uniform Superposition):
   |s> = (1 / sqrt(N)) sum_{x=0}^{N-1} |x>
3. Oracle Operation:
   O |x> = (-1)^{f(x)} |x>
4. Diffusion Operator (Inversion about the mean):
   D = 2|s><s| - I
   For any amplitude vector v, (D v)_i = 2 * mean(v) - v_i.
5. Optimal Grover Iterations:
   When |M| > 0:
       theta = arcsin(sqrt(|M| / N))
       R = round(pi / (4 * theta) - 0.5)  (bounded >= 1)
   When |M| == 0:
       R = 0 (no rotation possible; avoids false discovery)
6. Projective Measurement:
   Simulates measurement by sampling from probability distribution P(x) = |psi(x)|^2
   driven by an isolated NumPy PRNG stream.

Architectural Guarantees:
1. Zero Data Leakage: Simulator does not inspect graph topology or bypass quantum mechanics.
2. Safe Padding Handling: Resamples or rejects measurements landing on padded states (i >= K);
   never emits phantom node IDs.
3. No-Marked-Candidate Safety: Safely reports failure status rather than asserting a false hit.
"""

from __future__ import annotations

import math

import numpy as np

from aesw.quantum.mapping import CandidateRegister
from aesw.quantum.marking import MarkingRule
from aesw.quantum.oracle import QuantumOracle
from aesw.quantum.types import (
    QuantumSearchConfig,
    QuantumSearchResult,
    QuantumSearchStatus,
)


def compute_optimal_grover_iterations(num_items: int, num_marked: int) -> int:
    """Calculate the theoretical optimal number of Grover iterations R.

    Args:
        num_items: Basis dimension N = 2^n.
        num_marked: Count of target states |M|.

    Returns:
        Optimal integer iteration count R >= 0.
    """
    if num_marked <= 0 or num_items <= 0:
        return 0
    if num_marked >= num_items:
        return 0

    ratio = num_marked / num_items
    if ratio >= 1.0:
        return 0

    theta = math.asin(math.sqrt(ratio))
    if theta <= 0.0:
        return 0

    iterations = round((math.pi / (4.0 * theta)) - 0.5)
    return max(1, iterations)


class GroverSearchEngine:
    """Statevector quantum simulator executing Grover's search algorithm.

    Operates on a CandidateRegister through a QuantumOracle.
    """

    def __init__(self, config: QuantumSearchConfig | None = None) -> None:
        """Initialize the Grover engine with configuration settings."""
        self._config = config or QuantumSearchConfig()

    @property
    def config(self) -> QuantumSearchConfig:
        """Search engine configuration."""
        return self._config

    def create_uniform_superposition(self, dimension: int) -> np.ndarray:
        """Prepare the initial state |s> = (1 / sqrt(N)) sum |x>.

        Args:
            dimension: Basis state count N = 2^n.

        Returns:
            1D float64 NumPy array representing normalized amplitudes.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        amplitude = 1.0 / math.sqrt(dimension)
        return np.full(dimension, amplitude, dtype=np.float64)

    def apply_diffusion(self, state_vector: np.ndarray) -> np.ndarray:
        """Apply the Grover diffusion operator D = 2|s><s| - I (inversion about mean).

        Computes (D v)_i = 2 * mean(v) - v_i in O(N) time with strict numerical stability.

        Args:
            state_vector: State amplitude vector of dimension N.

        Returns:
            Diffused state vector of identical dimension.
        """
        mean_amplitude = np.mean(state_vector)
        return 2.0 * mean_amplitude - state_vector

    def search(
        self,
        register: CandidateRegister,
        marking_rule: MarkingRule,
        config: QuantumSearchConfig | None = None,
    ) -> QuantumSearchResult:
        """Execute Grover search on the provided candidate register.

        Args:
            register: CandidateRegister containing physical candidates.
            marking_rule: Target-marking rule to configure the oracle.
            config: Optional search configuration override.

        Returns:
            Strongly typed, structured QuantumSearchResult.
        """
        cfg = config or self._config
        rng = np.random.default_rng(cfg.seed)

        # 1. Edge Case: Empty candidate set
        if register.is_empty:
            return QuantumSearchResult(
                selected_candidate=None,
                selected_index=None,
                marked_candidates=(),
                marked_indices=(),
                iteration_count=0,
                oracle_call_count=0,
                success=False,
                status=QuantumSearchStatus.EMPTY_CANDIDATES,
                probabilities={},
                diagnostic_metadata={"reason": "Candidate register is empty"},
            )

        # 2. Instantiate and evaluate Oracle
        oracle = QuantumOracle(register=register, marking_rule=marking_rule)
        dim = register.dimension
        marked_indices = oracle.marked_indices
        marked_nodes = oracle.marked_node_ids
        num_marked = oracle.num_marked

        # 3. Edge Case: No candidates satisfy the marking rule
        if num_marked == 0:
            # When no target exists, Grover iterations would only cause spurious phase rotations
            return QuantumSearchResult(
                selected_candidate=None,
                selected_index=None,
                marked_candidates=(),
                marked_indices=(),
                iteration_count=0,
                oracle_call_count=oracle.call_count,
                success=False,
                status=QuantumSearchStatus.NO_MARKED_CANDIDATES,
                probabilities={},
                diagnostic_metadata={
                    "num_candidates": register.num_candidates,
                    "dimension": dim,
                    "num_marked": 0,
                    "reason": "No candidates satisfied the marking condition",
                },
            )

        # 4. Prepare initial uniform superposition |s>
        state = self.create_uniform_superposition(dim)

        # 5. Determine iteration count R
        if cfg.max_iterations is not None:
            r_target = cfg.max_iterations
        else:
            r_target = compute_optimal_grover_iterations(dim, num_marked)

        # 6. Execute Grover iterations: G^R |s> = (D * O)^R |s>
        actual_iterations = 0
        for _ in range(r_target):
            # Apply Oracle: phase flip marked states
            state = oracle.apply_phase_flip(state)
            # Apply Diffusion: inversion about the mean
            state = self.apply_diffusion(state)
            actual_iterations += 1

        # 7. Compute measurement probability distribution P(x) = |psi(x)|^2
        probabilities = np.square(state)
        # Numerical normalization safeguard
        prob_sum = float(np.sum(probabilities))
        if prob_sum > 0.0:
            probabilities = probabilities / prob_sum

        prob_dict = {i: float(probabilities[i]) for i in range(dim)}

        # 8. Projective Measurement Simulation
        sampled_index: int | None = None
        retries = 0
        max_retries = cfg.max_measurement_retries if cfg.allow_padding_retry else 1

        while retries < max_retries:
            idx = int(rng.choice(dim, p=probabilities))
            if register.is_valid_index(idx):
                sampled_index = idx
                break
            # Sampled a padded dummy state (idx >= K)
            retries += 1

        # 9. Result Validation & Epistemic Boundary Enforcement
        if sampled_index is None:
            # Fallback when all resamples landed in padded states
            return QuantumSearchResult(
                selected_candidate=None,
                selected_index=None,
                marked_candidates=marked_nodes,
                marked_indices=marked_indices,
                iteration_count=actual_iterations,
                oracle_call_count=oracle.call_count,
                success=False,
                status=QuantumSearchStatus.MEASUREMENT_PADDED_INDEX,
                probabilities=prob_dict,
                diagnostic_metadata={
                    "retries": retries,
                    "reason": "Projective measurement sampled padded ancillary basis states",
                    "num_candidates": register.num_candidates,
                    "dimension": dim,
                },
            )

        selected_node = register.get_node_id(sampled_index)
        is_hit = sampled_index in marked_indices

        status = QuantumSearchStatus.SUCCESS if is_hit else QuantumSearchStatus.MEASUREMENT_NON_TARGET

        return QuantumSearchResult(
            selected_candidate=selected_node,
            selected_index=sampled_index,
            marked_candidates=marked_nodes,
            marked_indices=marked_indices,
            iteration_count=actual_iterations,
            oracle_call_count=oracle.call_count,
            success=is_hit,
            status=status,
            probabilities=prob_dict,
            diagnostic_metadata={
                "num_candidates": register.num_candidates,
                "dimension": dim,
                "num_qubits": register.num_qubits,
                "num_padded": register.num_padded,
                "num_marked": num_marked,
                "theoretical_optimal_iterations": compute_optimal_grover_iterations(dim, num_marked),
                "actual_iterations": actual_iterations,
                "measured_probability": prob_dict.get(sampled_index, 0.0),
                "total_marked_probability": sum(prob_dict.get(i, 0.0) for i in marked_indices),
            },
        )
