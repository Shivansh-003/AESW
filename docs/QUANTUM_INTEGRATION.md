# Quantum Integration Foundation (Milestone 1)

**Package:** `aesw.quantum`  
**Status:** Implemented (Milestone 1 Complete)  
**Tests:** `tests/test_quantum_foundation.py` (26 automated unit & scientific tests passing)  

---

## 1. Executive Summary & Epistemic Scope

Milestone 1 establishes the mathematical, data model, and algorithmic foundation for integrating quantum search with the classical Adaptive Evidence-Sharing Walkers (AESW) system.

### Scope Boundaries
- **Simulator vs. Real Hardware**: All quantum operations in Milestone 1 are executed via exact classical statevector simulation ($N = 2^n$ complex/real amplitude evolution). No execution on physical quantum processing units (QPUs) is performed.
- **No Speedup Claim**: Simulating $N$-dimensional statevectors on classical CPUs scales exponentially in memory and time ($O(2^n)$). No practical quantum speedup is claimed or achieved in simulation.
- **Milestone 1 vs. Milestone 2**: Milestone 1 defines the candidate interface, deterministic index mapping, eligibility predicates, phase-flip oracle, and statevector Grover engine. It does **not** alter the live classical AESW search or coordinator jump execution. That integration is reserved for Milestone 2.

---

## 2. Quantum Candidate Interface

To prevent graph leakage and unnecessary overhead, the interface strictly decouples classical graph representations from the quantum register:

### 2.1 Data Models (`aesw.quantum.types`)
- **`NodeId`**: `int | str` representing original graph vertex identifiers.
- **`QuantumCandidate`**:
  ```python
  @dataclass(frozen=True)
  class QuantumCandidate:
      node_id: NodeId
      score: float | None = None
      evidence: NodeEvidence | WeightedEvidence | CandidateScore | None = None
      metadata: Mapping[str, Any] = field(default_factory=dict)
  ```
  Provides explicit factory constructors:
  - `QuantumCandidate.from_node_id(node_id)`
  - `QuantumCandidate.from_candidate_score(candidate_score)`
  - `QuantumCandidate.from_weighted_evidence(weighted_evidence)`
  - `QuantumCandidate.from_node_evidence(node_evidence)`
- **`QuantumSearchConfig`**:
  - `seed: int | None`: Isolated PRNG seed for deterministic projective measurement.
  - `max_iterations: int | None`: Grover iteration override (or auto-computed theoretical optimal).
  - `target_threshold: float | None`: Numerical threshold for score-based marking.
  - `score_field: str`: Attribute evaluated by the threshold rule.
  - `allow_padding_retry: bool`: Enables resampling if measurement lands on an ancillary padded state.
- **`QuantumSearchResult`**:
  - `selected_candidate: NodeId | None`: Chosen candidate vertex ID.
  - `selected_index: int | None`: Index in $[0, K-1]$ in the candidate register.
  - `marked_candidates: tuple[NodeId, ...]`: Physical candidate IDs satisfying the marking rule.
  - `marked_indices: tuple[int, ...]`: Register indices of marked candidates.
  - `iteration_count: int`: Number of Grover iterations executed.
  - `oracle_call_count: int`: Number of oracle query operations evaluated.
  - `success: bool`: True if and only if a valid marked candidate was measured.
  - `status: QuantumSearchStatus`: Discrete status code (`SUCCESS`, `NO_MARKED_CANDIDATES`, `EMPTY_CANDIDATES`, `MEASUREMENT_NON_TARGET`, `MEASUREMENT_PADDED_INDEX`, `ERROR`).
  - `probabilities: Mapping[int, float]`: Basis state probability distribution $|\psi(x)|^2$.
  - `diagnostic_metadata: Mapping[str, Any]`: Quantum register diagnostics.

---

## 3. Target-Marking Rules & Eligibility Predicates

Grover search requires a boolean oracle function $f(x) \in \{0, 1\}$. The marking rule defines which candidate states receive a phase flip.

### 3.1 Separation of Concerns
Marking rules evaluate candidate metadata independently of the quantum simulation:
```text
Classical Candidate (Score, Evidence) ──► MarkingRule.is_marked() ──► QuantumOracle (Phase Flip)
```

### 3.2 Implemented Rules (`aesw.quantum.marking`)
1. **`ThresholdMarkingRule(threshold, score_field, strict)`**:
   Marks candidates whose primary score or metadata attribute $\ge \text{threshold}$.
   Reuses existing AESW evidence fields (`effective_confidence`, `positive_evidence`, `signal_strength`, `total_score`).
2. **`PredicateMarkingRule(predicate, name)`**:
   Evaluates an arbitrary user-defined callable `Callable[[QuantumCandidate], bool]`.
3. **`TargetIdMarkingRule(target_ids)`**:
   Marks candidates matching a known set of node IDs (used for ground-truth oracle testing).

### 3.3 Epistemic Integrity
- High evidence scores indicate promising exploration targets, not ground-truth certainty.
- When no candidates satisfy the marking rule ($|M| = 0$), the engine does not execute blind rotations and explicitly returns `QuantumSearchStatus.NO_MARKED_CANDIDATES` with `success=False`.

---

## 4. Candidate Preparation, Index Mapping & Padding

### 4.1 Register Sizing (`aesw.quantum.mapping.CandidateRegister`)
Given $K$ physical candidates:
- Qubit count: $n = \lceil \log_2(K) \rceil$ (with $n \ge 1$ for $K \ge 1$).
- Basis dimension: $N = 2^n$.
- Padded states: $N - K$ basis states.

### 4.2 Non-Power-of-Two Padding Strategy
When $K$ is not an exact power of two (e.g. $K=3 \implies N=4$; $K=5 \implies N=8$):
- Basis states $0 \le i < K$ represent valid physical graph vertices.
- Basis states $K \le i < N$ represent padded ancillary states.
- **Oracle Invariant**: The oracle **never** marks padded states ($f(x) = 0$ for $x \ge K$).
- **Reverse Mapping Invariant**: `reverse_map(i)` returns `None` for $i \ge K$.
- **Measurement Resampling**: If projective measurement samples a padded state $i \ge K$, the simulator resamples up to `max_measurement_retries` times. If all retries land in padded states, the search fails safely with `MEASUREMENT_PADDED_INDEX` rather than returning a phantom node.

---

## 5. Quantum Oracle & Grover Search Engine

### 5.1 Oracle Operator (`aesw.quantum.oracle.QuantumOracle`)
Implements the phase-flip operator:
$$O |x\rangle = (-1)^{f(x)} |x\rangle$$
- Evaluates $f(x)$ against the configured `MarkingRule`.
- Tracks cumulative oracle calls (`call_count`).
- Flips amplitudes of marked basis states: $\psi_w \gets -\psi_w$ for all $w \in M$.

### 5.2 Grover Diffusion Operator
Implements the inversion-about-the-mean operator $D = 2|s\rangle\langle s| - I$:
$$(D \cdot v)_i = 2 \cdot \text{mean}(v) - v_i$$
This is computed in $O(N)$ operations with numerical stability.

### 5.3 Optimal Grover Iteration Calculation
When $|M| > 0$ targets exist in dimension $N$:
$$\theta = \arcsin\left(\sqrt{\frac{|M|}{N}}\right), \quad R = \text{round}\left(\frac{\pi}{4\theta} - \frac{1}{2}\right)$$
When $|M| = 0$, $R = 0$.

### 5.4 Projective Measurement
Simulates measurement by sampling from categorical distribution $P(x) = |\psi(x)|^2$ using an isolated NumPy PRNG stream initialized with `config.seed`.

---

## 6. How to Run Tests and Simulation

### Running Milestone 1 Quantum Tests
```bash
pytest tests/test_quantum_foundation.py -v
```

### Running Full Verification Suite
```bash
pytest
```

### Example Usage
```python
from aesw.quantum import (
    CandidateRegister,
    QuantumCandidate,
    ThresholdMarkingRule,
    GroverSearchEngine,
    QuantumSearchConfig,
)

# 1. Define candidate nodes from classical evidence
candidates = [
    QuantumCandidate(node_id="node_A", score=0.2),
    QuantumCandidate(node_id="node_B", score=0.85),  # Marked target
    QuantumCandidate(node_id="node_C", score=0.4),
]

# 2. Build candidate register (K = 3 -> n = 2 qubits, N = 4 basis states)
register = CandidateRegister(candidates)

# 3. Define target-marking threshold
rule = ThresholdMarkingRule(threshold=0.6)

# 4. Execute Grover search simulation
engine = GroverSearchEngine(QuantumSearchConfig(seed=42))
result = engine.search(register, rule)

assert result.success is True
assert result.selected_candidate == "node_B"
print(f"Chosen candidate: {result.selected_candidate} with status {result.status.value}")
```

---

## 7. Limitations & Incomplete Elements (Milestone 2 Roadmap)

The following capabilities are explicitly deferred to Milestone 2:
1. **Live AESW Long-Jump Integration**: Wiring `GroverSearchEngine` into `AESWAgent.step()` and `SimulationCoordinator.step()`.
2. **Dynamic Grover Long-Jump Policy**: Selecting between uniform random jumps and Grover candidate selection based on memory confidence and churn.
3. **Comparative Quantum Benchmarks**: Running statistical experiment sweeps comparing classical AESW vs. hybrid Quantum-AESW on dynamic benchmark topologies.
4. **Qiskit Circuit Exporter**: Optional translation of the statevector simulation to executable Qiskit QuantumCircuit representations for QASM backends.
