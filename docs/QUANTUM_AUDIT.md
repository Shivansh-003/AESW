# Repository Audit: Classical AESW & Quantum Integration Foundation

**Project:** Adaptive Search and Transport on Dynamic Graphs Under Uncertainty (AESW)  
**Milestone:** 1 — Quantum Integration Foundation  
**Date:** October 2026  
**Status:** Audit Completed  

---

## 1. Executive Summary

This audit assesses the current state of the Adaptive Evidence-Sharing Walkers (AESW) repository to prepare for Milestone 1: **Quantum Integration Foundation**. The overarching goal of the project is to evaluate hybrid classical-quantum search on dynamic graphs under uncertainty. Specifically, classical AESW explores local neighborhoods and builds evidence memory, while a quantum-search mechanism (Grover search) can accelerate global candidate selection during long-jump dispersion phases.

In accordance with Milestone 1 directives:
1. We audited all existing classical search, evidence scoring, and jump decision components.
2. We audited the repository for existing quantum components and dependencies.
3. We established the interface boundary and data contracts required to bridge classical candidate data with a Grover search simulator.
4. We preserve classical AESW operational integrity without altering live simulation behavior in Milestone 1.

---

## 2. Repository Audit Findings

### 2.1 Existing Classical Architecture
The repository contains a fully implemented, highly verified classical simulation framework:
- **`src/aesw/graph/`**: Deterministic generators for Erdős–Rényi, Barabási–Albert, Watts–Strogatz, 2D Grid with obstacles, and Random Geometric Graphs (`GraphInstance`).
- **`src/aesw/dynamics/`**: Discrete-time Markovian edge transitions ($p_{\text{on}}, p_{\text{off}}$) with synchronous state updates and named churn regimes (`STATIC` to `VERY_FAST`).
- **`src/aesw/environment/`**:
  - Problem specification (`ProblemDefinition`), ground truth state (`GroundTruthState`), and walker observation models (`Observation`).
  - Target locomotion engine (`TargetEngine`) and noisy sensing engine (`DetectionEngine`).
  - Central discrete-time simulation coordinator (`SimulationCoordinator`).
- **`src/aesw/baselines/`**: Six benchmark search policies (Random Walk, Non-Backtracking, $k$-Random Walkers, Degree-Based, Flooding, Ant Colony).
- **`src/aesw/memory/`**: Evidence data models (`NodeEvidence`, `WeightedEvidence`) and cache (`EvidenceCache`) with dynamic exponential churn decay:
  $$w(\text{age}) = \exp(-\hat{\lambda} \cdot \text{age})$$
- **`src/aesw/aesw/`**: Complete AESW agent suite:
  - `ChurnEstimator`: Online Jaccard-based dynamic churn estimation $\hat{\lambda}$.
  - `NodeMediatedExchange`: Decentralized vertex-anchored communication (`PUSH`, `PULL`, `PUSH_PULL`).
  - `AdaptiveModeController`: Information-driven mode switching between `LOCAL` and `LONG_JUMP` based on sliding window $W$ stagnation.
  - `NextHopDecisionEngine`: Explainable candidate scoring ($S(u) = \alpha P(u) + \beta N(u) - \gamma R(u) - \delta D(u)$) and softmax selection.
  - `AESWAgent` & `AESWPolicy`: Multi-walker coordination and harness integration.
- **`src/aesw/evaluation/`**: Benchmark harness (`run_experiment`, `run_benchmark`) and metrics (`RunMetrics`, `AggregatedMetrics`).
- **Test Suite**: 365 automated tests across 16 test modules passing with 100% success rate.

### 2.2 Quantum Codebase Audit
A thorough inspection of the repository was performed:
- **`quantum/` directory**: No `quantum/` directory was present in the repository root or submodules.
- **Qiskit dependencies**: `qiskit` is not declared in `requirements.txt` or `pyproject.toml`, and is not installed in the Python 3.13 virtual environment.
- **Git history**: No previous quantum circuits, oracles, or quantum walk implementations were ever committed to the repository history (only a placeholder check `create_baseline("quantum_walk")` in `tests/test_baselines.py` asserting that unknown baseline types raise `ValueError`).
- **Adaptation Strategy**:
  As mandated by project instructions (*"If the repository differs from the assumptions above, adapt to the actual code rather than blindly following the proposed file structure"*), we create a dedicated package `src/aesw/quantum/`.
  To ensure stability, deterministic reproducibility, and zero dependency conflicts on Python 3.13, we implement a pure scientific Python (NumPy) quantum statevector simulator for Grover's algorithm alongside modular abstractions for future Qiskit circuit compilation.

---

## 3. Inspection of Candidate Selection & Long-Jump Decision Paths

### 3.1 Candidate Selection & Scoring (`aesw.aesw.next_hop`)
In classical AESW, local next-hop candidate selection evaluates 1-hop neighbors from the partial observation:
$$S(u) = a \cdot P(u) + b \cdot N(u) - g \cdot R(u) - d \cdot D(u)$$
where:
- $P(u)$: Decayed positive evidence strength from `EvidenceCache` (bounded in $[0.0, 1.0]$).
- $N(u)$: Novelty metric $\frac{1}{1 + \text{visit\_count}(u)}$.
- $R(u)$: Revisit penalty $\text{visit\_count}(u)$.
- $D(u)$: Traversal delay cost $\ge 1.0$.

Each candidate is encapsulated in a `CandidateScore` dataclass containing `node_id`, individual score components, and `total_score`.

### 3.2 Evidence Memory (`aesw.memory`)
- `NodeEvidence`: Stores `node_id`, `visited`, `target_found`, `signal_strength` $\in [0, 1]$, `confidence` $\in [0, 1]$, `timestamp`, `polarity` (`POSITIVE` or `NEGATIVE`), and `source`.
- `WeightedEvidence`: Evaluates `NodeEvidence` at timestamp $t$ under churn estimate $\hat{\lambda}$:
  $$\text{effective\_confidence} = \text{confidence} \cdot \exp(-\hat{\lambda} \cdot (t - \text{timestamp}))$$
- `EvidenceCache.get_positive(current_time, lambda_hat, min_confidence)`: Returns all known positive evidence nodes meeting a confidence threshold.

### 3.3 Long-Jump Decision & Stagnation (`aesw.aesw.mode`, `aesw.aesw.agent`)
- `AdaptiveModeController`: Tracks useful information arrival over sliding window $W$. When consecutive stagnation steps $S(t) \ge W$, mode transitions to `SearchMode.LONG_JUMP`.
- `AESWAgent.step()`: When `mode == SearchMode.LONG_JUMP`, issues:
  ```python
  SearchAction(action_type=ActionType.JUMP, destination=None)
  ```
- `SimulationCoordinator.step()`: When `action.action_type == ActionType.JUMP`, the coordinator currently performs a uniform random jump among graph nodes:
  ```python
  dest = other_nodes[int(self._jump_rng.integers(0, len(other_nodes)))]
  ```

---

## 4. Components Reused vs. Components Created

| Category | Component | Status | Reused or Created |
| :--- | :--- | :--- | :--- |
| **Node Identifiers** | `NodeId` (`int \| str`) | Existing | Reused directly |
| **Evidence Models** | `NodeEvidence`, `WeightedEvidence` | Existing | Reused for candidate scoring |
| **Candidate Scores** | `CandidateScore` | Existing | Reused as candidate input |
| **PRNG Isolation** | `np.random.default_rng(seed)` | Existing | Reused for quantum measurement |
| **Action Contracts** | `SearchAction`, `ActionType.JUMP` | Existing | Reused for future jump execution |
| **Quantum Interface** | `QuantumCandidate`, `QuantumSearchResult`, `QuantumSearchConfig` | Missing | Created in `aesw.quantum.types` |
| **Marking Rules** | `ThresholdMarkingRule`, `PredicateMarkingRule`, `TargetIdMarkingRule` | Missing | Created in `aesw.quantum.marking` |
| **Candidate Register**| `CandidateRegister` (Index $\leftrightarrow$ NodeId mapping, Qubit sizing, Non-$2^n$ padding) | Missing | Created in `aesw.quantum.mapping` |
| **Quantum Oracle** | `QuantumOracle` (Phase flip $O = I - 2\|M\rangle\langle M\|$, Query tracking) | Missing | Created in `aesw.quantum.oracle` |
| **Grover Simulator** | `GroverSearchEngine` (Superposition, $R$ iterations, Diffusion $D$, Projective measurement) | Missing | Created in `aesw.quantum.grover` |

---

## 5. Proposed Integration Interface (Milestone 1 vs. Milestone 2)

```text
┌────────────────────────────────────────────────────────┐
│                     Classical AESW                     │
│  - EvidenceCache (Decayed positive evidence)           │
│  - NextHopDecisionEngine (Candidate scores)            │
│  - AdaptiveModeController (Stagnation trigger)         │
└───────────────────────────┬────────────────────────────┘
                            │
               Candidate Node IDs + Scores
                            │
┌───────────────────────────▼────────────────────────────┐
│          Milestone 1: Quantum Candidate Register        │
│  - CandidateRegister: NodeId <-> Index [0, K-1]        │
│  - Non-power-of-2 padding: indices [K, 2^n - 1]        │
│  - Target Marking Rule (Threshold / Predicate)         │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│          Milestone 1: Quantum Oracle & Grover          │
│  - QuantumOracle: evaluates marking rule on candidates │
│  - Padded states explicitly unmarked                   │
│  - GroverSearchEngine: exact statevector simulation    │
│  - Projective measurement + Reverse mapping            │
│  - Emits QuantumSearchResult                           │
└───────────────────────────┬────────────────────────────┘
                            │
              (Milestone 2 Future Integration)
                            │
┌───────────────────────────▼────────────────────────────┐
│      Milestone 2: Hybrid Long-Jump Execution           │
│  - Wire QuantumSearchResult into AESWAgent / Coord.    │
│  - Benchmark classical vs quantum jump performance     │
└────────────────────────────────────────────────────────┘
```

---

## 6. Audit Conclusion & Next Steps

Milestone 1 focuses exclusively on establishing a verified, isolated quantum interface:
1. Classical AESW components (`agent.py`, `coordinator.py`, `mode.py`) remain completely untouched.
2. The quantum search module operates in full isolation with comprehensive unit tests for mappings, edge cases, padding, and oracles.
3. No practical quantum speedup is claimed; the simulator faithfully models Grover state evolution and measurement statistics on classical hardware.
