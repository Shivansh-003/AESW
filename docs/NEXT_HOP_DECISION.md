# AESW Next-Hop Decision Engine

---

## 1. Overview & Research Objective

In dynamic graph search and transport under uncertainty, moving agents (walkers) must continually select their next traversal hop under **partial visibility**, **time-varying edge availability**, and **imperfect sensing**.

A naive greedy search that strictly follows target clues risks entrapment in local topological bottlenecks or following decayed, stale traces. Conversely, purely stochastic walks (such as random walk or degree-based walks) suffer from high hitting times and fail to exploit valuable sensory clues.

The **Next-Hop Decision Engine** of the **Adaptive Evidence-Sharing Walkers (AESW)** framework resolves this fundamental exploration–exploitation tension by formulating candidate neighbor selection as a principled, multi-criteria scoring optimization coupled with numerically stabilized softmax stochastic action selection:

```text
       Candidate Active Neighbors u ∈ N_checked(v)
                           │
                           ▼
          Multi-Factor Scoring Engine:
     S(u) = a·P(u) + b·N(u) - g·R(u) - d·D(u)
                           │
                           ▼
          Numerically Stabilized Softmax:
     P_select(u) = exp((S(u) - S_max) / T) / Z
                           │
                           ▼
               Isolated Stochastic Sampling
                           │
                           ▼
             Explainable NextHopDecision
```

---

## 2. Mathematical Formulation

### 2.1 Candidate Score Function

For each legal candidate active neighbor $u \in \mathcal{N}_{\text{active}}(v)$ incident to the walker's current node $v$, the engine computes the scalar score:

$$S(u) = a \cdot P(u) + b \cdot N(u) - g \cdot R(u) - d \cdot D(u)$$

where $a, b, g, d \ge 0$ are non-negative weighting parameters configured by the experiment or tuned by the mode controller.

#### 1. Positive Evidence Strength $P(u) \in [0.0, 1.0]$
Represents target presence information stored in the walker's local M8 `EvidenceCache` for vertex $u$, weighted by M9 estimated graph churn $\hat{\lambda}$:

$$P(u) = \begin{cases} c_{\text{eff}}(u, t) = c_0(u) \cdot \exp(-\hat{\lambda} \cdot (t - t_{\text{obs}})) & \text{if } \text{Polarity}(u) = \text{POSITIVE} \\ 0.0 & \text{otherwise} \end{cases}$$

- Unvisited nodes or cleared nodes (`NEGATIVE` polarity) contribute $P(u) = 0.0$.
- In highly volatile graphs (large $\hat{\lambda}$), older clues decay rapidly toward $0.0$, preventing pursuit of stale "ghost" traces.

#### 2. Local Novelty $N(u) \in (0.0, 1.0]$
Quantifies the intrinsic information value of exploring unvisited or rarely visited vertices:

$$N(u) = \frac{1.0}{1.0 + \text{visit\_count}(u)}$$

- Unvisited vertices ($\text{visit\_count} = 0$) receive maximum novelty $N(u) = 1.0$.
- Frequently visited vertices have novelty approaching $0.0$.

#### 3. Revisit Penalty $R(u) \ge 0.0$
Explicitly penalizes backtracking and local cycles:

$$R(u) = \text{float}(\text{visit\_count}(u))$$

- Disincentivizes cyclic re-traversal between adjacent nodes unless compensated by overwhelming positive evidence ($a \cdot P(u)$).

#### 4. Traversal Delay / Cost $D(u) \ge 1.0$
Reflects locally observable traversal duration or edge transit latency:

$$D(u) = \max(1.0, \text{delay}(v, u))$$

- Enforces edge delay constraints from the formal problem definition ($\ge 1.0$).

---

### 2.2 Numerically Stabilized Softmax Action Selection

Given scores $\{S(u)\}_{u \in \mathcal{C}}$, candidate selection probabilities are computed via the Boltzmann (softmax) distribution parameterized by exploration temperature $T > 0$:

$$P_{\text{select}}(u) = \frac{\exp\left(\frac{S(u) - \max_{w \in \mathcal{C}} S(w)}{T}\right)}{\sum_{k \in \mathcal{C}} \exp\left(\frac{S(k) - \max_{w \in \mathcal{C}} S(w)}{T}\right)}$$

#### Temperature Effects:
- **$T \to 0$ (Greedy / Exploitative)**: Selection probability concentrates sharply on $\arg\max_u S(u)$.
- **$T \to \infty$ (Uniform / Explorative)**: Selection probability approaches the uniform distribution $1 / |\mathcal{C}|$.
- **$T = 1.0$ (Balanced)**: Standard stochastic action selection matching relative score differentials.

#### Numerical Stability Guarantee:
Subtracting $S_{\max} = \max_w S(w)$ guarantees that the exponent numerator is always $\le 0$, ensuring $\exp((S(u) - S_{\max})/T) \in (0, 1]$. This completely eliminates floating-point overflow (`math.inf` or `NaN`) even when scores reach extreme values ($S(u) > 10^4$).

---

## 3. Architectural Guarantees & Epistemic Boundaries

1. **Strict Epistemic Firewall**:
   - The decision engine operates **exclusively** on candidates extracted from `observation.checked_neighbors` where `observation.is_edge_known_active(u)` is `True`.
   - Inactive dynamic links (`EdgeState.OFF`) are pruned before scoring.
   - The engine has **zero access** to the ground-truth graph, transition probabilities ($p_{\text{on}}, p_{\text{off}}$), hidden edge states, or true target coordinates.

2. **Decoupled Responsibilities**:
   - The engine computes scores, probabilities, and constructs a formatted `SearchAction(MOVE)` or `SearchAction(STAY)`.
   - The engine **never** modifies walker coordinates, mutates simulation time, or executes physical steps. Movement execution remains the sole responsibility of the environment runner / walker agent.

3. **RNG Isolation**:
   - Stochastic candidate sampling consumes an explicitly supplied `numpy.random.Generator` instance (`rng`).
   - Global Python `random` and NumPy global state are completely unaffected, maintaining strict experimental reproducibility across Monte Carlo runs.

4. **Explainable Output**:
   - Every decision emits a structured `NextHopDecision` containing:
     - `CandidateScore` breakdowns ($P, N, R, D, S$) for every evaluated candidate.
     - Full probability vector $\{P_{\text{select}}(u)\}$.
     - Configured `mode`, temperature, and estimated churn rate $\hat{\lambda}$.

---

## 4. Subsystem Integration Flow (M8–M12)

The Next-Hop Decision Engine integrates seamlessly into the end-to-end AESW processing pipeline:

```text
  Observation Snapshot (M5)
            │
            ├──► Churn Estimator (M9) ──► Computes λ_hat
            │                                  │
            ├──► Evidence Memory (M8) ◄────────┘ (Decays evidence weights)
            │         │
            ├──► Node Exchange (M10) ──► Pushes/Pulls shared evidence
            │         │
            ├──► Mode Controller (M11) ─► Determines SearchMode (LOCAL vs LONG_JUMP)
            │                                  │
            ▼                                  ▼
     Next-Hop Decision Engine (M12) ◄──────────┘
            │
            ▼
     SearchAction ready for environment execution
```

---

## 5. Edge Case Handling

| Edge Case | Behavior | Formal Guarantee |
| :--- | :--- | :--- |
| **Empty Candidates ($|\mathcal{C}| = 0$)** | Returns `SearchAction(ActionType.STAY)`, `selected_node = None` | Walker remains in place when all incident links are `OFF` or node is isolated. |
| **Single Candidate ($|\mathcal{C}| = 1$)** | Returns `SearchAction(ActionType.MOVE)`, $P(u) = 1.0$ | Deterministic selection without unnecessary RNG invocation. |
| **Identical Scores** | Uniform distribution $P(u) = 1 / |\mathcal{C}|$ | Unbiased symmetry across equivalent choices. |
| **Extreme Large Scores ($S \ge 10^4$)** | Stable softmax via $S(u) - S_{\max}$ | No overflow, probabilities sum to $1.0 \pm 10^{-6}$. |
| **Extreme Negative Scores ($S \le -10^4$)** | Stable softmax via $S(u) - S_{\max}$ | No underflow to NaN, preserves monotonic preference. |
| **LONG_JUMP Mode Scaling** | Optional boost to novelty ($1.5 \times b$) and dampening of revisit penalty ($0.5 \times g$) | Facilitates dispersion across previously traversed boundaries. |

---

## 6. Verification & Test Suite Summary

The Next-Hop Decision Engine is fully verified by `tests/test_next_hop_decision.py` comprising **53 unit and integration tests**:

- **Scorer Weights & Validation (7 tests)**: Default weights, custom weights, rejection of negative weights.
- **Positive Evidence Evaluation (6 tests)**: Cache lookup, missing entries, negative evidence suppression, exponential time-churn decay.
- **Novelty & Revisit Penalty (7 tests)**: Monotonicity in visit counts, exact functional values, boundary clamping.
- **Delay Evaluation (3 tests)**: Default unit delays, custom delay lookup, $\ge 1.0$ floor enforcement.
- **Total Score Arithmetic & Dataclasses (3 tests)**: Exact arithmetic $S = aP + bN - gR - dD$, serialization to dictionary.
- **Softmax Numerical Stability (10 tests)**: $T \le 0$ rejection, asymptotic behavior ($T \to 0$ greedy, $T \to \infty$ uniform), extreme score overflow/underflow protection.
- **Stochastic Sampling & RNG Isolation (5 tests)**: Empirical probability convergence, seed reproducibility, global RNG isolation.
- **Decision Engine & Observation Integration (6 tests)**: Extraction of legal candidates, active link filtering, STAY on isolation, explainable metadata.
- **Mode Weight Scaling (1 test)**: Verification of exploration boost under LONG_JUMP.
- **Scientific Scenarios (3 tests)**:
  - *Scenario A*: Fresh positive target evidence attracts walker over unvisited alternative.
  - *Scenario B*: High churn decays stale clues, allowing fresh novelty to dominate.
  - *Scenario C*: Revisit penalty strongly avoids immediate backtracking.
- **Subsystem Integration (1 test)**: End-to-end data pipeline coupling M8 + M9 + M10 + M11 + M12.

Full regression status across the entire codebase: **320 tests passed, 0 failures**.
