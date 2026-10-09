# AESW Churn Estimator & Online Volatility Inference

---

## 1. Overview & Research Objective

In dynamic graph search and transport under uncertainty, the rate of topological change governs how quickly historical observations become obsolete. The true environment dynamics are parameterized by transition probabilities:

$$p_{\text{on}}, \quad p_{\text{off}}$$

However, in realistic decentralized and partially observable scenarios, autonomous search agents do **not** have privileged access to these ground-truth environment parameters or global graph topology.

The **Adaptive Evidence-Sharing Walkers (AESW)** framework introduces an **online, local churn estimator** that infers the apparent graph change rate $\hat{\lambda}$ strictly from partial, locally observed neighborhood availability transitions over time:

$$\text{Local Observation} \implies \text{Neighborhood Overlap } s \implies \text{Raw Churn Rate} \implies \text{Exponential Smoothing} \implies \hat{\lambda}_t \implies \text{Adaptive Memory Decay}$$

This estimated volatility $\hat{\lambda}_t$ directly modulates the exponential memory decay equation:

$$w(t) = \exp\left(-\hat{\lambda} \cdot \text{age}\right)$$

---

## 2. Mathematical Formulation

### Neighborhood Overlap Metric

At simulation time $t$, a walker at node $u$ inspects its incident edges subject to observation budget $B$. Let $N_{\text{curr}}$ be the set of locally inspected, confirmed active neighbors:

$$N_{\text{curr}} = \{v \in \text{checked\_neighbors}(u) \mid \text{state}(u, v) = \text{ON}\}$$

Comparing the current active neighborhood $N_{\text{curr}}$ with the previously observed active neighborhood $N_{\text{prev}}$ yields the Jaccard neighborhood overlap:

$$s = \frac{|N_{\text{prev}} \cap N_{\text{curr}}|}{|N_{\text{prev}} \cup N_{\text{curr}}|}$$

### Instantaneous (Raw) Churn Rate

Let $\Delta t = t_{\text{curr}} - t_{\text{prev}} > 0$ be the elapsed simulation time between consecutive observations. Under Poissonian / exponential link transition assumptions, the probability of link persistence over $\Delta t$ scales with $s \approx \exp(-\lambda_{\text{raw}} \cdot \Delta t)$.

Inverting this relationship yields the instantaneous raw churn rate:

$$\lambda_{\text{raw}} = \frac{-\ln(s)}{\Delta t}$$

### Numerical Stability & Boundary Cases

To prevent singular values ($-\ln(0) = \infty$, $\frac{0}{0}$, or $\text{NaN}$), the estimator implements explicit boundary logic:

1. **Identical Neighborhoods ($s = 1.0$)**:
   $$\lambda_{\text{raw}} = \frac{-\ln(1.0)}{\Delta t} = 0.0$$

2. **Both Neighborhoods Empty ($N_{\text{prev}} = \emptyset$ and $N_{\text{curr}} = \emptyset$)**:
   The Jaccard formula gives $\frac{0}{0}$. Because two empty observations reveal no evidence of topological change, the overlap is defined as $s = 1.0$, yielding:
   $$\lambda_{\text{raw}} = 0.0$$

3. **Disjoint / Asymmetric Transitions ($s = 0.0$)**:
   When $N_{\text{prev}} \cap N_{\text{curr}} = \emptyset$ (including empty $\to$ non-empty or non-empty $\to$ empty transitions), a strictly positive numerical stability floor $\epsilon > 0$ (default $\epsilon = 10^{-4}$) is applied:
   $$s_{\text{safe}} = \max(s, \epsilon)$$
   $$\lambda_{\text{raw}} = \frac{-\ln(s_{\text{safe}})}{\Delta t} > 0$$
   This guarantees that $\lambda_{\text{raw}}$ remains finite and strictly non-negative.

### Online Exponential Smoothing

To attenuate high-frequency observational noise, the online estimator applies exponential smoothing with learning factor $\eta \in [0.0, 1.0]$:

$$\hat{\lambda}_t = (1 - \eta) \cdot \hat{\lambda}_{t-1} + \eta \cdot \lambda_{\text{raw}}$$

- **$\eta = 1.0$**: The estimator completely trusts the newest instantaneous measurement ($\hat{\lambda}_t = \lambda_{\text{raw}}$).
- **$\eta = 0.0$**: The estimator maintains its prior belief indefinitely ($\hat{\lambda}_t = \hat{\lambda}_{t-1}$).
- **$0 < \eta < 1$**: Balances historical belief with recent evidence (default $\eta = 0.2$).

Non-negativity and finiteness are strictly enforced:

$$\hat{\lambda}_t = \max\left(0.0, \, (1 - \eta) \cdot \hat{\lambda}_{t-1} + \eta \cdot \lambda_{\text{raw}}\right)$$

---

## 3. Epistemic Boundary Principle

The churn estimator respects the strict partial visibility firewall established in the observation model:

| Information Type | Available to Estimator? | Rationale |
| :--- | :---: | :--- |
| **Locally Observed Active Links** | **YES** | Sensed by walker within observation budget $B$ |
| **Elapsed Time $\Delta t$** | **YES** | Local simulation timestamp delta |
| **Ground-Truth $p_{\text{on}}, p_{\text{off}}$** | **NO** | Hidden environment dynamics parameters |
| **Global Graph Topology $V, E$** | **NO** | Walkers only observe local degree and checked links |
| **Active Graph View $G_t$** | **NO** | Global active graph state is hidden |
| **Target State $x_t$** | **NO** | Target position is unknown to churn estimator |

---

## 4. Software Architecture & API

Located in `aesw.aesw.churn`, the `ChurnEstimator` class provides an online state machine:

```python
from aesw.aesw.churn import ChurnEstimator

# Initialize estimator with configurable smoothing factor and numerical floor
estimator = ChurnEstimator(eta=0.2, epsilon=1e-4, initial_lambda=0.0)

# First observation: initializes baseline state without manufacturing churn
lambda_0 = estimator.update({1, 2, 3}, timestamp=0)
assert lambda_0 == 0.0

# Second observation: computes overlap, raw rate, and smoothed lambda_hat
lambda_1 = estimator.update({2, 3, 4}, timestamp=1)
print(f"Overlap: {estimator.overlap:.4f}, Lambda Hat: {estimator.estimated_lambda:.4f}")
```

### Diagnostic Attributes

The estimator exposes transparent diagnostics for scientific logging and auditability:
- `overlap`: Latest Jaccard similarity $s \in [0.0, 1.0]$.
- `raw_churn_rate`: Instantaneous rate before exponential smoothing.
- `estimated_lambda` (or alias `lambda_hat`): Smoothed non-negative churn rate.
- `previous_neighborhood`, `current_neighborhood`: Immutable `frozenset` snapshots.
- `previous_timestamp`, `current_timestamp`: Simulation time tracking.
- `update_count`: Number of processed observation epochs.

### Key Methods

- `update(neighborhood, timestamp)`: Updates state from raw neighbor collections or `Observation` instances.
- `update_from_observation(obs)`: Convenience method extracting inspected ON edges from `Observation`.
- `compute_overlap(nbrs1, nbrs2)`: Static utility computing Jaccard overlap between any two collections.
- `reset(initial_lambda=None)`: Resets observation history to initial state.
- `clone()`: Creates an isolated replica with identical state.

---

## 5. Integration with Evidence Memory

The output of `ChurnEstimator` feeds directly into the evidence decay mechanism of `EvidenceCache`:

```python
from aesw.memory import EvidenceCache, NodeEvidence

cache = EvidenceCache()

# Record evidence from observation at t=0
cache.store_from_observation(obs, target_found=False, confidence=1.0)

# Update churn estimator from subsequent observations
estimator.update_from_observation(obs_next)

# Query decayed evidence parameterized by online estimated churn rate
weighted_ev = cache.get(node_id="target_cell", current_time=10, lambda_hat=estimator)
print(f"Decayed weight: {weighted_ev.weight:.4f}")
```

---

## 6. Multi-Walker Independence

In multi-agent search configurations, each walker maintains its own private `ChurnEstimator` instance:
1. Walkers exploring stable dense clusters estimate low $\hat{\lambda}$, maintaining high memory retention.
2. Walkers traversing volatile boundary edges estimate high $\hat{\lambda}$, rapidly discounting stale observations.
3. Estimators never leak internal state across walker boundaries, ensuring strict decentralization.
