# AESW Evidence Memory & Adaptive Exponential Decay

---

## 1. Overview & Research Objective

In dynamic graph search and transport under uncertainty, search agents accumulate information about vertex states, target sensor detections, and cleared (unoccupied) nodes over time. However, in time-varying topologies governed by edge transitions:

$$G_t = (V, E_t)$$

historical observations risk obsolescence as the graph evolves and moving targets relocate. 

The **Adaptive Evidence-Sharing Walkers (AESW)** framework introduces an **adaptive evidence-memory** model where the value of historical evidence decays exponentially at a rate proportional to the network churn:

> **Information collected in a stable environment persists longer, whereas information collected in a rapidly changing environment decays quickly.**

---

## 2. Mathematical Formulation

### The Adaptive Decay Equation

For an evidence record gathered at simulation step $t_{\text{rec}}$, its age at current simulation time $t$ is:

$$\text{age} = t - t_{\text{rec}} \ge 0$$

The current time-dependent evidence weight $w(t) \in [0.0, 1.0]$ is computed as:

$$w(t) = \exp\left(-\hat{\lambda} \cdot \text{age}\right)$$

where:
- $w$: Current evidence weight scalar.
- $\hat{\lambda} \ge 0$: Estimated or configured graph churn / change rate.
- $\text{age} \ge 0$: Elapsed simulation time steps.

### Behavioral Regimes

1. **Static Graph ($\hat{\lambda} = 0.0$)**:
   $$w = \exp(0) = 1.0 \quad \forall \, \text{age} \ge 0$$
   Evidence never decays; historical observations remain indefinitely valid.

2. **Slowly Changing Graph ($\hat{\lambda} \ll 1$)**:
   $$\hat{\lambda} = 0.01 \implies w(20) = \exp(-0.2) \approx 0.8187$$
   Evidence decays gently; historical clues remain influential across multiple decision epochs.

3. **Rapidly Changing Graph ($\hat{\lambda} \gg 0$)**:
   $$\hat{\lambda} = 0.20 \implies w(20) = \exp(-4.0) \approx 0.0183$$
   Evidence decays aggressively; searchers rapidly discard stale clues to prevent misleading dead-end attraction.

4. **Newly Collected Evidence ($\text{age} = 0$)**:
   $$w = \exp(0) = 1.0 \quad \forall \, \hat{\lambda} \ge 0$$
   Fresh evidence carries full weight immediately upon collection.

### Half-Life Formulation

The characteristic half-life $t_{1/2}$ represents the duration in simulation steps after which evidence weight halves ($w = 0.5$):

$$t_{1/2} = \frac{\ln(2)}{\hat{\lambda}}$$

Conversely, a target half-life translates to the decay rate:

$$\hat{\lambda} = \frac{\ln(2)}{t_{1/2}}$$

### Effective Confidence

Given an initial baseline observation confidence $c_0 \in [0.0, 1.0]$ (e.g. sensor true positive probability $p_d$ or verified physical clearance), the time-decayed effective confidence is:

$$c_{\text{eff}}(t) = c_0 \cdot w(t) = c_0 \cdot \exp\left(-\hat{\lambda} \cdot \text{age}\right)$$

---

## 3. Evidence Data Models

### `NodeEvidence`

Represents an immutable record of evidence collected at or regarding a graph vertex:

```python
from aesw.memory import NodeEvidence, EvidencePolarity

record = NodeEvidence(
    node_id=42,
    visited=True,
    target_found=False,
    signal_strength=0.0,
    timestamp=17,
    walker_id=2,
    confidence=0.8,
)
```

#### Attributes

- `node_id`: Vertex identifier $v \in V$.
- `visited`: Boolean indicating physical presence at the vertex.
- `target_found`: Boolean indicating physical target acquisition.
- `signal_strength`: Observed sensor scalar in $[0.0, 1.0]$.
- `timestamp`: Simulation time step $t$ when recorded.
- `walker_id`: Agent that recorded the observation.
- `confidence`: Baseline reliability $c_0 \in [0.0, 1.0]$.
- `polarity`: `EvidencePolarity` (`POSITIVE` or `NEGATIVE`).
- `source`: `EvidenceSource` (`DIRECT_VISIT`, `REMOTE_SENSOR`, `RECEIVED_EXCHANGE`).

### Dual Evidence Polarities

1. **Positive Evidence (`POSITIVE`)**:
   - Represents positive sensor readings ($y_t(u) = 1$) or target presence indications.
   - Guides exploitation search gradients towards high-likelihood candidate vertices.
2. **Negative Evidence (`NEGATIVE`)**:
   - Represents confirmed absences ($y_t(u) = 0$ or visited empty vertices).
   - Informs search dispersion by signaling that a vertex has recently been cleared, preventing redundant re-inspections until the negative evidence decays.

---

## 4. Local Evidence Memory: `EvidenceCache`

The `EvidenceCache` manages local node-level memory storage, update policies, time evaluations, and memory pruning.

### Update Rules

When a candidate `NodeEvidence` record for vertex $v$ is submitted:
1. **New Vertex**: Stored directly.
2. **Existing Vertex**:
   - **Strictly Newer ($t_{\text{new}} > t_{\text{existing}}$)**: Overwrites existing entry.
   - **Equal Timestamp ($t_{\text{new}} = t_{\text{existing}}$)**: Overwrites only if $c_{\text{new}} \ge c_{\text{existing}}$.
   - **Older ($t_{\text{new}} < t_{\text{existing}}$)**: Ignored (retains the fresher observation).

### Querying and Filtering

- `cache.get(node_id, current_time, lambda_hat)`: Returns a `WeightedEvidence` snapshot with evaluated `age`, `weight`, and `effective_confidence`.
- `cache.get_positive(current_time)`: Returns positive evidence entries ordered by effective confidence descending.
- `cache.get_negative(current_time)`: Returns negative evidence entries ordered by effective confidence descending.
- `cache.prune_stale(current_time, min_weight)`: Discards entries whose decay weight has fallen below $w_{\min}$.

### Ingestion from Local Observations

```python
cache.store_from_observation(
    observation=walker_observation,
    current_time=12,
    confidence=0.9,
)
```
Constructs a typed `NodeEvidence` instance from the walker's local perception snapshot, maintaining strict epistemic boundaries.

---

## 5. Scope Boundaries

- **Input $\hat{\lambda}$**: In this subsystem, $\hat{\lambda}$ is supplied externally (or uses a configured default).
- **Subsequent Development**: Dynamic online churn estimation ($\hat{C} \to \hat{\lambda}$), inter-walker evidence exchange, and search policy decision heuristics are implemented in subsequent modules.

