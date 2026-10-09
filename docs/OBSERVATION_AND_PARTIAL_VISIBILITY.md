# Observation and Partial Visibility Layer

---

## 1. Purpose & Research Objective

In real-world distributed search environments—such as autonomous robots exploring disaster zones, mobile ad-hoc communication networks, and distributed peer-to-peer systems—agents operate without global omniscience. A searching agent (walker) does not possess global topology graphs, cannot peek at true target positions, and cannot forecast future link activations.

The **Observation and Partial Visibility Layer** serves as an architectural information firewall:

$$\text{GroundTruthState} \xrightarrow{\text{Controlled Projection}} \text{ObservationBuilder} \xrightarrow{\text{Immutable Snapshot}} \text{Observation} \xrightarrow{\text{Perception}} \text{Walker}$$

This guarantees that search algorithms evaluate exclusively under realistic partial visibility, bounded sensor inspection budgets, and noisy observations.

---

## 2. Research-Specified vs. Implementation Choices

To maintain methodological clarity and scientific integrity, architectural decisions are distinguished as follows:

| Category | Aspect | Specification |
| :--- | :--- | :--- |
| **Research-Specified** | **Partial Local Visibility** | Walkers observe only immediate $1$-hop active neighbors in $G_t = (V, E_t)$. Multi-hop paths and inactive edges are hidden. |
| **Research-Specified** | **Neighbor Budget $B$** | Walkers can inspect at most $B$ candidate incident edges/neighbors during an observation opportunity ($B > 0$). |
| **Research-Specified** | **Hidden Target & Sensing Noise** | Target location $x_t \in V$ is hidden. Sensing is radius-constrained ($s$) with true detection probability $p_d$ and false alarm probability $p_{\text{fa}}$. |
| **Research-Specified** | **No Future Knowledge** | Walkers receive no future edge transition schedules, future target trajectories, or delay realizations. |
| **Implementation Choice** | **Neighbor Sampling under $B$** | When local degree $\text{deg}_{G_t}(u) > B$, $B$ neighbors are sampled uniformly without replacement using an isolated PRNG stream. |
| **Implementation Choice** | **Data Structures** | Immutable frozen dataclasses (`Observation`, `ObservedEdgeInfo`, `TargetSignalObservation`, `LocalObservationHistory`). |
| **Implementation Choice** | **Decoupled RNG Streams** | Dedicated NumPy generator stream for observation sampling, isolated from graph generation, edge churn, target locomotion, and sensor noise. |

---

## 3. Information Boundary: Ground Truth vs. Walker Perception

```text
┌────────────────────────────────────────────────────────┐
│             Ground Truth Simulation Space              │
│                                                        │
│   • Complete static topology G = (V, E)                │
│   • Exact dynamic edge availability E_t                │
│   • True target node x_t ∈ V and locomotion history    │
│   • Sensor classification (POSITIVE, MISSED, etc.)     │
│   • Exact shortest-path active graph distance          │
│   • Future transition schedules                        │
└──────────────────────────┬─────────────────────────────┘
                           │
                ObservationBuilder Firewall
                (Applies Budget B & Hides Ground Truth)
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Walker Observation Space                 │
│                                                        │
│   Observation                                          │
│   • current_node: NodeId (walker's physical position)  │
│   • time: int (current discrete step t)                │
│   • checked_neighbors: tuple (inspected, ≤ B)          │
│   • visible_neighbors: tuple (bounded by B)            │
│   • observed_edges: dict[NodeId -> ObservedEdgeInfo]   │
│   • target_signal: TargetSignalObservation (detected)  │
│   • neighbor_budget_used: int                          │
│   • local_history: Optional[LocalObservationHistory]   │
│                                                        │
│   ❌ NO target_node or true_target                     │
│   ❌ NO exact graph distance to target                 │
│   ❌ NO detection classification (outcome_category)    │
│   ❌ NO full graph topology or global adjacency list   │
│   ❌ NO future link states or edge transition matrix   │
└────────────────────────────────────────────────────────┘
```

---

## 4. Local Neighbor Visibility Model

The default visibility model evaluates the active subgraph $G_t = (V, E_t)$ from the walker's current vertex $u$:

$$\mathcal{N}_{G_t}(u) = \{v \in V \mid (u, v) \in E_t\}$$

- **Inactive / OFF Edges**: Edges $(u, v) \in E \setminus E_t$ are not operational and do not appear as traversable links.
- **Multi-Hop Isolation**: Vertices at graph distance $\ge 2$ in $G_t$ are completely unobservable.
- **Topological Locality**: The walker has no global adjacency matrix or vertex catalog $V$.

---

## 5. Neighbor Checking Budget ($B$)

The computational and bandwidth limits of local physical sensing are governed by neighbor budget $B \in \mathbb{N}_{> 0}$:

1. **Sub-Budget Regime ($\text{deg}_{G_t}(u) \le B$)**:
   The walker inspects all active neighbors:
   $$\text{checked\_neighbors} = \mathcal{N}_{G_t}(u)$$
   $$\text{neighbor\_budget\_used} = |\mathcal{N}_{G_t}(u)| \le B$$

2. **Super-Budget Regime ($\text{deg}_{G_t}(u) > B$)**:
   The walker cannot inspect all candidates simultaneously. Exactly $B$ distinct neighbors are selected uniformly at random without replacement:
   $$\text{checked\_neighbors} \sim \text{UniformSubset}(\mathcal{N}_{G_t}(u), B)$$
   $$\text{neighbor\_budget\_used} = B$$

**Strict Information Limiting**: Unchecked neighbors are **never** returned in the observation. The walker receives information strictly for the candidates it actually inspected.

---

## 6. Observed Edge Information

For each inspected neighbor $v \in \text{checked\_neighbors}$, the observation records an [`ObservedEdgeInfo`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/observation.py#L14-L33) record:
- `neighbor_id`: Node identifier $v$.
- `state`: Observed operational state (`EdgeState.ON` or `EdgeState.OFF`).
- `observed_at_time`: Simulation timestamp $t$.

The walker does **not** receive:
- Edge transition probabilities ($p_{\text{on}}, p_{\text{off}}$) unless provided as prior configuration.
- Future link states at $t+1, t+2, \dots$.
- Uninspected edges or global edge sets $E$.

---

## 7. Target Detection Signal Transformation

Sensor readings are obtained via the [`DetectionEngine`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/detection.py#L54-L187):

$$\text{DetectionResult}(u, x_t, G_t) \xrightarrow{\text{Projection}} \text{TargetSignalObservation}(\text{detected} \in \{\text{True}, \text{False}\})$$

### Epistemic Indistinguishability
The walker cannot distinguish:
- A **True Positive** ($\text{dist} \le s$, roll $< p_d$) from a **False Alarm** ($\text{dist} > s$, roll $< p_{\text{fa}}$). Both present identically as `detected = True`.
- A **True Negative** ($\text{dist} > s$, roll $\ge p_{\text{fa}}$) from a **Missed Detection** ($\text{dist} \le s$, roll $\ge p_d$). Both present identically as `detected = False`.

The ground-truth outcome classification (`POSITIVE`, `MISSED`, `FALSE_POSITIVE`, `NEGATIVE`) and exact distance $\text{dist}_{G_t}(u, x_t)$ are strictly withheld (`outcome_category = None`).

---

## 8. Unknown vs. Known Absent Information

The observation model explicitly differentiates unobserved information from confirmed absence:

| Condition | Status | Method Query |
| :--- | :--- | :--- |
| Inspected neighbor $v$, edge is `ON` | **Known Available** | `obs.is_edge_known_active(v) == True` |
| Inspected neighbor $v$, edge is `OFF` | **Known Unavailable** | `obs.is_edge_known_inactive(v) == True` |
| Node $w$ was not inspected in this step | **Unknown / Unobserved** | `obs.is_edge_unknown(w) == True` |

Unobserved nodes evaluate to `get_edge_status(w) is None`. The walker never assumes "not observed = does not exist."

---

## 9. Local History & Snapshot Semantics

[`LocalObservationHistory`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/observation.py#L60-L115) stores an immutable chronological sequence of past observations made by a walker:

$$H_t = \langle \text{Obs}_0, \text{Obs}_1, \dots, \text{Obs}_t \rangle$$

- **Snapshot Invariance**: Later environment mutations (dynamic edge transitions, target movements) do not mutate past history entries.
- **Local Scope**: History contains only walker-experienced observations and visited nodes.

---

## 10. Reproducibility & RNG Stream Decoupling

Observation sampling uses an independent NumPy generator stream created with `create_rng(seed)`.

$$\text{Observation Seed} \perp \text{Graph Seed} \perp \text{Dynamics Seed} \perp \text{Target Seed} \perp \text{Detection Seed}$$

Modifying the observation seed alters which neighbors are sampled when $\text{deg} > B$, but produces zero variance in graph topology, edge transitions, or target locomotion paths.

---

## 11. Concrete Example: Partial Visibility under Budget $B=1$

Given path graph $A - B - C - D - E$ with walker at $B$ at time $t$:
1. Active neighbor candidates: $\{A, C\}$.
2. Budget $B = 1$. The engine samples candidate $C$.
3. Resulting `Observation`:
   ```python
   current_node = "B"
   checked_neighbors = ("C",)
   visible_neighbors = ("C",)
   observed_edges = {"C": ObservedEdgeInfo(neighbor_id="C", state=EdgeState.ON, observed_at_time=t)}
   target_signal = TargetSignalObservation(detected=False, outcome_category=None, timestamp=t)
   neighbor_budget_used = 1
   ```
4. **Information Boundary Assertions**:
   - Node $A$ is **unknown** (`obs.is_edge_unknown("A") is True`), not absent.
   - Nodes $D$ and $E$ are completely unobserved.
   - Target location and exact distance are absent.

---

## 12. Explicit Non-Goals

The following components are intentionally excluded from this layer:
- **Search Intelligence**: No walker movement decision logic, heuristic scoring, or random walk navigation.
- **AESW Memory Caching**: No global churn estimation $\hat{C}$, evidence half-life decay, or node cache sharing.
- **Simulation Coordinator**: Multi-walker step orchestration and benchmark execution loops are handled by the evaluation and coordinator modules.
