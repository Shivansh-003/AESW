# Formal Problem Definition & Computational Contract

---

## 1. Mathematical Environment Formulation

The search and transport environment is modeled as a discrete-time dynamic graph:

$$G_t = (V, E_t)$$

where:
- **Node Set**: $V = \{v_1, v_2, \dots, v_n\}$ represents the static set of $n = |V|$ vertices.
- **Dynamic Edge Set**: $E_t \subseteq E$ denotes the subset of active edges at discrete simulation step $t \in \{0, 1, 2, \dots\}$.
- **Underlying Edge Universe**: $E$ represents all edges that can potentially exist in the graph topology.

### Edge State Representation
Each edge $e = (u, v) \in E$ possesses a state at time $t$:
$$s_t(e) \in \{\text{ON}, \text{OFF}\}$$
- $\text{ON}$: The edge is active and traversable; $e \in E_t$.
- $\text{OFF}$: The edge is inactive, impassable, and blocked; $e \notin E_t$.

The dynamic transition between states is governed by transition rates:
- $p_{\text{on}} = \Pr(s_{t+1}(e) = \text{ON} \mid s_t(e) = \text{OFF})$
- $p_{\text{off}} = \Pr(s_{t+1}(e) = \text{OFF} \mid s_t(e) = \text{ON})$

---

## 2. Target Locomotion Model

The target occupies a vertex:
$$x_t \in V$$

At each discrete time step $t$:
- If target mode is `STATIC`, $x_{t+1} = x_t$.
- If target mode is `MOVING`, the target transitions to an adjacent active neighbor with probability $p_{\text{move}}$:
$$\Pr(x_{t+1} \in \mathcal{N}_t(x_t)) = p_{\text{move}}, \quad \text{where } \mathcal{N}_t(u) = \{v \in V \mid (u, v) \in E_t\}$$
- With probability $1 - p_{\text{move}}$, the target remains at $x_t$.

---

## 3. Traversal Delay Model

Traversing an active edge $e \in E_t$ incurs a discrete duration:
$$d(e, t) \ge 1$$
- **Information Asymmetry**: A searcher does **not** know the realized traversal delay before initiating traversal along $e$.
- During traversal, the walker is marked busy until $t + d(e, t)$ and cannot make new decisions or inspect intermediate nodes.

---

## 4. Controlled Observation Boundary & Noisy Sensing

The simulation enforces a strict boundary between the unobserved `GroundTruthState` and the local `Observation` emitted to a walker $w$ at current vertex $u = \text{pos}(w, t)$.

```text
┌────────────────────────────────────────────────────────┐
│               Ground Truth State (Hidden)              │
│  - Full topology V, E                                  │
│  - Active edge configuration E_t                       │
│  - True target location x_t                            │
│  - Future delays and transitions                       │
└───────────────────────────┬────────────────────────────┘
                            │
               Observation Boundary / Sensor
                            │
┌───────────────────────────▼────────────────────────────┐
│              Walker Observation (Observable)           │
│  - Walker identity and current vertex u                │
│  - Current simulation time t                           │
│  - Checked neighbors (up to budget B)                  │
│  - Observed states of checked incident edges           │
│  - Noisy local target detection signal                 │
└────────────────────────────────────────────────────────┘
```

### Neighbor Check Budget $B$
At vertex $u$, a walker can only inspect up to $B$ incident edges in a single decision step:
$$|\text{Checked}(u, t)| \le B$$

### Noisy Target Detection Model
At node $u$, the walker receives a binary sensor reading $y_t(u) \in \{0, 1\}$:
- **Signal Radius $s$**: Nodes within shortest-path distance $s$ hops from $x_t$ emit a detection signal with probability $p_d$:
$$\Pr(y_t(u) = 1 \mid \text{dist}_{G_t}(u, x_t) \le s) = p_d$$
- **Missed Detection**: Target is present/proximate, but sensor fails: $\Pr(y_t = 0 \mid \text{dist} \le s) = 1 - p_d$.
- **False Alarm**: Sensor produces a positive signal at a distant node with probability $p_{\text{fa}}$:
$$\Pr(y_t(u) = 1 \mid \text{dist}_{G_t}(u, x_t) > s) = p_{\text{fa}}$$

---

## 5. Cost Model & Evaluation Contract

### Total Search Cost
For an episode involving $M$ total physical moves and $Q$ total exchanged messages:
$$C_{\text{total}} = M + c_m \cdot Q$$
where $c_m \ge 0$ is the communication cost penalty coefficient.

### Per-Run Termination Contract
A run terminates under one of the following validated states:
1. `SUCCESS`: At least one walker occupies the target node $x_t$ and validates acquisition within the step budget.
2. `BUDGET_EXHAUSTED`: Simulation reaches maximum time horizon $t = T_{\max}$ without acquiring the target.
3. `DISCONNECTED`: The walker's component becomes entirely disconnected from the target due to edge attrition.
