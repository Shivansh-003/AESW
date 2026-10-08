# Target Locomotion and Detection Uncertainty Engine

---

## 1. Overview & Research Scope

The Target and Uncertainty subsystem models the hidden physical state and imperfect observability of a search target within dynamic networks:

$$G_t = (V, E_t)$$

In search and transport scenarios, the target occupies a node in the network and may remain stationary or dynamically relocate over operational links. Furthermore, observer agents (walkers) possess imperfect detection sensors characterized by non-zero miss rates and false alarms.

This subsystem provides:
1. **Target Locomotion**: Stochastic movement across active edges governed by mobility parameter $p_{\text{move}}$.
2. **Probabilistic Target Detection**: Radius-constrained sensor readings governed by true positive rate $p_d$ and false alarm rate $p_{\text{fa}}$.
3. **Rigorous Epistemic Boundary**: Complete decoupling between ground-truth environmental state and observer-accessible signals.

---

## 2. Target Locomotion Model

### State Representation
The ground-truth target state is represented at discrete time step $t$ by:

$$x_t \in V$$

The state transition history is tracked deterministically via `TargetState`:
- `node`: Current node $x_t \in V$.
- `previous_node`: Immediate predecessor $x_{t-1} \in V$ (or `None` at $t=0$).
- `time`: Current discrete simulation step $t \ge 0$.
- `move_attempted`: Boolean flag indicating whether mobility was triggered.
- `move_succeeded`: Boolean flag indicating whether the target transitioned to a distinct active neighbor.

### Transition Dynamics & Simulation Order
Target locomotion occurs **strictly after** the dynamic graph edge states for step $t$ have settled:

$$G_{t-1} \xrightarrow{\text{edge churn}} G_t \xrightarrow{\text{target transition}} x_t$$

At discrete step $t$:
1. With probability $1 - p_{\text{move}}$, the target remains at its current node:
   $$x_t = x_{t-1}$$
2. With probability $p_{\text{move}}$, the target attempts to move. It inspects the set of currently active neighbors:
   $$\mathcal{N}_{G_t}(x_{t-1}) = \{v \in V \mid (x_{t-1}, v) \in E_t\}$$
3. **Trapped State Condition**: If $\mathcal{N}_{G_t}(x_{t-1}) = \emptyset$ (the target is topologically isolated or all incident edges are `OFF`), the target cannot traverse any link:
   $$x_t = x_{t-1}, \quad \text{move\_attempted} = \text{True}, \quad \text{move\_succeeded} = \text{False}$$
4. **Active Relocation**: If $\mathcal{N}_{G_t}(x_{t-1}) \ne \emptyset$, the target selects a destination node uniformly at random:
   $$x_t \sim \text{Uniform}(\mathcal{N}_{G_t}(x_{t-1}))$$
   $$x_t \ne x_{t-1}, \quad \text{move\_attempted} = \text{True}, \quad \text{move\_succeeded} = \text{True}$$

---

## 3. Probabilistic Detection & Uncertainty Model

Sensor models are parametrized by a detection radius $s \in \mathbb{N}_{\ge 0}$, detection probability $p_d \in [0, 1]$, and false alarm probability $p_{\text{fa}} \in [0, 1]$.

### Dynamic Graph Distance
Given an observer node $u \in V$ and ground-truth target node $x_t \in V$, the sensor distance is defined as the shortest path hop distance evaluated strictly over the currently active topology $G_t = (V, E_t)$:

$$\text{dist}_{G_t}(u, x_t)$$

If no active path exists between $u$ and $x_t$ at step $t$ (e.g., disconnected components or inactive bridges), the distance is defined as:

$$\text{dist}_{G_t}(u, x_t) = \infty$$

### Probabilistic Signal Generation
A probe from observer node $u$ produces a binary detection signal:

$$\sigma(u) \in \{0, 1\}$$

sampled according to the distance regime:

$$\Pr(\sigma(u) = 1 \mid u, x_t, G_t) = \begin{cases} p_d & \text{if } \text{dist}_{G_t}(u, x_t) \le s \\ p_{\text{fa}} & \text{if } \text{dist}_{G_t}(u, x_t) > s \end{cases}$$

### Detection Classification
The ground-truth classification of an observation probe falls into four mutually exclusive categories:

| Distance Condition | Signal Received ($\sigma=1$) | No Signal ($\sigma=0$) |
| :--- | :--- | :--- |
| $\text{dist}_{G_t}(u, x_t) \le s$ (In Radius) | **`POSITIVE`** (True Positive, prob $p_d$) | **`MISSED`** (False Negative, prob $1 - p_d$) |
| $\text{dist}_{G_t}(u, x_t) > s$ (Out of Radius) | **`FALSE_POSITIVE`** (False Alarm, prob $p_{\text{fa}}$) | **`NEGATIVE`** (True Negative, prob $1 - p_{\text{fa}}$) |

### Blind-Search Boundary Condition
When $p_d = p_{\text{fa}}$, the likelihood ratio:

$$\Lambda = \frac{\Pr(\sigma = 1 \mid \text{dist} \le s)}{\Pr(\sigma = 1 \mid \text{dist} > s)} = \frac{p_d}{p_{\text{fa}}} = 1$$

Under this condition, sensor readings provide zero mutual information regarding target distance or location, reducing walker dynamics to unguided blind search.

---

## 4. Ground Truth vs. Observer Information Boundary

A central requirement of the research framework is the strict epistemic separation between ground truth and agent-accessible data:

```text
┌────────────────────────────────────────────────────────┐
│             Ground Truth Simulation Space              │
│                                                        │
│   TargetState (x_t)          DetectionResult           │
│   - target_node              - signal: bool            │
│   - trajectory history       - graph_distance: float   │
│   - move success/attempt     - outcome: Classification │
│   - incident edge states     - target_node: NodeId     │
└──────────────────────────┬─────────────────────────────┘
                           │
                 Controlled Projection
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Observer Observation Space               │
│                                                        │
│   Observation                                          │
│   - observer_node: NodeId                              │
│   - signal: bool  (EXCLUSIVELY this bit is exposed)    │
│                                                        │
│   ❌ NO target_node                                    │
│   ❌ NO true graph_distance                            │
│   ❌ NO POSITIVE vs FALSE_POSITIVE classification      │
└────────────────────────────────────────────────────────┘
```

The `DetectionResult` structure is reserved strictly for internal engine telemetry, validation, and benchmarking metrics. Walkers and algorithms receive only the binary signal bit.

---

## 5. Independent Pseudo-Random Number Generation

To guarantee experimental reproducibility and enable variance reduction via common random numbers, all stochastic processes use decoupled NumPy PRNG streams:

| Subsystem | Seed Parameter / Configuration | Responsibility |
| :--- | :--- | :--- |
| **Topology Generation** | `graph.seed` | Static adjacency structure |
| **Edge Dynamics** | `dynamics.seed` | Markovian ON/OFF transitions |
| **Target Locomotion** | `target.seed` | Target initial placement & movement choices |
| **Detection Noise** | `detection.seed` | Sensor detection and false alarm draws |

Modifying `detection.seed` alters sensor noise without modifying the physical path taken by the target or the dynamic evolution of graph edges.

---

## 6. Architectural Components

- **`TargetEngine`** ([`src/aesw/environment/target.py`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/target.py)):
  - Manages target node tracking, initialization, and stochastic mobility transitions.
  - Interacts directly with `ActiveGraphView` to ensure relocation occurs only across operational edges.
- **`DetectionEngine`** ([`src/aesw/environment/detection.py`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/detection.py)):
  - Evaluates active graph shortest-path hop distances via breadth-first search.
  - Samples stochastic detections based on $s$, $p_d$, and $p_{\text{fa}}$.
  - Produces internal `DetectionResult` audit records.
- **`TargetState` & `DetectionResult`** ([`src/aesw/environment/models.py`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/models.py)):
  - Immutable dataclasses encapsulating complete ground-truth telemetry.

---

## 7. Programmatic Usage Example

```python
from aesw.generation import create_graph_generator
from aesw.dynamics import DynamicGraphState
from aesw.environment import TargetEngine, DetectionEngine

# 1. Instantiate static graph and dynamic state
generator = create_graph_generator("erdos_renyi", n=50, p=0.1, seed=42)
static_graph = generator.generate()
dyn_graph = DynamicGraphState(static_graph, p_on=0.1, p_off=0.05, seed=100)

# 2. Instantiate target and detection engines
target_engine = TargetEngine(
    graph_instance=static_graph,
    p_move=0.4,
    initial_node=0,
    seed=200,
)
detection_engine = DetectionEngine(
    radius=2,
    p_d=0.9,
    p_fa=0.05,
    seed=300,
)

# 3. Step environment: graph evolves first, then target relocates
dyn_graph.advance()
active_view = dyn_graph.view()
target_state = target_engine.step(active_view)

# 4. Probe sensor from walker node
result = detection_engine.detect(
    walker_node=10,
    target_node=target_state.node,
    active_graph=active_view,
)

print(f"Target at node: {target_state.node}")
print(f"Sensor signal: {result.signal}")
print(f"Internal distance: {result.graph_distance}")
print(f"Ground truth outcome: {result.outcome.value}")
```
