# System Architecture: Adaptive Graph Search

---

## 1. Architectural Overview & Component Hierarchy

The system follows a strict layered design to ensure modularity, separation of concerns, and scientific integrity:

```text
┌────────────────────────────────────────────────────────┐
│                      Graph Layer                       │
│    Topologies: ER, BA, Watts-Strogatz, Grid, AS-733    │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│                     Dynamics Layer                     │
│         Edge ON/OFF transitions, dynamic regimes       │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│                   Environment Layer                    │
│     Time step orchestration, Target movement, Delays   │
│         (Maintains Hidden Ground-Truth State)          │
└───────────────────────────┬────────────────────────────┘
                            │
              Controlled Observation Boundary
              (Sensor noise, partial visibility)
                            │
┌───────────────────────────▼────────────────────────────┐
│                    Observation Layer                   │
│   Local 1-hop view, Noisy target cue, Edge status      │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│                      Walker Layer                      │
│     Generic walker abstraction, step & state loop      │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
┌─────────────▼───────────────┐ ┌─────────▼──────────────┐
│     Baseline Algorithms     │ │         AESW           │
│  RW, NBW, k-RW, Flooding    │ │  Churn estimator,      │
│  Degree-Based, Ant Colony   │ │  Mode switch, Softmax  │
└─────────────────────────────┘ └─────────┬──────────────┘
                                          │
                                ┌─────────▼──────────────┐
                                │     Memory Layer       │
                                │  Node cache, Evidence  │
                                │  Adaptive churn decay  │
                                └────────────────────────┘
                                          │
┌─────────────────────────────────────────▼──────────────┐
│                    Evaluation Layer                    │
│   Search time, success rate, cost, message volume      │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│             Results & Visualization Layer              │
│       Tabular benchmarks, confidence intervals,        │
│          trajectory plots, regime sweeps               │
└────────────────────────────────────────────────────────┘
```

---

## 2. Package Responsibilities

- **`aesw.graph`**: Responsible for static graph representations, adjacency matrices, and synthetic/real graph generators.
- **`aesw.dynamics`**: Manages edge availability transitions over time, parameterized by transition probabilities ($p_{\text{on}}, p_{\text{off}}$) and dynamic regimes (`STATIC` to `VERY_FAST`).
- **`aesw.environment`**: The central simulation coordinator and data model package. Defines:
  - [`ProblemDefinition`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/problem.py): The immutable computational specification.
  - [`GroundTruthState`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/state.py): Complete unobserved dynamic world state.
  - [`Observation`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/observation.py): Local, noisy perceptions bounded by budget $B$.
  - Strongly typed models (`Node`, `Edge`, `TargetState`, `WalkerState`).
- **`aesw.walkers`**: Defines base interfaces for search agents. Provides standard lifecycle methods: `observe()`, `decide_step()`, and `emit_messages()`.
- **`aesw.baselines`**: Standard non-adaptive and heuristic graph search algorithms for scientific comparison.
- **`aesw.aesw`**: The proposed Adaptive Evidence-Sharing Walkers algorithm, containing churn estimation, exploration/exploitation mode gating, and softmax neighbor selection.
- **`aesw.memory`**: Dual evidence data structures (positive/negative), node cache management, and adaptive decay functions.
- **`aesw.evaluation`**: Standardized metric contracts ([`RunMetrics`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/evaluation/metrics.py), [`AggregatedMetrics`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/evaluation/metrics.py)), episode aggregation, and statistical significance testing.
- **`aesw.utils`**: Deterministic configuration loading, isolated pseudo-random number generators (`create_rng`), and logging utilities.

---

## 3. ProblemDefinition Contract vs. Runtime State

To preserve immutability and testability across all simulation runs:

1. **Specification Phase**:
   ```text
   YAML Configs ──► load_config() ──► Validation ──► ProblemDefinition (Frozen Dataclass)
   ```
   `ProblemDefinition` contains validated parameters for the graph, dynamics, target, detection, observation budget, delay, walkers, simulation limits, and cost model. It contains **no mutable execution state**.

2. **Simulation Execution**:
   A simulation instance will consume the immutable `ProblemDefinition`, instantiate an initial `GroundTruthState`, and advance discrete simulation steps $t = 0, 1, 2, \dots$.

---

## 4. The Controlled Observation Principle

A foundational requirement for scientific fairness and realistic simulation is:

> **Algorithms must interact with the environment through controlled interfaces rather than accessing hidden ground-truth information.**

1. **No Target Peeking**: Walkers can never query `environment.target_node` or `GroundTruthState.target_state`. They only receive a sensor observation `TargetSignalObservation` which is subject to detection probability $p_d$, false alarm probability $p_{\text{fa}}$, and signal radius $s$.
2. **Local Horizon Only**: Walkers can only perceive the availability of edges incident to their current vertex ($k$-hop neighborhood where $k=1$) and are strictly constrained by the neighbor check budget $B$.
3. **No Global Adjacency Leakage**: Walkers cannot view the global connectivity matrix or predict upcoming edge state transitions.
4. **Fairness across Baselines**: Both AESW and baseline algorithms receive the exact same `Observation` instances from the environment.
