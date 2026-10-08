# Project Specification: Adaptive Graph Search (AESW)

---

## 1. Research Context & Problem Formulation

In distributed autonomous systems, peer-to-peer computing, wireless sensor networks, and disaster response scenarios, agents must locate targets on non-stationary, evolving network topologies.

### Challenges
- **Dynamic Topology**: Edges appear and disappear stochastically, altering shortest paths, connectivity, and reachability.
- **Moving Target**: The target does not stay fixed at a static node; it can drift across active edges.
- **Stochastic Delays**: Traversal through dynamic edges entails variable traversal times and delays.
- **Partial Observability**: Walkers lack a centralized global map; they observe only their immediate local neighborhood.
- **Unreliable Sensing**: Target presence sensors exhibit both false positive detections (false alarms) and false negative detections (misses).

---

## 2. Implementation Status & Roadmap

| Component | Status | Details |
| :--- | :--- | :--- |
| **Project Foundation & Package Structure** | **Implemented** | Modular package hierarchy, clean `__init__.py` files, PEP 8 layout |
| **Configuration Subsystem** | **Implemented** | Deterministic YAML loaders, structured configuration templates |
| **Reproducibility Utilities** | **Implemented** | Isolated RNG generator creation (`create_rng`), seed locking |
| **Formal Problem Definition Contract** | **Implemented** | Immutable `ProblemDefinition`, data models (`Node`, `Edge`, `TargetState`), types (`EdgeState`, `DetectionOutcome`), observation boundary |
| **State Representation & Observation Boundary**| **Implemented** | `GroundTruthState`, `Observation`, `ObservedEdgeInfo`, `TargetSignalObservation` |
| **Graph Generation Engine** | **Implemented** | Deterministic generators for ER, BA, WS, 2D Grid with obstacles, RGG; `GraphInstance` abstraction; topology statistics |
| **Dynamic Graph Engine** | **Implemented** | Stochastic Markovian edge transitions ($p_{\text{on}}, p_{\text{off}}$), synchronous updates, dynamic regimes, `DynamicGraphState`, `ActiveGraphView` |
| **Target & Uncertainty Engine** | **Implemented** | Hidden target locomotion ($p_{\text{move}}$), active edge constraint, trapped target handling, radius sensing ($s$), detection noise ($p_d, p_{\text{fa}}$), blind search boundary, isolated RNG |
| **Observation & Partial Visibility Layer** | **Implemented** | Information firewall, `ObservationBuilder`, neighbor checking budget $B$, binary detection reduction, local history snapshots, known absent vs unknown distinction |
| **Baseline Search Algorithms** | **Implemented** | Six benchmark policies (RW, NBW, k-RW, Degree-Based, Flooding, Ant Colony) operating strictly on local `Observation` snapshots |
| **Automated Verification Suite** | **Implemented** | 150 unit and statistical tests covering foundation, problem contracts, graph generators, dynamic transitions, target mobility, sensing uncertainty, observation firewall, and baselines |
| **Simulation Environment Coordinator** | *Planned* | Discrete-time orchestration, hidden vs. observable state separation |
| **Walker Abstractions** | *Planned* | Common walker interfaces and message handling |
| **AESW Search Algorithm** | *Planned* | Core AESW algorithm, mode selector, softmax decision mechanism |
| **Evidence Memory & Caching** | *Planned* | Node cache, positive/negative evidence, churn-driven decay |
| **Evaluation Suite & Metrics** | *Planned* | Search time, success rate, messages, movement/communication cost |
| **Experimental Benchmarks** | *Planned* | Synthetic topologies, dynamic regimes, noise sweeps, AS-733 dataset |
| **Statistical Analysis** | *Planned* | Hypothesis testing, p-values, 95% confidence intervals |
| **Visualizations & Report** | *Planned* | Trajectory plots, metric bar plots, final manuscript figures |

---

## 3. Scope & Non-Goals

### In-Scope
- Discrete-time multi-agent simulations on dynamic graphs.
- Rigorous scientific benchmarking of proposed AESW against 6 baseline algorithms under identical conditions.
- Strict partial observability: walkers only receive local observations.
- Realistic sensor noise and edge churn modeling.
- Statistical significance evaluation with confidence intervals over $\ge 50$ independent stochastic runs.

### Explicit Non-Goals
- Web servers, REST APIs, or frontend visualization dashboards.
- Relational or document databases (results are persisted as structured CSV/JSON/NPY files).
- Heavyweight distributed architectures or cloud orchestrators (the simulation runs efficiently in standard scientific Python).
- Full global state access for walkers (hidden ground truth must never leak into algorithm decisions).

---

## 4. Novelty Pillars of AESW

1. **Adaptive Memory Decay via Estimated Churn**:
   Walkers estimate local network dynamics $\hat{C}$ and scale evidence half-life accordingly.
2. **Dynamic Exploitation vs. Exploration Mode Switching**:
   Walkers autonomously transition between greedy gradient ascent (exploitation) and uniform dispersion (exploration) based on evidence confidence and information gain.
3. **Dual Evidence Representation (Positive & Negative)**:
   Walkers share both presence clues and absence confirmations to avoid mutual redundancy and dead-end trapping.

---

## 5. Experimental & Reproducibility Philosophy

1. **Hermetic Determinism**: All stochastic behaviors are driven by explicit seed injection. Given seed $S$ and configuration $C$, the trajectory and metrics must be bit-for-bit identical across runs.
2. **Fairness across Algorithms**: All algorithms (baselines and AESW) operate under the identical environment harness, observation budget, and sensor noise profiles.
3. **Decoupled Configuration**: Research parameters are never hardcoded in source modules.
