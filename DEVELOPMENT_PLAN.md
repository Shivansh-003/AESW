# Research Implementation Roadmap

Adaptive Search and Transport on Dynamic Graphs Under Uncertainty (AESW)

---

## Overview

This document outlines the research implementation roadmap for the decentralized target search framework and the Adaptive Evidence-Sharing Walkers (AESW) method.

---

## 1. Project Foundation & Infrastructure
- **Status**: **Completed**
- **Scope**:
  - Modular Python package layout under `src/aesw`.
  - Declarative YAML configuration system (`configs/*.yaml`).
  - Isolated, reproducible random number generator utilities (`create_rng`, `seed_everything`).
  - Automated testing foundation with pytest.

---

## 2. Formal Problem Definition
- **Status**: **Completed**
- **Scope**:
  - Mathematical definition of dynamic graphs $G_t = (V, E_t)$, discrete time steps $t$, and Markovian edge transition parameters ($p_{\text{on}}, p_{\text{off}}$).
  - Strongly typed data models and enums (`EdgeState`, `TargetMode`, `DynamicRegime`, `DetectionOutcome`, `SearchTerminationStatus`, `Node`, `Edge`, `TargetState`, `WalkerState`).
  - Strict architectural separation between unobserved ground-truth state (`GroundTruthState`) and walker perceptions (`Observation`).
  - Immutable specification container (`ProblemDefinition`) with validation and YAML configuration mapping.
  - Evaluation contracts for per-run metrics (`RunMetrics`) and statistical summaries (`AggregatedMetrics`) under total cost model $C_{\text{total}} = M + c_m \cdot Q$.

---

## 3. Graph Generation Engine
- **Status**: **Completed**
- **Scope**:
  - Concrete deterministic generators for Erdős–Rényi $G(n, p)$, Barabási–Albert $\text{BA}(n, m)$, Watts–Strogatz $\text{WS}(n, k, p)$, 2D Spatial Grid with Obstacles, and Random Geometric Graphs (RGG).
  - `GraphInstance` research abstraction encapsulating NetworkX internally.
  - Topology metrics (`compute_statistics()`), metadata provenance (`GraphMetadata`), and invariant assertions.
  - Factory dispatchers and extension interface (`TemporalGraphSource`) for future temporal network traces.

---

## 4. Dynamic Edge Transition Engine
- **Status**: **Completed**
- **Scope**:
  - Stochastic discrete-time Markovian edge transition model ($p_{\text{on}}, p_{\text{off}}$) over immutable static topologies $G = (V, E)$.
  - Synchronous, order-independent state update engine ([`DynamicGraphState`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/dynamics/engine.py)).
  - Edge initialization policies (`ALL_ON`, `ALL_OFF`, `STATIONARY`).
  - Named churn regimes (`STATIC`, `SLOW`, `MEDIUM`, `FAST`, `VERY_FAST`) with configurable parameter presets.
  - Read-only active topology presentation ([`ActiveGraphView`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/dynamics/view.py)) and step transition tracking ([`TransitionStatistics`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/dynamics/models.py)).
  - 19 new automated tests (66 total passing).

---

## 5. Target Locomotion & Uncertainty Modeling
- **Status**: **Completed**
- **Scope**:
  - Ground-truth target model ($x_t \in V$) tracking discrete time progression, transition histories, and movement attempts/successes.
  - Target locomotion engine ([`TargetEngine`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/target.py)) evaluating active graph topology $G_t = (V, E_t)$ after dynamic edge churn.
  - Stochastic movement attempts with probability $p_{\text{move}}$, uniform selection among active neighbors, and stationary trapping when isolated or when all incident edges are OFF.
  - Sensor uncertainty engine ([`DetectionEngine`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/detection.py)) with BFS shortest path hop distance in $G_t$ ($\text{dist}_{G_t}(u, x_t) \le s$), infinite distance handling for disconnected targets, detection probability $p_d$, and false alarm probability $p_{\text{fa}}$.
  - Complete four-category detection classification (`POSITIVE`, `MISSED`, `FALSE_POSITIVE`, `NEGATIVE`) and blind-search verification ($p_d = p_{\text{fa}}$).
  - Strict epistemic boundary enforcement preventing leakage of ground-truth target coordinates or internal distance to observer agents.
  - Decoupled, isolated PRNG streams for target mobility and sensor noise.
  - 21 automated unit and statistical tests (87 total passing suite).

---

## 6. Observation & Partial Visibility Layer
- **Status**: **Completed**
- **Scope**:
  - Architectural information firewall projecting unobserved ground-truth state to local walker observations.
  - Observation builder engine ([`ObservationBuilder`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/builder.py)) enforcing partial visibility over active graph $G_t = (V, E_t)$.
  - Neighbor checking budget $B$ limiting candidate inspections with uniform sampling under degree > $B$.
  - Detection signal transformation reducing ground-truth sensor records to binary observable signals (`detected: bool`) with zero outcome or distance leakage.
  - Clear epistemic differentiation between known absent edges and uninspected/unknown nodes.
  - Immutable observation history tracking ([`LocalObservationHistory`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/observation.py)) under snapshot semantics.
  - Decoupled, isolated PRNG stream for observation neighbor sampling.
  - 17 automated tests (104 total passing suite).

---

## 7. Baseline Search Algorithms
- **Status**: **Completed**
- **Scope**:
  - Unified search policy interface ([`BaselinePolicy`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/base.py)) operating strictly on partial-visibility `Observation` snapshots.
  - Action representation ([`SearchAction`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/actions.py)) and observation-bounded validation (`validate_action`).
  - Six benchmark policies:
    - Random Walk ([`RandomWalkPolicy`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/random_walk.py))
    - Non-Backtracking Walk ([`NonBacktrackingWalkPolicy`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/non_backtracking.py))
    - $k$ Independent Random Walkers ([`IndependentRandomWalkers`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/independent_walkers.py))
    - Degree-Based Walk ([`DegreeBasedWalkPolicy`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/degree_based.py)) using locally observed degrees
    - Flooding / Frontier Search ([`FloodingPolicy`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/flooding.py)) with local discovery frontier
    - Ant Colony Walk ([`AntColonyWalkPolicy`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/ant_colony.py)) with private pheromone evaporation and reinforcement
  - Baseline factory dispatcher ([`create_baseline`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/factory.py)) and metadata provenance.
  - Decoupled, isolated PRNG streams and policy reset reproducibility.
  - 46 automated tests (150 total passing suite).

---

## 8. Simulation Environment Coordinator & Walker Orchestration
- **Status**: *Planned*
- **Scope**: Central discrete-time simulation coordinator orchestrating dynamic edge transitions, target movements, traversal delays, multi-walker step scheduling, and action execution.

---

## 9. Proposed AESW Algorithm
- **Status**: *Planned*
- **Scope**: Adaptive Evidence-Sharing Walkers decision-making engine:
  - Local dynamic churn estimation $\hat{C}$
  - Adaptive exploration/exploitation mode gating based on information gain
  - Softmax candidate selection probability distribution

---

## 10. Evidence Memory & Information Sharing
- **Status**: *Planned*
- **Scope**: Evidence cache structures:
  - Dual positive/negative evidence records
  - Node-resident and walker-local cache buffers
  - Adaptive memory decay driven by estimated graph churn

---

## 11. Evaluation Metrics & Benchmark Harness
- **Status**: *Planned*
- **Scope**: Automated benchmark orchestrator computing search time, success rate, unique nodes visited, node revisits, message volume, movement cost, communication cost, and total cost.

---

## 12. Synthetic Graph Experiments
- **Status**: *Planned*
- **Scope**: Systematic benchmarking across Erdős–Rényi, Barabási–Albert, Watts–Strogatz, Grid, and RGG topologies across diverse network sizes and edge densities.

---

## 13. Dynamic Regime Sensitivity Analysis
- **Status**: *Planned*
- **Scope**: Stress testing under varying churn regimes from static topologies up to very fast edge churn.

---

## 14. Noise, Partial Observability, & Uncertainty Sweeps
- **Status**: *Planned*
- **Scope**: Sensitivity analysis across false alarm rates, detection failure rates, and traversal delays.

---

## 15. Real-World Dynamic Network Validation
- **Status**: *Planned*
- **Scope**: Empirical evaluation on real-world Autonomous Systems graph trace series (AS-733) demonstrating performance in realistic non-synthetic dynamic environments.

---

## 16. Statistical Significance Analysis & Hypothesis Testing
- **Status**: *Planned*
- **Scope**: Two-sample t-tests, Mann-Whitney U tests, effect size calculations (Cohen's $d$), and 95% confidence intervals across $\ge 50$ independent stochastic seeds.

---

## 17. Research Visualization Suite
- **Status**: *Planned*
- **Scope**: Generation of publication-quality figures: trajectory traces, success rate curves, cost-benefit trade-offs, and regime response heatmaps.

---

## 18. Comprehensive Manuscript Preparation & Artifact Packaging
- **Status**: *Planned*
- **Scope**: Synthesis of findings into a structured scientific research manuscript and reproducible artifact bundle.
