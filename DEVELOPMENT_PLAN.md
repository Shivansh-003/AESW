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
- **Status**: *Planned*
- **Scope**: Stochastic discrete-time edge transition engine implementing Markovian $p_{\text{on}}$ and $p_{\text{off}}$ dynamics across defined churn regimes (`STATIC`, `SLOW`, `MEDIUM`, `FAST`, `VERY_FAST`).

---

## 5. Target Locomotion & Uncertainty Modeling
- **Status**: *Planned*
- **Scope**: Target movement dynamics along active edges with transition probability $p_{\text{move}}$, and sensor observation generation subject to signal radius $s$, detection probability $p_d$, and false alarm probability $p_{\text{fa}}$.

---

## 6. Simulation Environment Coordinator & Controlled Observation
- **Status**: *Planned*
- **Scope**: Central discrete-time simulation coordinator orchestrating graph state updates, moving target positions, traversal delays, and enforcing the Controlled Observation Principle across search agents.

---

## 7. Generic Walker Abstractions & Interfaces
- **Status**: *Planned*
- **Scope**: Extensible base walker interface specifying lifecycle hooks (`observe`, `decide_step`, `update_state`, `emit_messages`) and agent state tracking.

---

## 8. Baseline Search Algorithms
- **Status**: *Planned*
- **Scope**: Benchmark search algorithms evaluated under identical environment conditions:
  - Random Walk (RW)
  - Non-Backtracking Walk (NBW)
  - $k$ Independent Random Walkers ($k$-RW)
  - Degree-Based Walk
  - Flooding / Parallel Breadth-First Search
  - Ant Colony Walk

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
