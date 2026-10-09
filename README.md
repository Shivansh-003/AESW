# Adaptive Search and Transport on Dynamic Graphs Under Uncertainty

A research-grade simulation framework for decentralized target search and transport on dynamic graphs subject to partial observability, stochastic edge transitions, and sensing noise.

---

## 1. Research Motivation & Problem Overview

Real-world physical, biological, and communication networks—including opportunistic mobile sensor networks, peer-to-peer routing meshes, and autonomous search-and-rescue grids—exhibit intrinsic non-stationarity:

- **Dynamic Edge Churn**: Communication and traversal links activate and deactivate dynamically under stochastic transitions ($p_{\text{on}}, p_{\text{off}}$).
- **Mobile Targets**: Targets wander across active edges with transition probability $p_{\text{move}}$ rather than remaining stationary.
- **Traversal Delays**: Traversing active edges incurs variable discrete durations ($d \ge 1$), introducing latency.
- **Partial Observability**: Search agents possess no global network map; they observe only their immediate $1$-hop local neighborhood bounded by a neighbor inspection budget $B$.
- **Noisy Sensor Readings**: Target sensing is imperfect, corrupted by detection failures ($1 - p_d$) within signal radius $s$ and false alarms ($p_{\text{fa}}$) at distant nodes.
- **Decentralized Multi-Agent Coordination**: Multiple searchers explore simultaneously without centralized coordinators.

---

## 2. Proposed Method: Adaptive Evidence-Sharing Walkers (AESW)

To overcome the inefficiencies of blind random walks (excessive node revisits) and static heuristic walks (vulnerability to topological churn), the framework implements **Adaptive Evidence-Sharing Walkers (AESW)**.

### Core Architectural Pillars
1. **Adaptive Memory Decay Driven by Graph Churn**:
   Rather than relying on static temporal horizons, walkers estimate localized edge turnover $\hat{C}$ and dynamically adjust memory decay rates. Rapid topological churn accelerates decay to discard obsolete observations, whereas quasi-static conditions retain long-term structural knowledge.
2. **Adaptive Local Exploitation vs. Global Exploration**:
   Walkers autonomously transition between directed local gradient descent/exploitation (concentrating search near high-confidence cues) and dispersed global exploration (expanding spatial coverage) based on estimated information gain.
3. **Dual Evidence Representation & Sharing (Positive + Negative Clues)**:
   Walkers broadcast and cache both positive evidence (sensor detections indicating target proximity) and negative evidence (confirmed absent subgraphs), preventing redundant exploration across the search team.

---

## 3. Supported Graph Topologies

The framework provides deterministic synthetic graph generation:

- **Erdős–Rényi ($G(n, p)$)**: Random graphs with independent edge probabilities.
- **Barabási–Albert ($\text{BA}(n, m)$)**: Scale-free topologies generated via preferential attachment.
- **Watts–Strogatz ($\text{WS}(n, k, p)$)**: Small-world networks with high clustering and short path lengths.
- **2D Spatial Grid with Obstacles**: Planar lattices with 4-neighbor orthogonal connectivity, coordinate preservation, and obstacle masking.
- **Random Geometric Graphs (RGG)**: Spatial networks connecting nodes within Euclidean radius $r$ in $[0, 1]^2$.
- **Autonomous Systems (AS-733)**: Architecture incorporates a clean adapter extension point for real-world temporal network traces.

---

## 4. Repository Structure

```text
adaptive-graph-search/
├── pyproject.toml              # Build system, package metadata, and test paths
├── requirements.txt            # Core research dependencies
├── .gitignore                  # Git exclusions for artifacts and cache directories
├── README.md                   # Public project overview, setup, and documentation
├── PROJECT_SPEC.md             # Formal research problem specification
├── ARCHITECTURE.md             # System layer hierarchy and interface contracts
├── DEVELOPMENT_PLAN.md         # Implementation roadmap
│
├── configs/                    # Declarative YAML experiment configurations
│   ├── default.yaml            # Project metadata, reproducibility seed, logging
│   ├── graph.yaml              # Topology generation parameters
│   ├── dynamics.yaml           # Dynamic edge transition parameters & regimes
│   └── experiments.yaml        # Benchmark and evaluation parameters
│
├── docs/                       # Detailed technical documentation
│   ├── PROBLEM_DEFINITION.md   # Mathematical equations and formal problem contract
│   ├── GRAPH_GENERATION.md     # Graph family models, invariants, and metadata
│   ├── DYNAMIC_GRAPH.md        # Edge state transitions, regimes, and active graph views
│   ├── TARGET_AND_UNCERTAINTY.md # Target mobility, sensing radius, and detection noise
│   ├── OBSERVATION_AND_PARTIAL_VISIBILITY.md # Firewall, budget B, and local history
│   ├── BASELINE_ALGORITHMS.md  # Benchmark search policies, fairness, and action model
│   ├── EXPERIMENTAL_HARNESS.md # Experiment engine, generation vs execution, and artifacts
│   ├── EVIDENCE_MEMORY.md      # Adaptive exponential decay and local evidence cache
│   ├── CHURN_ESTIMATOR.md      # Local online churn estimation and volatility inference
│   ├── INFORMATION_SHARING.md  # Decentralized node-mediated information sharing
│   ├── MODE_CONTROLLER.md      # Adaptive mode controller (LOCAL vs LONG_JUMP)
│   └── NEXT_HOP_DECISION.md    # Multi-factor candidate scoring and softmax action selection
│
├── src/
│   └── aesw/                   # Core research package
│       ├── __init__.py         # Package root
│       ├── graph/              # Topology generation & GraphInstance abstraction
│       │   ├── types.py        # GraphType enumeration
│       │   ├── metadata.py     # GraphMetadata and TopologyStatistics dataclasses
│       │   ├── base.py         # GraphInstance abstraction wrapping NetworkX
│       │   ├── validation.py   # Parameter validation and invariant assertions
│       │   ├── generators.py   # Concrete deterministic graph generators
│       │   ├── factory.py      # Dispatcher and generator factories
│       │   └── adapters.py     # Static and temporal graph interfaces
│       ├── dynamics/           # Dynamic edge transition engine
│       │   ├── types.py        # EdgeState, DynamicRegime, EdgeInitPolicy enums
│       │   ├── models.py       # TransitionStatistics and snapshot records
│       │   ├── engine.py       # DynamicGraphState Markovian transition engine
│       │   ├── view.py         # ActiveGraphView read-only dynamic presentation
│       │   └── factory.py      # Dynamic graph instantiation factory
│       ├── environment/        # Simulation coordinator & observation boundary
│       │   ├── types.py        # Enums (EdgeState, TargetMode, DetectionOutcome)
│       │   ├── models.py       # Data models (Node, Edge, TargetState, WalkerState)
│       │   ├── state.py        # GroundTruthState unobserved representation
│       │   ├── observation.py  # Local, noisy Observation model & history
│       │   ├── problem.py      # Immutable ProblemDefinition contract
│       │   ├── target.py       # TargetEngine locomotion across active edges
│       │   ├── detection.py    # DetectionEngine sensor uncertainty model
│       │   ├── builder.py      # ObservationBuilder partial visibility firewall
│       │   └── coordinator.py  # SimulationCoordinator discrete-time orchestrator
│       ├── walkers/            # Generic walker interfaces and tracking
│       ├── baselines/          # Benchmark search algorithms
│       │   ├── types.py        # BaselineType and ActionType enums
│       │   ├── actions.py      # SearchAction and observation validation
│       │   ├── base.py         # BaselinePolicy abstract base and metadata
│       │   ├── random_walk.py  # Unbiased RandomWalkPolicy
│       │   ├── non_backtracking.py # 1-step memory NonBacktrackingWalkPolicy
│       │   ├── independent_walkers.py # k IndependentRandomWalkers
│       │   ├── degree_based.py # Locally observed DegreeBasedWalkPolicy
│       │   ├── flooding.py     # Local frontier FloodingPolicy
│       │   ├── ant_colony.py   # Private pheromone AntColonyWalkPolicy
│       │   └── factory.py      # Baseline factory dispatcher
│       ├── aesw/               # Proposed AESW search algorithm
│       │   ├── churn.py        # Local online ChurnEstimator and Jaccard overlap
│       │   ├── communication.py # Node-mediated SharedNodeCache & NodeMediatedExchange
│       │   ├── mode.py         # AdaptiveModeController and SearchMode gating
│       │   └── next_hop.py     # Multi-criteria scoring, softmax selection, NextHopDecisionEngine
│       ├── memory/             # Evidence caching and decay mechanisms
│       │   ├── types.py        # EvidencePolarity and EvidenceSource enums
│       │   ├── models.py       # NodeEvidence and WeightedEvidence dataclasses
│       │   ├── decay.py        # Adaptive exponential decay and half-life formulations
│       │   └── cache.py        # Local EvidenceCache container and update rules
│       ├── evaluation/         # Performance metrics, harness & benchmark runner
│       │   ├── metrics.py      # RunMetrics, AggregatedMetrics, and CI computation
│       │   └── experiment.py   # ExperimentInstance, run_experiment, run_benchmark
│       └── utils/              # Configuration loaders and isolated RNG helpers
│           ├── config.py
│           └── reproducibility.py
│
├── experiments/                # Experiment runner scripts and benchmarks
├── scripts/                    # Utility scripts and dataset helpers
├── tests/                      # Automated test suite
│   ├── test_foundation.py      # Configuration and reproducibility tests
│   ├── test_problem_definition.py # Formal contracts, state, and observation tests
│   ├── test_graph_generators.py   # Graph generators, invariants, and determinism tests
│   ├── test_dynamic_graph.py   # Markovian edge transitions and dynamic regimes
│   ├── test_target_and_uncertainty.py # Target locomotion and detection noise tests
│   ├── test_observation.py     # Partial visibility, budget B, and firewall tests
│   ├── test_baselines.py       # Benchmark search policy tests and action validation
│   ├── test_experiment_engine.py # Experimental harness, coordination, and benchmark tests
│   ├── test_evidence_memory.py # Node evidence, exponential decay, and cache tests
│   ├── test_churn_estimator.py # Online churn estimation and scientific coupling tests
│   ├── test_information_sharing.py # Node-mediated information sharing & cost tests
│   ├── test_mode_controller.py # Adaptive mode controller (LOCAL vs LONG_JUMP) tests
│   └── test_next_hop_decision.py # Candidate scoring, softmax selection, and next-hop decision tests

├── results/                    # Output directory for simulation data (.gitkeep)
└── plots/                      # Output directory for generated figures (.gitkeep)

```

---

## 5. Technology Stack

- **Python**: $\ge 3.10$
- **PyYAML** ($\ge 6.0.1$): Configuration parsing
- **NetworkX** ($\ge 3.2$): Internal graph data structures
- **NumPy** ($\ge 1.26.0$): Numerical computation and isolated random generators
- **SciPy** ($\ge 1.12.0$): Statistical hypothesis testing and distributions
- **Matplotlib** ($\ge 3.8.0$): Research visualizations and figure generation
- **Pytest** ($\ge 8.0.0$): Automated unit testing framework

---

## 6. Installation & Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd adaptive-graph-search
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 7. Usage & Graph Generation

The graph generation subsystem generates deterministic graph topologies directly from configuration or via Python API:

```python
from aesw.graph import generate_graph_by_family, GraphType

# Generate an Erdős–Rényi random graph
g_er = generate_graph_by_family(
    graph_type=GraphType.ERDOS_RENYI,
    num_nodes=50,
    seed=42,
    parameters={"p": 0.1},
)
print(f"Nodes: {g_er.node_count}, Edges: {g_er.edge_count}")

# Generate a 2D grid with obstacles
g_grid = generate_graph_by_family(
    graph_type=GraphType.GRID_OBSTACLE,
    seed=42,
    parameters={"rows": 10, "cols": 10, "obstacle_ratio": 0.15},
)
stats = g_grid.compute_statistics()
print(f"Grid Average Degree: {stats.average_degree:.2f}, Components: {stats.number_of_connected_components}")
```

---

## 8. Reproducibility Guarantee

Experimental reproducibility is maintained through strict isolation of random number generators:
$$\text{Configuration} + \text{Graph Family} + \text{Parameters} + \text{Seed} \implies \text{Exact Topology Realization}$$

- Every stochastic component accepts an explicit integer seed.
- Randomness is managed via isolated NumPy `Generator` objects created using `create_rng(seed)`.
- Global random state is not modified, ensuring reproducibility across repeated executions.

---

## 9. Testing

Run the automated test suite:

```bash
python -m pytest -q
```

All 320 tests verify:
- Package initialization and clean imports.
- Configuration loading, schema parsing, and error handling.
- Deterministic random number generation.
- Data models, immutable `ProblemDefinition`, and ground-truth vs. observation boundaries.
- Topological invariants and parameter validations across all supported graph families.
- Stochastic discrete-time edge transitions, synchronous updates, initialization policies, and dynamic regimes.
- Ground-truth target locomotion across active edges, trapped target conditions, and mobility probability ($p_{\text{move}}$).
- Probabilistic target sensing ($s, p_d, p_{\text{fa}}$), dynamic shortest-path hop distances, disconnected target handling, and blind-search conditions ($p_d = p_{\text{fa}}$).
- Observation firewall (`ObservationBuilder`), neighbor checking budget $B$, binary detection reduction, local history snapshots, and known absent vs unknown status distinction.
- Six benchmark search policies (Random Walk, Non-Backtracking, $k$ Independent Walkers, Degree-Based, Flooding, Ant Colony) operating strictly over partial observations.
- Action validation preventing movement to unobserved nodes or across inactive links.
- Decoupled, isolated PRNG streams across graph generation, dynamic churn, target mobility, sensor noise, observation sampling, and policy decisions.
- Discrete-time simulation coordination (`SimulationCoordinator`), move tracking, visit/revisit counters, and termination detection (`SUCCESS`, `BUDGET_EXHAUSTED`, `DISCONNECTED`).
- Experimental harness (`generate_experiment`, `run_experiment`, `run_benchmark`) guaranteeing fair evaluation of multiple algorithms on identical environment realizations.
- Statistical metrics aggregation (`compute_aggregated_metrics`) with 95% Student's t confidence intervals and structured JSON artifact persistence.
- AESW evidence memory models (`NodeEvidence`, `WeightedEvidence`), adaptive exponential decay ($w = \exp(-\hat{\lambda} \cdot \text{age})$), characteristic half-life, dual positive/negative evidence polarities, local cache management (`EvidenceCache`), update rules, and observation ingestion.
- AESW online churn estimation (`ChurnEstimator`), local Jaccard neighborhood overlap ($s$), numerical stability floors, exponential smoothing ($\hat{\lambda}_t$), and dynamic coupling to evidence memory decay.
- AESW node-mediated information sharing (`NodeMediatedExchange`, `SharedNodeCache`), communication modes (`NO_SHARING`, `PUSH`, `PULL`, `PUSH_PULL`), and exact communication cost accounting ($Q$).
- AESW adaptive mode controller (`AdaptiveModeController`), $W$-step sliding information history, stagnation counter ($S(t)$), and dynamic switching between `SearchMode.LOCAL` and `SearchMode.LONG_JUMP`.
- AESW next-hop decision engine (`NextHopDecisionEngine`), multi-factor candidate scoring ($S = aP + bN - gR - dD$), numerically stabilized softmax action selection ($P_{\text{select}}(u) = \exp((S(u) - S_{\max})/T) / Z$), and explainable decision records (`CandidateScore`, `NextHopDecision`).

---

## 10. Research Status

Current implementation includes:
- **Reproducible Foundation**: Configuration loader, isolated RNG utilities, and package structure.
- **Formal Problem Contract**: Mathematical definitions, immutable `ProblemDefinition`, data models, and observation boundaries.
- **Graph Generation Engine**: Deterministic generators for Erdős–Rényi, Barabási–Albert, Watts–Strogatz, 2D Grid with Obstacles, and Random Geometric Graphs, along with structural statistics and invariant validation.
- **Dynamic Graph Engine**: Markovian edge ON/OFF transition models ($p_{\text{on}}, p_{\text{off}}$), synchronous updates, dynamic regimes (`STATIC` through `VERY_FAST`), and active graph views ($G_t = (V, E_t)$).
- **Target & Uncertainty Engine**: Hidden target locomotion ($p_{\text{move}}$) over dynamic active topologies, trapped target handling, radius sensing ($s$), probabilistic sensor noise ($p_d, p_{\text{fa}}$), epistemic boundary enforcement, and isolated random streams.
- **Observation & Partial Visibility Layer**: Architectural information firewall, `ObservationBuilder`, neighbor checking budget $B$, binary signal reduction, local history snapshots, and absence vs unknown distinction.
- **Baseline Search Algorithms**: Six standardized search policies (Random Walk, Non-Backtracking, $k$-RW, Degree-Based, Flooding, Ant Colony) with observation-bounded action validation and isolated PRNG streams.
- **Experimental Harness & Simulation Coordinator**: Discrete-time simulation coordinator, immutable `ExperimentInstance` generation, exact fair comparison across all baselines, cross-seed benchmark automation, and structured JSON artifact generation.
- **AESW Evidence Memory & Adaptive Exponential Decay**: Structured evidence representation, exponential decay weighting ($w = \exp(-\hat{\lambda} \cdot \text{age})$), half-life formulations, positive and negative evidence polarities, local node-level memory cache (`EvidenceCache`), timestamp-prioritized update rules, and observation ingestion.
- **AESW Online Churn Estimator**: Local observation-based Jaccard overlap, instantaneous turnover estimation, smoothed dynamic graph volatility $\hat{\lambda}$, zero leakage of hidden transition parameters, and seamless parameterization of evidence memory decay.
- **AESW Node-Mediated Communication**: Decentralized vertex-anchored shared caches, strict absence of direct peer-to-peer links, formal communication modes, received evidence provenance tracking, and rigorous communication cost accounting ($Q$).
- **AESW Adaptive Mode Controller**: Information-gain driven transitions between local exploitation (`LOCAL`) and global dispersion (`LONG_JUMP`), bounded sliding window $W$, consecutive stagnation tracking, and automatic recovery upon novel positive cues.
- **AESW Next-Hop Decision Engine**: Multi-criteria candidate scoring trading off positive evidence ($P$), novelty ($N$), revisit penalty ($R$), and traversal delay ($D$); numerically stabilized Boltzmann softmax action selection; observation candidate pruning; mode integration; and isolated PRNG sampling.



