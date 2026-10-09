# Reproducible Experimental Harness & Benchmark Engine

---

## 1. Overview & Research Objective

In dynamic graph search and transport under uncertainty, fair and rigorous empirical evaluation requires that:

> **Every search algorithm must be evaluated on the exact same underlying environment realization whenever algorithms are compared.**

If different algorithms were evaluated on different graph instances, varying initial node assignments, or divergent stochastic edge churn realizations, differences in empirical search cost ($C_{\text{total}}$) or search time ($t$) could stem from environmental variance rather than algorithmic efficacy.

The **Experimental Harness & Reproducible Experiment Engine** decouples:
1. **Experiment Generation** (constructing the immutable environment realization).
2. **Algorithm Execution** (evaluating policies across discrete decision steps).

---

## 2. Experimental Pipeline Architecture

The end-to-end execution pipeline operates as follows:

```
                  YAML Configuration / Defaults
                                │
                                ▼
                           Master Seed
                                │
                                ▼
                       NumPy SeedSequence
                  (Spawns isolated sub-seeds)
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
 Static Graph Generation                     Initial Placements
  G = (V, E) via Factory                   Target x_0, Walkers w_i
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
                       ExperimentInstance
                (Immutable Environment Snapshot)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  Random Walk            Degree-Based              Flooding ...
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                     SimulationCoordinator
            (Discrete-time orchestration, edge churn,
             target moves, noisy sensing, delays)
                                │
                                ▼
                    Structured ExperimentResult
                  (RunMetrics, trajectories, params)
                                │
                                ▼
                          JSON Artifact
```

---

## 3. Mathematical Model & Sub-Seed Isolation

Given a master integer seed $S \in \mathbb{N}$, independent random number generator streams are derived using a deterministic seed sequence tree:

$$\text{SeedSequence}(S) \longrightarrow \{S_{\text{graph}}, S_{\text{dynamics}}, S_{\text{target}}, S_{\text{detection}}, S_{\text{observation}}, S_{\text{init}}, S_{\text{algorithm}}\}$$

Each subsystem operates on its own dedicated PRNG stream:
- $S_{\text{graph}}$: Deterministic topology construction ($G = (V, E)$).
- $S_{\text{dynamics}}$: Synchronous Markovian edge churn ($p_{\text{on}}, p_{\text{off}}$).
- $S_{\text{target}}$: Target locomotion attempts and active neighbor selection ($p_{\text{move}}$).
- $S_{\text{detection}}$: True detection ($p_d$) and false alarm ($p_{\text{fa}}$) sensing rolls.
- $S_{\text{observation}}$: Neighbor inspection budget sampling ($B$).
- $S_{\text{init}}$: Uniform selection of starting vertices $x_0$ and $u_{w, 0}$.
- $S_{\text{algorithm}}$: Stochastic search policy decisions.

This isolation guarantees that algorithmic decisions never contaminate the underlying environmental dynamics or sensing noise.

---

## 4. Separation of Generation and Execution

### A. Experiment Generation: `generate_experiment`

The generator produces an immutable `ExperimentInstance` encapsulating the entire environmental realization:

```python
from aesw.evaluation import generate_experiment

experiment = generate_experiment(
    seed=42,
    graph_type="watts_strogatz",
    num_nodes=500,
    regime="MEDIUM",
    p_move=0.02,
    p_d=0.8,
    p_fa=0.1,
    walker_count=4,
    time_horizon=1000,
)
```

### B. Algorithm Execution: `run_experiment`

Multiple algorithms can be executed against the same `ExperimentInstance`:

```python
from aesw.evaluation import run_experiment

result_rw = run_experiment(experiment, "random_walk")
result_nb = run_experiment(experiment, "non_backtracking")
result_db = run_experiment(experiment, "degree_based")
result_fl = run_experiment(experiment, "flooding")
result_ac = run_experiment(experiment, "ant_colony")
result_irw = run_experiment(experiment, "independent_random_walkers")
```

Every run initiates with:
- Identical initial target location $x_0 = \text{experiment.initial\_target\_node}$.
- Identical initial walker starting locations $u_{w, 0} = \text{experiment.initial\_walker\_nodes}[w]$.
- Identical temporal edge availability transitions $E_1, E_2, \dots, E_T$.
- Identical target movement attempts and choices.

---

## 5. Simulation Coordinator & Epistemic Firewall

The `SimulationCoordinator` orchestrates the discrete-time execution loop while enforcing the **Controlled Observation Principle**:

1. **Information Firewall**:
   - The coordinator encapsulates `GroundTruthState`, `DynamicGraphState`, `TargetEngine`, and `DetectionEngine`.
   - Walkers and policies receive strictly bounded, local, noisy `Observation` snapshots via `coordinator.get_observations()`.
2. **Observation Caching & Idempotence**:
   - Observations are generated and cached once per discrete step $t$.
   - The exact observation snapshot evaluated by a policy is verified when validating the chosen `SearchAction`.
3. **Action Execution & Physical State**:
   - Actions (`MOVE`, `STAY`) are validated via `validate_action(action, obs)`.
   - Traversal delays $d(e, t) \ge 1$ mark walkers as busy until $t + d(e, t)$.
   - Unique nodes visited and repeated visits are tracked across steps.
4. **Target Acquisition & Termination**:
   - `SUCCESS`: Walker arrives at target vertex ($u_{w, t} = x_t$) or target relocates to walker vertex.
   - `BUDGET_EXHAUSTED`: Elapsed simulation time $t \ge T_{\max}$.
   - `DISCONNECTED`: Target becomes completely unreachable in static or non-recovering networks ($p_{\text{on}} = 0$).

---

## 6. Formal Metrics & Statistical Aggregation

### Single-Run Metrics (`RunMetrics`)

- $t_{\text{final}}$: Search time (simulation steps elapsed).
- $M$: Total physical moves executed across all searchers.
- $Q$: Total communication messages exchanged.
- $C_{\text{total}} = M + c_m \cdot Q$: Total evaluated search cost.
- $|V_{\text{visited}}|$: Count of unique vertices inspected.
- $N_{\text{revisit}}$: Count of repeated vertex inspections.

### Multi-Run Summary (`AggregatedMetrics`)

Across $N$ stochastic seeds, empirical metrics are aggregated:
- **Success Rate**: $\hat{p} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\{\text{status}_i = \text{SUCCESS}\}}$
- **Mean & Standard Deviation**: $\bar{C} = \frac{1}{N}\sum C_i$, $s = \sqrt{\frac{1}{N-1}\sum (C_i - \bar{C})^2}$
- **95% Confidence Interval**:
  $$\text{CI}_{95\%} = \left[\bar{C} - t_{0.975, N-1} \cdot \frac{s}{\sqrt{N}}, \quad \bar{C} + t_{0.975, N-1} \cdot \frac{s}{\sqrt{N}}\right]$$

---

## 7. JSON Artifact Schema

Executing `result.save("results")` or `result.to_json("artifact.json")` outputs a machine-readable JSON record:

```json
{
  "experiment_id": "exp_0042_random_walk",
  "algorithm": "random_walk",
  "seed": 42,
  "status": "SUCCESS",
  "success": true,
  "search_time": 47,
  "total_moves": 47,
  "total_messages": 0,
  "total_cost": 47.0,
  "nodes_visited": 29,
  "node_revisits": 18,
  "metrics": {
    "algorithm": "random_walk",
    "seed": 42,
    "status": "SUCCESS",
    "success": true,
    "search_time": 47,
    "nodes_visited": 29,
    "node_revisits": 18,
    "total_moves": 47,
    "total_messages": 0,
    "total_cost": 47.0
  },
  "target_trajectory": [12, 12, 15, 18, ...],
  "walker_trajectories": {
    "0": [4, 7, 11, 15, ...]
  },
  "parameters": {
    "graph": { "type": "watts_strogatz", "num_nodes": 500, "k": 4, "p": 0.1 },
    "dynamics": { "regime": "MEDIUM", "p_on": 0.05, "p_off": 0.025 },
    "target": { "mode": "MOVING", "p_move": 0.02, "initial_node": 12 },
    "detection": { "signal_radius": 1, "p_d": 0.8, "p_fa": 0.1 },
    "walker": { "count": 1, "neighbor_budget": 4, "initial_nodes": { "0": 4 } },
    "cost": { "c_m": 0.1 }
  },
  "timestamp": "2026-10-09T10:45:00.000000+00:00"
}
```

---

## 8. Benchmark Automation: `run_benchmark`

The harness includes a multi-algorithm, multi-seed comparative benchmark dispatcher:

```python
from aesw.evaluation import run_benchmark

benchmark = run_benchmark(
    algorithms=["random_walk", "non_backtracking", "degree_based", "flooding", "ant_colony"],
    seeds=[42, 43, 44, 45, 46],
    output_dir="results/benchmark_01",
    suite_name="baseline_comparison",
    graph_type="watts_strogatz",
    num_nodes=200,
    time_horizon=500,
)
```

Generates structured JSON artifacts for each individual run along with `baseline_comparison_summary.json` containing cross-seed aggregated metrics and confidence intervals.
