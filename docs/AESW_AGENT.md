# AESW Agent Architecture & Experiment Harness Integration

## Architectural Overview

This document describes the end-to-end architecture and integration of the proposed method:
**Adaptive Evidence-Sharing Walkers (AESW)**.

It unifies the adaptive search subcomponents into a cohesive, executable multi-agent search framework seamlessly integrated with the evaluation experiment harness.

```
       ┌────────────────────────────────────────────────────────────┐
       │                Environment Coordinator                     │
       │    (Static Topology, Dynamic Edge Churn, Target State)     │
       └─────────────────────────────┬──────────────────────────────┘
                                     │ Local Observation Snapshot
                                     ▼
                      ┌─────────────────────────────┐
                      │          AESWAgent          │
                      │  ┌───────────────────────┐  │
                      │  │ Churn Estimator       │  │
                      │  │   (Online λ_hat)      │  │
                      │  └───────────┬───────────┘  │
                      │              │ λ_hat        │
                      │              ▼              │
                      │  ┌───────────────────────┐  │
                      │  │ Evidence Memory       │  │
                      │  │  w = exp(-λ_hat*age)  │  │
                      │  └───────────┬───────────┘  │
                      │              │              │
                      │  ┌───────────▼───────────┐  │
                      │  │ Mode Controller       │  │
                      │  │  (LOCAL vs LONG_JUMP) │  │
                      │  └───────────┬───────────┘  │
                      │              │              │
                      │  ┌───────────▼───────────┐  │
                      │  │ Next-Hop Decision     │  │
                      │  │  (Candidate Scoring   │  │
                      │  │   & Softmax Selection)│  │
                      │  └───────────────────────┘  │
                      └──────────────┬──────────────┘
                                     │
           Push/Pull Clues           │ SearchAction (MOVE, STAY, or JUMP)
                  ▼                  ▼
    ┌───────────────────────────┐  ┌───────────────────────────────────┐
    │ Node-Mediated             │  │ Environment Coordinator           │
    │ Communication Exchange    │  │   • Validates Action Bounds       │
    │   • SharedNodeCache @ u   │  │   • Executes Mediated Long Jumps  │
    │   • Tracks Q Messages     │  │   • Evaluates C = M + c_m * Q     │
    └───────────────────────────┘  └───────────────────────────────────┘
```

---

## Key Components

### 1. `AESWAgent` ([`src/aesw/aesw/agent.py`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/aesw/agent.py))
An autonomous single-walker cognitive search agent maintaining private internal state:
- **Local Decaying Memory**: [`EvidenceCache`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/memory/cache.py) applying $w(t) = \exp(-\hat{\lambda} \cdot \Delta t)$.
- **Online Churn Estimator**: [`ChurnEstimator`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/aesw/churn.py) computing local Jaccard neighborhood turnover.
- **Adaptive Mode Controller**: [`AdaptiveModeController`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/aesw/mode.py) monitoring stagnation across window $W$.
- **Next-Hop Decision Engine**: [`NextHopDecisionEngine`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/aesw/next_hop.py) computing multi-criteria candidate scores $S(u) = \alpha P(u) + \beta N(u) - \gamma R(u) - \delta D(u)$ and softmax action selection.
- **Visit History**: Private visit frequency dictionary $R(u)$ and novelty tracking $N(u)$.
- **PRNG Stream**: Independent NumPy Generator seeded per walker.

### 2. `AESWPolicy` ([`src/aesw/aesw/agent.py`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/aesw/agent.py))
Multi-walker policy conforming to [`BaselinePolicy`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/baselines/base.py):
- Coordinates $k$ independent `AESWAgent` instances (default $k=4$).
- Houses one vertex-anchored [`NodeMediatedExchange`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/aesw/communication.py) substrate.
- Supports all 4 communication regimes: `NO_SHARING`, `PUSH`, `PULL`, and `PUSH_PULL`.
- Synchronizes message tracking $Q$ and step audit traces.

### 3. Environment-Mediated Long Jump
- When an agent experiences $W$ consecutive uninformative steps, `AdaptiveModeController` switches mode to `SearchMode.LONG_JUMP`.
- The agent emits `SearchAction(action_type=ActionType.JUMP, destination=None)`.
- The simulation coordinator intercepts the jump action, samples a destination node from the graph using its isolated `_jump_rng`, applies traversal delay $d \ge 1$, increments move counter $M$, and updates the walker's physical location without leaking any global or target state to the agent.

---

## Research Integrity & Epistemic Firewall Guarantees

1. **Zero Ground-Truth Leaks**:
   - `AESWAgent` operates strictly on partial-visibility `Observation` snapshots.
   - It possesses no references to `p_on`, `p_off`, true target coordinates, or future graph transitions.
2. **Decentralized Multi-Walker Isolation**:
   - Walkers do not communicate directly point-to-point.
   - All shared evidence is mediated by local `SharedNodeCache` containers.
   - Each walker maintains its own distinct PRNG seed, visit counts, and cognitive models.
3. **Reproducibility & Fairness**:
   - Deterministic execution guaranteed by `np.random.SeedSequence` sub-seed hierarchy.
   - Any `ExperimentInstance` evaluated with `run_experiment(exp, "aesw")` or any baseline (`random_walk`, `non_backtracking`, `k_walk`, `degree`, `flooding`, `ant_colony`) undergoes the exact identical environment realization.
4. **Cost Model Adherence**:
   - Total search cost accurately evaluates $C_{\text{total}} = M + c_m \cdot Q$, charging $c_m$ per node-mediated message.

---

## Verification

The complete AESW agent architecture is verified by dedicated unit and integration tests in [`tests/test_aesw_agent.py`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/tests/test_aesw_agent.py), bringing the total verified regression suite to **346 passed tests (0 failures)**.
