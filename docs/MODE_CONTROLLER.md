# AESW Adaptive Mode Controller

---

## 1. Overview & Research Objective

In dynamic graph search and transport under uncertainty, searchers often face a fundamental trade-off between **local exploitation** (searching intensively around recent clues) and **global dispersion** (escaping barren or depleted subgraphs). Traditional random or heuristic search methods either:
1. Suffer from extreme over-exploration (blind random walks constantly revisiting the same cluster), or
2. Enforce rigid, predetermined jump schedules (e.g. jumping every fixed $N$ steps regardless of clue availability).

The **Adaptive Evidence-Sharing Walkers (AESW)** framework introduces an **Adaptive Mode Controller** that dynamically toggles between:

- **`SearchMode.LOCAL`**: Focused local exploration concentrating search near high-confidence cues.
- **`SearchMode.LONG_JUMP`**: Dispersive search triggered adaptively when search progress stagnates.

### Core Architectural Principle

The mode transition is **adaptive to information gain**, never governed by a static time schedule:

```text
             Recent Information
                    │
                    ▼
        Has useful information appeared?
                 /       \
               YES        NO
                │          │
                ▼          ▼
              LOCAL     Check W-step window
                           stagnation ≥ W?
                             │          │
                            YES         NO
                             │          │
                             ▼          ▼
                         LONG_JUMP    LOCAL
```

---

## 2. Mathematical Formulation & Stagnation Tracking

### Recent-Information Window $W$

The controller maintains a bounded sliding horizon of the most recent $W \ge 1$ decision steps:

$$\mathcal{H}_W(t) = [I(t - W + 1), \, \dots, \, I(t)], \quad I(\tau) \in \{0, 1\}$$

where:
- $I(\tau) = 1$: Useful new information was acquired at step $\tau$.
- $I(\tau) = 0$: No useful information was acquired at step $\tau$.

### Stagnation Counter $S(t)$

Let $S(t)$ represent the number of consecutive decision steps elapsed without useful new information:

$$S(t) = \begin{cases} 0 & \text{if } I(t) = 1 \\ S(t-1) + 1 & \text{if } I(t) = 0 \end{cases}$$

### Mode Transition Rules

$$\text{Mode}(t) = \begin{cases} \text{LOCAL} & \text{if } I(t) = 1 \text{ or } S(t) < W \\ \text{LONG\_JUMP} & \text{if } S(t) \ge W \end{cases}$$

1. **Initial State**: $t = 0 \implies \text{Mode} = \text{LOCAL}, \quad S = 0$.
2. **Local Retention**: As long as $S(t) < W$, the walker remains in `LOCAL` mode.
3. **Stagnation-Triggered Transition**: When exactly $W$ stagnant steps elapse without useful information ($S(t) \ge W$), the controller transitions to `LONG_JUMP`.
4. **Cue-Triggered Recovery**: The moment useful positive evidence arrives ($I(t) = 1$), stagnation resets to $S(t) = 0$ and mode returns immediately to `LOCAL`.

---

## 3. Definition of "Useful Information"

The controller evaluates information availability strictly from the walker's legitimate observation and memory state:

| Evidence Type | Qualifies as Useful ($I(t)=1$)? | Rationale |
| :--- | :---: | :--- |
| **Positive Sensor Detection** | **YES** | Sensor detection ($\text{signal} > 0$ or `detected=True`) provides strong proximity clues, even if noisy. |
| **Positive Node Evidence** | **YES** | Records with `EvidencePolarity.POSITIVE` indicate target cues. |
| **Legitimately Received Clue** | **YES** | Positive clues received through node-cache exchange (`RECEIVED_EXCHANGE`) count as real information. |
| **Repeated / Stale Evidence** | **NO** | Re-observing the exact same record with identical timestamp and signal does **not** reset stagnation. |
| **Negative Evidence** | **NO** | Ordinary absence confirmation (empty node) does not indicate target proximity and does not reset stagnation. |
| **Empty Payload** | **NO** | No information acquired ($I(t) = 0$). |

---

## 4. Scope Boundary & Epistemic Separation

- **Mode State Only**: The controller's sole responsibility is deciding the current operational mode (`LOCAL` vs `LONG_JUMP`).
- **No Destination Selection**: The controller does **not** select destination nodes, compute candidate next-hop scores, rank vertices, or execute teleportations/movements (reserved for downstream decision engines).
- **Epistemic Firewall**: Zero access to ground-truth target position $x_t$, true target mobility, $p_{\text{on}}$, $p_{\text{off}}$, `GroundTruthState`, or future graph transitions.

---

## 5. Software Architecture & API

Located in `aesw.aesw.mode`:

```python
from aesw.aesw.mode import AdaptiveModeController, SearchMode
from aesw.memory import NodeEvidence

# Initialize controller with configurable horizon W
controller = AdaptiveModeController(window_size=5)

assert controller.mode == SearchMode.LOCAL
assert controller.stagnation_steps == 0

# Step 1-4: No useful cues acquired
for t in range(1, 5):
    controller.update(timestamp=t, useful=False)
assert controller.mode == SearchMode.LOCAL  # 4 < 5 steps

# Step 5: Exactly W stagnant steps -> transitions to LONG_JUMP
controller.update(timestamp=5, useful=False)
assert controller.mode == SearchMode.LONG_JUMP
assert controller.transition_count == 1

# Step 6: Positive target cue arrives -> returns to LOCAL
cue = NodeEvidence(node_id=42, visited=True, target_found=True, signal_strength=1.0, timestamp=6, walker_id="w1", confidence=1.0)
controller.update(timestamp=6, evidence=cue)
assert controller.mode == SearchMode.LOCAL
assert controller.stagnation_steps == 0
assert controller.transition_count == 2
```

---

## 6. Integration Across Subsystems

- **Evidence Memory**: Consumes `NodeEvidence` and `EvidencePolarity` without duplicate data structures.
- **Churn Estimator**: Exposes `lambda_hat` property and accepts churn rate updates for diagnostics.
- **Information Sharing**: Processes evidence received via `NodeMediatedExchange`. When shared positive cues arrive at a visited node, they reset walker stagnation and maintain `LOCAL` exploration.
