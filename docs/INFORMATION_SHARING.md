# AESW Information Sharing & Node-Mediated Communication

---

## 1. Overview & Research Objective

In multi-agent search over dynamic graphs under uncertainty, individual searchers gather partial observations regarding target proximity and vertex clearance. However, direct peer-to-peer radio transmissions or centralized cloud blackboards are often infeasible in decentralized, bandwidth-constrained, or ad-hoc sensor networks.

The **Adaptive Evidence-Sharing Walkers (AESW)** framework introduces a strictly **node-mediated communication architecture**:

```text
    Walker A
       │
       ▼ (PUSH)
   Node Cache (at vertex u)
       │
       ├──────────► (PULL) Walker B
       │
       └──────────► (PULL) Walker C
```

### Core Architectural Invariants

1. **Strict Node Mediation**: Walkers **never** communicate directly with one another. There is no direct $A \to B$ messaging link.
2. **Spatial Locality**: Evidence deposited at vertex $u$ is stored exclusively in the shared cache anchored at vertex $u$. It does not magically propagate to an unrelated vertex $v \ne u$.
3. **Source Ownership Preservation**: When Walker A deposits evidence, `evidence.walker_id` remains `A`. When Walker B retrieves it, source attribution remains `A`.
4. **Exchange Provenance Tagging**: Exchanged evidence ingested into a walker's local cache is tagged with `EvidenceSource.RECEIVED_EXCHANGE`.
5. **Epistemic Firewall**: Zero transmission of hidden environment parameters ($p_{\text{on}}, p_{\text{off}}$), true target coordinates, or future graph states.
6. **Quantifiable Communication Cost**: Communication is explicitly counted as $Q$ discrete message units, directly feeding the cost model $C_{\text{total}} = M + c_m \cdot Q$.

---

## 2. Communication Modes

The architecture supports exactly four formal communication modes:

| Communication Mode | Push Permitted? | Pull Permitted? | Description & Cost Behavior |
| :--- | :---: | :---: | :--- |
| **`NO_SHARING`** | No | No | Searchers operate in complete epistemic isolation. No evidence is exchanged; message count $Q = 0$. |
| **`PUSH`** | **Yes** | No | Searchers deposit locally gathered evidence into the node cache when visiting a vertex. Other walkers can subsequently retrieve it. |
| **`PULL`** | No | **Yes** | Searchers query and retrieve available shared evidence from the node cache when visiting a vertex. |
| **`PUSH_PULL`** | **Yes** | **Yes** | Searchers both deposit local evidence and query for available shared evidence at visited vertices. |

---

## 3. Communication Cost Model & Accounting Convention

The total search cost model is formally defined as:

$$C_{\text{total}} = M + c_m \cdot Q$$

where:
- $M$: Number of physical movements executed across the graph.
- $Q$: Total number of discrete communication messages exchanged.
- $c_m \ge 0$: Unit communication cost coefficient.

### Explicit Message Counting Convention

To prevent ambiguity between evidence records and network packets:
- **Push Transaction**: A walker depositing $k \ge 1$ evidence records into a node cache constitutes **1 communication message** ($Q \mathrel{+}= 1$). Depositing an empty payload transmits 0 messages.
- **Pull Transaction**: A walker querying a node cache for available shared evidence constitutes **1 communication message** ($Q \mathrel{+}= 1$).
- **Exchange Operation**: In `PUSH_PULL` mode, executing both a push and a pull constitutes **2 communication messages** ($1 + 1 = 2$).
- **`NO_SHARING` Mode**: All communication attempts are no-ops with zero cost ($Q = 0$).

Cumulative message accounting is tracked by `CommunicationMetrics`:
- `total_messages`: Integer $Q$.
- `messages_sent`, `messages_received`: Transaction traffic counters.
- `push_messages`, `pull_messages`: Categorical breakdown.
- `evidence_records_pushed`, `evidence_records_pulled`: Aggregate payload volume.

---

## 4. Software Architecture & API

The communication subsystem resides in `aesw.aesw.communication`:

```python
from aesw.aesw.communication import (
    NodeMediatedExchange,
    CommunicationMode,
    SharedNodeCache,
)
from aesw.memory import EvidenceCache, NodeEvidence

# Initialize communication exchange with PUSH_PULL semantics
exchange = NodeMediatedExchange(mode=CommunicationMode.PUSH_PULL)

# Walker A local cache
cache_a = EvidenceCache()
ev_a = NodeEvidence(
    node_id=42,
    visited=True,
    target_found=True,
    signal_strength=1.0,
    timestamp=5,
    walker_id="Walker_A",
    confidence=1.0,
)
cache_a.store(ev_a)

# Walker A visits vertex 10 and publishes local evidence
pushed = exchange.push_from_cache(walker_id="Walker_A", at_node=10, local_cache=cache_a, timestamp=5)
assert pushed == 1
assert exchange.total_messages == 1

# Walker B later visits vertex 10 and pulls available evidence
cache_b = EvidenceCache()
pulled = exchange.pull(walker_id="Walker_B", at_node=10, timestamp=8, target_cache=cache_b)
assert len(pulled) == 1
assert exchange.total_messages == 2

# Walker B now holds Walker A's evidence tagged as RECEIVED_EXCHANGE
received_record = cache_b.get_raw(42)
assert received_record.walker_id == "Walker_A"  # Creator preserved!
assert received_record.source.value == "RECEIVED_EXCHANGE"
```

---

## 5. Duplicate Handling & Memory Bounds

Searchers frequently revisit high-degree communication hubs. When a searcher repeatedly pulls from the same vertex cache:
1. `EvidenceCache` enforces deterministic update precedence: existing entries are only overwritten if the incoming record has a strictly newer timestamp ($t_{\text{new}} > t_{\text{old}}$) or higher confidence.
2. Identical evidence received multiple times does **not** create duplicate records or cause memory explosion.
3. Message accounting $Q$ continues to track queries accurately.

---

## 6. Integration with Churn Estimation & Decay

When Walker B retrieves evidence originally gathered by Walker A at time $t_{\text{rec}}$:
- The evidence age is evaluated at Walker B's current time $t$: $\text{age} = t - t_{\text{rec}}$.
- Walker B evaluates the evidence weight using **Walker B's own current churn estimate** $\hat{\lambda}_B$ (from its local `ChurnEstimator`):

$$w_B(t) = \exp\left(-\hat{\lambda}_B \cdot (t - t_{\text{rec}})\right)$$

This ensures that evidence interpretation adapts dynamically to the receiver's local perception of environmental volatility.
