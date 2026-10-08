# Baseline Search Algorithms

---

## 1. Overview & Research Objective

To rigorously evaluate the proposed **Adaptive Evidence-Sharing Walkers (AESW)** method, this repository implements six standard search algorithms as empirical baselines:

1. **Random Walk** (Unbiased stochastic benchmark)
2. **Non-Backtracking Walk** (1-step memory avoiding immediate reversal)
3. **$k$ Independent Random Walkers** (Multi-agent scaling without communication)
4. **Degree-Based Walk** (Topology-biased walk using locally observed degrees)
5. **Flooding / Frontier Search** (Aggressive local frontier discovery)
6. **Ant Colony Walk** (Pheromone-guided local adaptive exploration)

### The Fundamental Fairness Constraint
Every baseline operates under the identical partial-observability contract established by the observation layer:
- Algorithms consume strictly the [`Observation`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/environment/observation.py#L118-L230) snapshot.
- No baseline receives the global graph topology, target coordinates, exact target distance, ground-truth detection classifications, or future transition schedules.
- All stochastic choices utilize isolated NumPy PRNG streams.

---

## 2. Baseline Fairness Matrix

| Baseline Algorithm | Local Observation | Memory Maintained | Inter-Agent Communication | Global Graph Access |
| :--- | :--- | :--- | :--- | :--- |
| **Random Walk** | Yes | None (memoryless) | No | No |
| **Non-Backtracking** | Yes | 1-step (preceding node) | No | No |
| **Independent RW ($k$)** | Yes | Per-walker isolated | No | No |
| **Degree-Based** | Yes | Locally observed degrees | No | No |
| **Flooding** | Yes | Discovered/visited frontier | No (baseline policy) | No |
| **Ant Colony** | Yes | Private pheromone trails | No | No |
| **AESW** *(Future)* | Yes | Adaptive evidence cache | Shared node caches | No |

*Note: No performance superiority is claimed. Baselines establish controlled benchmarks for formal comparative analysis.*

---

## 3. Detailed Baseline Specifications

### Baseline 1: Random Walk (`RandomWalkPolicy`)
- **Motivation**: Classical unbiased diffusion process providing the theoretical lower bound for unguided search on graphs.
- **Mathematical Decision Rule**:
  Let $u$ be the current node and $\mathcal{N}_{\text{obs}}(u)$ be the set of currently observed active neighbors.
  $$\Pr(X_{t+1} = v \mid v \in \mathcal{N}_{\text{obs}}(u)) = \frac{1}{|\mathcal{N}_{\text{obs}}(u)|}$$
  If $\mathcal{N}_{\text{obs}}(u) = \emptyset$, the walker stays: $X_{t+1} = u$.
- **Information Available**: Immediate $1$-hop active neighbors in current observation.
- **Memory Used**: None (strictly Markovian / memoryless).
- **Randomness**: Uniform discrete distribution via isolated PRNG.
- **Limitations**: Excessive node revisits, high cover times, vulnerability to topological traps.
- **Implementation Assumptions**: Candidate list is deterministically sorted prior to indexing to ensure platform-independent RNG reproducibility.

---

### Baseline 2: Non-Backtracking Walk (`NonBacktrackingWalkPolicy`)
- **Motivation**: Suppresses immediate $2$-hop cycles (ping-pong backtracking), accelerating network exploration and graph expansion.
- **Mathematical Decision Rule**:
  Let $p = X_{t-1}$ be the previous node. Define non-reversing candidates:
  $$\mathcal{N}' = \mathcal{N}_{\text{obs}}(u) \setminus \{p\}$$
  $$\Pr(X_{t+1} = v) = \begin{cases} \frac{1}{|\mathcal{N}'|} & \text{if } \mathcal{N}' \ne \emptyset \text{ and } v \in \mathcal{N}' \\ \frac{1}{|\mathcal{N}_{\text{obs}}(u)|} & \text{if } \mathcal{N}' = \emptyset \text{ and } v \in \mathcal{N}_{\text{obs}}(u) \end{cases}$$
- **Information Available**: Current observation and immediate predecessor $p$.
- **Memory Used**: Strictly 1-step memory ($p \in V \cup \{\text{None}\}$).
- **Randomness**: Uniform choice over $\mathcal{N}'$ (or fallback $\mathcal{N}_{\text{obs}}(u)$).
- **Limitations**: Does not prevent higher-order loops ($k \ge 3$) or dead-end entrapment under topological churn.

---

### Baseline 3: $k$ Independent Random Walkers (`IndependentRandomWalkers`)
- **Motivation**: Assesses the pure resource-scaling hypothesis: does deploying multiple searchers without coordination or communication sufficiently reduce search time?
- **Mathematical Decision Rule**:
  Team of $k \in \{1, 2, 4, 8, \dots\}$ searchers where each walker $i \in \{1, \dots, k\}$ independently runs `RandomWalkPolicy` with an isolated seed:
  $$S_i = S_{\text{base}} + i \cdot 1000$$
- **Information Available**: Per-walker local observation.
- **Memory Used**: None per walker.
- **Randomness**: $k$ independent, orthogonal PRNG streams.
- **Limitations**: High exploration redundancy due to zero coordination; searchers frequently overlap paths.

---

### Baseline 4: Degree-Based Walk (`DegreeBasedWalkPolicy`)
- **Motivation**: Preferential exploration toward topological hubs, exploiting structural heterogeneity in scale-free networks.
- **Observable Degree Formulation**:
  Under partial observability, global degrees are unknown. The walker maintains a local cache of degrees observed during physical visits:
  $$d_{\text{obs}}(w) = \begin{cases} |\mathcal{N}_{\text{obs}}(w)| & \text{if } w \text{ has been occupied} \\ 1.0 & \text{otherwise (unvisited prior)} \end{cases}$$
- **Mathematical Decision Rule**:
  $$\text{score}(v) = d_{\text{obs}}(v) + \epsilon, \quad \epsilon = 10^{-3}$$
  $$\Pr(X_{t+1} = v) = \frac{\text{score}(v)}{\sum_{w \in \mathcal{N}_{\text{obs}}(u)} \text{score}(w)}$$
- **Information Available**: Current observation plus historical locally observed degrees.
- **Memory Used**: Mapping of visited node identifiers to observed active degrees.
- **Randomness**: Proportional roulette sampling via isolated PRNG.
- **Limitations**: Hub attraction can trap searchers in high-degree cycles when edge churn disconnects outward links.

---

### Baseline 5: Flooding / Frontier Search (`FloodingPolicy`)
- **Motivation**: Aggressive breadth-first frontier expansion that systematically prioritizes newly discovered, unvisited vertices.
- **Mathematical Decision Rule**:
  Maintains physically visited set $V_{\text{visited}}$ and locally discovered set $V_{\text{disc}}$:
  $$V_{\text{visited}} \leftarrow V_{\text{visited}} \cup \{u\}, \quad V_{\text{disc}} \leftarrow V_{\text{disc}} \cup \mathcal{N}_{\text{obs}}(u)$$
  $$\mathcal{N}_{\text{unvisited}} = \mathcal{N}_{\text{obs}}(u) \setminus V_{\text{visited}}$$
  $$\Pr(X_{t+1} = v) = \begin{cases} \frac{1}{|\mathcal{N}_{\text{unvisited}}|} & \text{if } \mathcal{N}_{\text{unvisited}} \ne \emptyset \text{ and } v \in \mathcal{N}_{\text{unvisited}} \\ \frac{1}{|\mathcal{N}_{\text{obs}}(u)|} & \text{otherwise (fallback)} \end{cases}$$
- **Information Available**: Current observation and local visited/discovered history.
- **Memory Used**: Sets $V_{\text{visited}}$ and $V_{\text{disc}}$.
- **Randomness**: Uniform selection over unvisited frontier candidates.
- **Limitations**: In dynamic networks, previously recorded paths to frontier nodes frequently deactivate, causing dead-end delays.

---

### Baseline 6: Ant Colony Walk (`AntColonyWalkPolicy`)
- **Motivation**: Classic reinforcement learning baseline utilizing artificial pheromone trails with evaporation.
- **Pheromone Dynamics**:
  For directed link $(u, v)$, pheromone $\tau(u, v) \ge \tau_{\text{min}} = 10^{-4}$ (initialized to $\tau_0 = 1.0$):
  - **Evaporation**: $\tau(e) \leftarrow \max(\tau_{\text{min}}, (1 - \rho) \tau(e))$, with $\rho = 0.05$.
  - **Reinforcement**: Upon traversal, $\tau(u, v) \leftarrow \tau(u, v) + \Delta \tau$, with $\Delta \tau = 1.0$.
  - **Optional Signal Reinforcement**: If $\sigma_t(u) = 1$, adds $\Delta \tau_{\text{signal}}$.
- **Mathematical Decision Rule**:
  $$\text{score}(v) = [\tau(u, v)]^\alpha, \quad \alpha = 1.0$$
  $$\Pr(X_{t+1} = v) = \frac{\text{score}(v)}{\sum_{w \in \mathcal{N}_{\text{obs}}(u)} \text{score}(w)}$$
- **Information Available**: Current observation and private pheromone table.
- **Memory Used**: Mapping of directed edge tuples to pheromone scalars.
- **Randomness**: Stochastic selection weighted by pheromone concentration.
- **Distinction from AESW**: Pheromone is purely scalar edge reinforcement with constant evaporation $\rho$; it does **not** estimate topological churn $\hat{C}$, does not maintain positive/negative evidence caches, does not perform mode switching, and does not broadcast information across walkers.

---

## 4. Architectural Summary

```text
                     Observation (from ObservationBuilder)
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │     BaselinePolicy (ABC)      │
                      │  - decide(obs) -> SearchAction│
                      │  - reset()                    │
                      │  - metadata                   │
                      └───────┬───────────────┬───────┘
                              │               │
       ┌──────────────────────┼───────────────┼──────────────────────┐
       │                      │               │                      │
       ▼                      ▼               ▼                      ▼
RandomWalkPolicy     NonBacktracking   DegreeBasedWalk    AntColonyWalkPolicy
(unbiased diffusion) (1-step memory)   (observed degree)  (private pheromone)
       │                      │
       ▼                      ▼
IndependentWalkers     FloodingPolicy
(k independent RW)   (unvisited frontier)
```
