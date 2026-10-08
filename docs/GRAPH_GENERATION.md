# Graph Generation Subsystem

---

## 1. Overview & Research Scope

The Graph Generation subsystem provides a deterministic, configuration-driven foundation for constructing the underlying static graph topologies $G = (V, E)$ defined in the research specification:

1. **Erdős–Rényi** $G(n, p)$
2. **Barabási–Albert** $\text{BA}(n, m)$
3. **Watts–Strogatz** $\text{WS}(n, k, p)$
4. **2D Spatial Grid with Obstacles**
5. **Random Geometric Graph (RGG)**

It also defines an explicit architectural extension point ([`TemporalGraphSource`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/graph/adapters.py)) for loading **AS-733** real-world temporal network traces.

---

## 2. Mathematical Definitions & Graph Families

### A. Erdős–Rényi $G(n, p)$
- **Model**: An undirected graph with $n$ vertices where each possible pair of distinct vertices $(u, v)$ is connected with independent probability $p \in [0, 1]$.
- **Invariants**:
  - $n > 0$, $0 \le p \le 1$.
  - When $p = 0$, $|E| = 0$.
  - When $p = 1$, $|E| = \binom{n}{2}$ (complete graph).
  - No self-loops.

### B. Barabási–Albert $\text{BA}(n, m)$
- **Model**: Scale-free preferential attachment network starting with $m$ vertices, where each new vertex connects to $m$ existing vertices with probability proportional to their current degree.
- **Invariants**:
  - $n > 0$, $1 \le m < n$.
  - Generates power-law degree distributions $P(k) \sim k^{-3}$.
  - Strictly connected when starting from an initial connected seed.

### C. Watts–Strogatz $\text{WS}(n, k, p)$
- **Model**: Small-world network constructed by initializing a ring lattice of $n$ vertices with degree $k$, then rewiring each edge with probability $p \in [0, 1]$.
- **Invariants**:
  - $n > 0$, $k$ even, $2 \le k < n$, $0 \le p \le 1$.
  - Total edges $|E| = \frac{n k}{2}$.

### D. 2D Spatial Grid with Obstacles
- **Model**: Planar lattice of dimensions $\text{rows} \times \text{cols}$ where cells represent vertices unless masked as obstacles. Edges connect strictly orthogonal neighbors: $\Delta r + \Delta c = 1$ (up, down, left, right).
- **Invariants**:
  - Without obstacles: $|V| = \text{rows} \times \text{cols}$ and $|E| = (\text{rows}-1)\text{cols} + \text{rows}(\text{cols}-1)$.
  - Obstacle cells are completely excluded from active vertex set $V$.
  - Vertices retain exact spatial coordinates $(r, c)$ in metadata.

### E. Random Geometric Graph (RGG)
- **Model**: $n$ vertices distributed uniformly in the unit Euclidean hypercube $[0, 1]^d$. An edge connects $(u, v)$ if and only if $\|x_u - x_v\|_2 \le r$.
- **Invariants**:
  - $n > 0$, $r \ge 0.0$.
  - Every edge satisfies $\text{dist}(u, v) \le r + \epsilon$.
  - Spatial coordinates are preserved on nodes.

---

## 3. Architecture & `GraphInstance` Abstraction

To ensure that research algorithms and future simulation modules do not depend directly on NetworkX internals, all generators return a [`GraphInstance`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/graph/base.py):

```text
GraphInstance
├── node_count, edge_count
├── nodes: Mapping[node_id, Node]
├── edges: tuple[Edge, ...]
├── neighbors(node_id) -> list[node_id]
├── degree(node_id) -> int
├── has_edge(u, v) -> bool
├── is_connected() -> bool
├── number_of_connected_components() -> int
├── compute_statistics() -> TopologyStatistics
└── metadata: GraphMetadata (seed, parameters, spatial, etc.)
```

NetworkX is encapsulated as an internal implementation dependency.

---

## 4. Reproducibility Recipe

For any experiment:
$$\text{Configuration} + \text{Graph Family} + \text{Parameters} + \text{Seed} \implies \text{Exact Topology Realization}$$

- Generators use [`create_rng(seed)`](file:///c:/Users/shiva/OneDrive/Desktop/AESW/src/aesw/utils/reproducibility.py#L37-L50) to produce isolated NumPy `Generator` streams.
- No global `random.seed()` or `numpy.random.seed()` pollution is introduced.
- Identical seeds yield identical vertex sets, edge sets, obstacle masks, and spatial coordinates.

---

## 5. Distinction: Static Topology vs. Dynamic Graph

- **Static Topology**: Generates the underlying physical topology $G = (V, E)$. All edges represent structural adjacency.
- **Dynamic Graph Engine**: Layers stochastic temporal edge availability $s_t(e) \in \{\text{ON}, \text{OFF}\}$ over this static topology, producing dynamic graphs $G_t = (V, E_t)$.
