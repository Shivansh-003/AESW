# Interactive Research Workstation: Adaptive Evidence-Sharing Walkers (AESW)

This document provides a comprehensive overview of the Interactive Research Workstation designed for simulating, analyzing, and demonstrating Adaptive Evidence-Sharing Walkers (AESW) on dynamic graphs under uncertainty.

---

## 1. Architecture Overview

The workstation consists of two decoupled tiers:

1. **Backend Simulation Engine & REST API (`src/aesw/api`)**:
   - Built on FastAPI, Uvicorn, NetworkX, and NumPy.
   - Wraps the core search algorithms (`AESWPolicy`, `RandomWalkPolicy`, `DegreePolicy`, `AntColonyPolicy`, `KWalkPolicy`, `FloodingPolicy`, `NonBacktrackingPolicy`) and simulation coordination (`SimulationCoordinator`).
   - Maintains an in-memory session store (`SimulationManager`) with thread-safe re-entrant locking (`threading.RLock`).
   - Computes deterministic 2D spring layouts for graph visualization using fixed random seeds.
   - Exposes complete state inspection: active/inactive edge states, walker states, evidence caches, churn volatility estimations, candidate score audit traces, and metrics.

2. **Frontend Interactive Workstation (`frontend/`)**:
   - Modern Single Page Application built with React 18, TypeScript, Vite, Tailwind CSS, Lucide React, and Recharts.
   - Dark graphite/navy scientific workstation aesthetic with cyan/teal accents.
   - Real-time interactive SVG graph explorer with pan, zoom, reset, node selection, edge state toggles, and walker/target visualization.
   - Explainable decision scoring inspection table showing raw candidate scores and softmax selection probabilities.
   - Multi-tab analytic views for Evidence Cache, Volatility / Churn tracking ($\hat{\lambda}_t$), Walker activity, and Trajectory logs.

```
┌────────────────────────────────────────────────────────┐
│             Interactive Web UI (React/TS)              │
│  - ConfigPanel       - MetricsStrip   - GraphExplorer  │
│  - SimToolbar        - Inspector      - AnalysisPanels │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / JSON
┌──────────────────────────▼─────────────────────────────┐
│                 FastAPI REST Service                   │
│  - Session Lifecycle (/api/simulations)                │
│  - Deterministic 2D Coordinates (NetworkX Spring)      │
│  - Step / Reset / Status Control                       │
└──────────────────────────┬─────────────────────────────┘
                           │ Direct Python Call
┌──────────────────────────▼─────────────────────────────┐
│            Authoritative Python Engine                 │
│  - SimulationCoordinator & ExperimentInstance          │
│  - AESW Multi-Agent Search Policy                      │
│  - Strict Epistemic Firewall (No Target/Graph Leaks)   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Epistemic Firewall & Scientific Integrity

A fundamental tenet of the research framework is the strict separation between environment ground truth and walker perception:

- **Walker Decisions**: Walkers and policies receive only partial local observations (`LocalObservation` with local degree, active neighbors within visibility horizon, and local noisy sensor evidence).
- **No Target Telepathy**: Walkers have no access to the true target coordinate, global graph adjacency, or future edge transitions.
- **Researcher Visualization**: The interactive graph explorer visualizes true target position and inactive edge states solely for human researcher evaluation and academic demonstrations. This ground truth is never passed into the agent decision pipeline.

---

## 3. Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### Backend Installation and Startup
1. Ensure virtual environment has required dependencies installed:
   ```bash
   pip install fastapi uvicorn pydantic networkx numpy scipy pytest
   ```
2. Start the FastAPI server:
   ```bash
   uvicorn aesw.api.app:app --host 127.0.0.1 --port 8000 --reload
   ```
   The interactive Swagger documentation will be available at:
   `http://127.0.0.1:8000/docs`

### Frontend Installation and Startup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies (if not already done):
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Access the interactive workstation at:
   `http://localhost:5173`

To build the static distribution for production:
```bash
npm run build
```

---

## 4. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check and system timestamp |
| `GET` | `/api/algorithms` | List available search algorithms and default configs |
| `GET` | `/api/graph-types` | List supported synthetic graph topologies |
| `POST` | `/api/simulations` | Instantiate a new simulation experiment session |
| `GET` | `/api/simulations/{id}` | Retrieve current simulation status and metrics |
| `POST` | `/api/simulations/{id}/step` | Advance simulation by one discrete time step |
| `POST` | `/api/simulations/{id}/reset` | Reset simulation to initial deterministic seed state ($t=0$) |
| `GET` | `/api/simulations/{id}/graph` | Retrieve 2D node coordinates, active/inactive edge states |
| `GET` | `/api/simulations/{id}/walkers` | Retrieve walker states, current nodes, and modes |
| `GET` | `/api/simulations/{id}/metrics` | Retrieve search metrics ($T, M, Q, C$, coverage, revisits) |
| `GET` | `/api/simulations/{id}/evidence` | Inspect current evidence cache entries for all walkers |
| `GET` | `/api/simulations/{id}/churn` | Inspect online estimated churn volatility $\hat{\lambda}_t$ |
| `GET` | `/api/simulations/{id}/decisions` | Inspect latest explainable candidate selection scores |
| `GET` | `/api/simulations/{id}/trajectory`| Retrieve history of walker paths and actions |
| `DELETE` | `/api/simulations/{id}` | Terminate and release simulation session |

---

## 5. Demonstration Workflow for Academic Presentations

When presenting to research supervisors or faculty:

1. **Baseline Comparison**:
   - Select `random_walk` or `degree` with 30 nodes on a Barabási-Albert or Erdős-Rényi dynamic graph.
   - Advance the simulation by 20 steps. Observe high node revisit rates and lack of coordinated coverage.
2. **AESW Multi-Agent Search**:
   - Generate an identical graph seed using algorithm `aesw`.
   - Observe walker coordination:
     - **Mode Switching**: Notice walkers switching to `LONG_JUMP` exploration when stagnating.
     - **Evidence Decay**: Inspect the Evidence Cache tab to see confidence weights decay dynamically according to local churn rates.
     - **Explainable Scores**: Open the Decision Inspector to view candidate scores across positive evidence $P(u)$, negative evidence $N(u)$, revisit avoidance $R(u)$, degree preference $D(u)$, and stagnation penalty $S(u)$.
3. **Volatility Response**:
   - Switch between dynamic regimes (`LOW`, `MEDIUM`, `HIGH`).
   - Monitor the Volatility & Churn tab to demonstrate how $\hat{\lambda}_t$ dynamically adapts to graph topology fluctuations.
