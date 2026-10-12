"""
FastAPI Application and Route Handlers
======================================
Defines the REST API endpoints for interactive simulation control, graph streaming,
cognitive state inspection, and metrics monitoring.
"""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from aesw.api.models import (
    AlgorithmInfo,
    ChurnItemDto,
    DecisionTraceDto,
    EvidenceItemDto,
    GraphResponse,
    GraphTypeInfo,
    MetricsDto,
    SimulationConfig,
    SimulationStatusResponse,
    StepResultResponse,
    TrajectoryDto,
    WalkerStateDto,
)
from aesw.api.simulation_manager import SimulationManager, default_manager


def create_app(manager: Optional[SimulationManager] = None) -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    sim_manager = manager or default_manager

    app = FastAPI(
        title="AESW Simulation API",
        description="REST API for Adaptive Evidence-Sharing Walkers dynamic graph search simulation.",
        version="0.1.0",
    )

    # Enable CORS for local web clients and dashboards
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", tags=["System"])
    def get_health() -> Dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "service": "aesw-simulation-api",
            "version": "0.1.0",
        }

    @app.get("/api/algorithms", response_model=List[AlgorithmInfo], tags=["Configuration"])
    def get_algorithms() -> List[AlgorithmInfo]:
        """List supported search algorithms."""
        return [
            AlgorithmInfo(
                id="aesw",
                name="Adaptive Evidence-Sharing Walkers (AESW)",
                description="Decentralized multi-walker search with adaptive churn estimation, exponential memory decay, and node-mediated information sharing.",
                is_aesw=True,
            ),
            AlgorithmInfo(
                id="random_walk",
                name="Random Walk",
                description="Standard unbiased random walker with uniform transition probabilities among active incident links.",
                is_aesw=False,
            ),
            AlgorithmInfo(
                id="non_backtracking",
                name="Non-Backtracking Walk",
                description="Memory-efficient random walk avoiding immediate return to the predecessor vertex.",
                is_aesw=False,
            ),
            AlgorithmInfo(
                id="k_walk",
                name="k Independent Random Walkers",
                description="Team of k uncoordinated random walkers searching concurrently.",
                is_aesw=False,
            ),
            AlgorithmInfo(
                id="degree",
                name="Degree-Based Walk",
                description="Heuristic greedy walk preferentially selecting highest locally observed active degree neighbors.",
                is_aesw=False,
            ),
            AlgorithmInfo(
                id="flooding",
                name="Flooding / Frontier Walk",
                description="Breadth-first frontier discovery prioritizing uninspected boundary vertices.",
                is_aesw=False,
            ),
            AlgorithmInfo(
                id="ant_colony",
                name="Ant Colony Walk",
                description="Pheromone trail reinforced walk with local evaporation and probabilistic selection.",
                is_aesw=False,
            ),
        ]

    @app.get("/api/graph-types", response_model=List[GraphTypeInfo], tags=["Configuration"])
    def get_graph_types() -> List[GraphTypeInfo]:
        """List supported static graph topologies."""
        return [
            GraphTypeInfo(
                id="er",
                name="Erdős–Rényi G(n, p)",
                description="Random graph with independent edge attachment probability p.",
                parameters={"p": 0.15},
            ),
            GraphTypeInfo(
                id="ba",
                name="Barabási–Albert BA(n, m)",
                description="Scale-free preferential attachment network with power-law degree distribution.",
                parameters={"m": 2},
            ),
            GraphTypeInfo(
                id="ws",
                name="Watts–Strogatz WS(n, k, p)",
                description="Small-world network with high clustering coefficient and short average path length.",
                parameters={"k": 4, "p": 0.1},
            ),
            GraphTypeInfo(
                id="grid",
                name="2D Spatial Grid with Obstacles",
                description="Square grid network with configurable dimensions and obstacle ratio.",
                parameters={"rows": 6, "cols": 6, "obstacle_ratio": 0.1},
            ),
            GraphTypeInfo(
                id="rgg",
                name="Random Geometric Graph RGG(n, radius)",
                description="Spatial proximity network where nodes connect within Euclidean radius r.",
                parameters={"radius": 0.28, "dim": 2},
            ),
        ]

    @app.post("/api/simulations", response_model=SimulationStatusResponse, status_code=status.HTTP_201_CREATED, tags=["Simulations"])
    def create_simulation(config: SimulationConfig) -> SimulationStatusResponse:
        """Create a new simulation session from configuration parameters."""
        try:
            session = sim_manager.create_simulation(config)
            return session.get_status()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to generate simulation: {str(exc)}",
            )

    @app.get("/api/simulations", response_model=List[SimulationStatusResponse], tags=["Simulations"])
    def list_simulations() -> List[SimulationStatusResponse]:
        """List all active simulation sessions."""
        return sim_manager.list_sessions()

    @app.get("/api/simulations/{simulation_id}", response_model=SimulationStatusResponse, tags=["Simulations"])
    def get_simulation(simulation_id: str) -> SimulationStatusResponse:
        """Retrieve status of an active simulation session."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return session.get_status()

    @app.delete("/api/simulations/{simulation_id}", tags=["Simulations"])
    def delete_simulation(simulation_id: str) -> Dict[str, str]:
        """Delete an active simulation session."""
        success = sim_manager.delete_session(simulation_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return {"status": "deleted", "simulation_id": simulation_id}

    @app.post("/api/simulations/{simulation_id}/step", response_model=StepResultResponse, tags=["Execution"])
    def step_simulation(simulation_id: str) -> StepResultResponse:
        """Advance the simulation by exactly one discrete step."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        try:
            return session.step()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error executing step: {str(exc)}",
            )

    @app.post("/api/simulations/{simulation_id}/reset", response_model=SimulationStatusResponse, tags=["Execution"])
    def reset_simulation(simulation_id: str) -> SimulationStatusResponse:
        """Reset the simulation back to initial state."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        try:
            return session.reset()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error resetting simulation: {str(exc)}",
            )

    @app.get("/api/simulations/{simulation_id}/graph", response_model=GraphResponse, tags=["Inspection"])
    def get_simulation_graph(simulation_id: str) -> GraphResponse:
        """Get the current graph snapshot with active links and walker positions."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return session.get_graph()

    @app.get("/api/simulations/{simulation_id}/walkers", response_model=List[WalkerStateDto], tags=["Inspection"])
    def get_simulation_walkers(simulation_id: str) -> List[WalkerStateDto]:
        """Get current walker states, visit counts, and recent decisions."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return session.get_walkers()

    @app.get("/api/simulations/{simulation_id}/metrics", response_model=MetricsDto, tags=["Inspection"])
    def get_simulation_metrics(simulation_id: str) -> MetricsDto:
        """Get live evaluation metrics."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return session.get_metrics()

    @app.get("/api/simulations/{simulation_id}/evidence", response_model=List[EvidenceItemDto], tags=["Inspection"])
    def get_simulation_evidence(simulation_id: str) -> List[EvidenceItemDto]:
        """Get active evidence memory cache items."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return session.get_evidence()

    @app.get("/api/simulations/{simulation_id}/churn", response_model=List[ChurnItemDto], tags=["Inspection"])
    def get_simulation_churn(simulation_id: str) -> List[ChurnItemDto]:
        """Get churn estimation values and historical time series per walker."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return session.get_churn()

    @app.get("/api/simulations/{simulation_id}/decisions", response_model=List[DecisionTraceDto], tags=["Inspection"])
    def get_simulation_decisions(simulation_id: str) -> List[DecisionTraceDto]:
        """Get latest decision audit traces."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return session.get_decisions()

    @app.get("/api/simulations/{simulation_id}/trajectory", response_model=TrajectoryDto, tags=["Inspection"])
    def get_simulation_trajectory(simulation_id: str) -> TrajectoryDto:
        """Get walker and target trajectories over time."""
        session = sim_manager.get_session(simulation_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation '{simulation_id}' not found.",
            )
        return session.get_trajectories()

    return app


# Default application instance for uvicorn
app = create_app()
