"""
API Data Transfer Objects and Request/Response Schemas
======================================================
Defines strongly typed Pydantic models for configuration, simulation states,
graph visualization, walker decisions, metrics, and evidence caching.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """Configuration payload for instantiating a new simulation session."""
    algorithm: str = Field(
        default="aesw",
        description="Search algorithm identifier (aesw, random_walk, non_backtracking, k_walk, degree, flooding, ant_colony)",
    )
    graph_type: str = Field(
        default="er",
        description="Graph family identifier (er, ba, ws, grid, rgg)",
    )
    num_nodes: int = Field(
        default=30,
        ge=4,
        le=500,
        description="Number of vertices in the static topology",
    )
    graph_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Topology-specific parameters (e.g. p, m, k, rows, cols, obstacle_ratio, radius)",
    )
    regime: str = Field(
        default="MEDIUM",
        description="Named dynamic churn regime (STATIC, SLOW, MEDIUM, FAST, VERY_FAST)",
    )
    p_on: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional explicit Markovian edge ON transition probability",
    )
    p_off: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional explicit Markovian edge OFF transition probability",
    )
    p_move: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Target movement probability per discrete step",
    )
    target_mode: str = Field(
        default="MOVING",
        description="Target locomotion mode (STATIC or MOVING)",
    )
    signal_radius: int = Field(
        default=1,
        ge=0,
        description="Sensor detection signal hop radius s",
    )
    p_d: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Sensor true detection probability",
    )
    p_fa: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Sensor false alarm probability",
    )
    neighbor_budget: int = Field(
        default=8,
        ge=1,
        description="Observation candidate inspection budget B",
    )
    walker_count: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of autonomous search walkers k",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Master pseudo-random number generator seed",
    )
    time_horizon: int = Field(
        default=100,
        ge=5,
        le=2000,
        description="Maximum discrete simulation steps before budget exhaustion",
    )
    communication_mode: str = Field(
        default="PUSH_PULL",
        description="AESW node-mediated communication mode (NO_SHARING, PUSH, PULL, PUSH_PULL)",
    )
    c_m: float = Field(
        default=0.1,
        ge=0.0,
        description="Communication cost coefficient per message Q",
    )


class GraphNodeDto(BaseModel):
    """Serialized node for graph explorer visualization."""
    id: Union[int, str]
    label: str
    degree: int
    is_target: bool = False
    is_visited: bool = False
    walker_ids: List[int] = Field(default_factory=list)
    has_evidence: bool = False
    evidence_polarity: Optional[str] = None
    evidence_weight: Optional[float] = None
    x: float = 0.0
    y: float = 0.0


class GraphEdgeDto(BaseModel):
    """Serialized edge with dynamic active/inactive state."""
    source: Union[int, str]
    target: Union[int, str]
    is_active: bool
    state: str
    is_recent_transition: bool = False


class GraphResponse(BaseModel):
    """Full graph snapshot for interactive visualization."""
    simulation_id: str
    nodes: List[GraphNodeDto]
    edges: List[GraphEdgeDto]
    target_node: Union[int, str]
    visited_nodes: List[Union[int, str]]
    active_edge_count: int
    total_edge_count: int


class CandidateScoreDto(BaseModel):
    """Detailed score decomposition for an active neighbor candidate."""
    node_id: Union[int, str]
    positive_evidence: float
    novelty: float
    revisit_penalty: float
    delay: float
    total_score: float
    probability: float


class DecisionTraceDto(BaseModel):
    """Audit trace of the latest cognitive decision made by a walker."""
    walker_id: int
    time: int
    current_node: Union[int, str]
    mode: str
    estimated_lambda: float
    stagnation_steps: int
    selected_node: Optional[Union[int, str]] = None
    action_type: str
    destination: Optional[Union[int, str]] = None
    candidate_scores: List[CandidateScoreDto] = Field(default_factory=list)


class WalkerStateDto(BaseModel):
    """Current state of an individual search walker."""
    walker_id: int
    current_node: Union[int, str]
    step_count: int
    message_count: int
    mode: Optional[str] = None
    estimated_lambda: Optional[float] = None
    stagnation_steps: Optional[int] = None
    recent_action: Optional[str] = None
    recent_destination: Optional[Union[int, str]] = None
    visit_counts: Dict[str, int] = Field(default_factory=dict)
    latest_decision: Optional[DecisionTraceDto] = None


class MetricsDto(BaseModel):
    """Standardized search evaluation metrics from the simulation engine."""
    algorithm: str
    search_time: int
    status: str
    success: bool
    total_moves: int
    node_revisits: int
    nodes_visited: int
    total_messages: int
    total_cost: float
    c_m: float


class EvidenceItemDto(BaseModel):
    """Serialized record from an agent's local evidence memory cache."""
    node_id: Union[int, str]
    walker_id: int
    polarity: str
    confidence: float
    effective_weight: float
    effective_confidence: float
    timestamp: int
    age: int
    source: str


class ChurnItemDto(BaseModel):
    """Online churn estimation status and history per walker."""
    walker_id: int
    estimated_lambda: float
    overlap: float
    raw_churn_rate: float
    update_count: int
    history: List[Dict[str, Any]] = Field(default_factory=list)


class TrajectoryDto(BaseModel):
    """Trajectory trace over discrete time."""
    time: int
    target_trajectory: List[Union[int, str]]
    walker_trajectories: Dict[int, List[Union[int, str]]]


class SimulationStatusResponse(BaseModel):
    """Summary status of a simulation session."""
    simulation_id: str
    status: str
    termination_status: Optional[str] = None
    time: int
    time_horizon: int
    algorithm: str
    is_aesw: bool
    seed: int
    walker_count: int
    config: SimulationConfig
    metrics: MetricsDto


class StepResultResponse(BaseModel):
    """Response returned upon advancing one simulation step."""
    simulation_id: str
    time: int
    is_terminated: bool
    termination_status: Optional[str] = None
    metrics: MetricsDto
    walkers: List[WalkerStateDto]
    latest_decisions: List[DecisionTraceDto]


class AlgorithmInfo(BaseModel):
    """Metadata describing a supported search algorithm."""
    id: str
    name: str
    description: str
    is_aesw: bool


class GraphTypeInfo(BaseModel):
    """Metadata describing a supported graph family."""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
