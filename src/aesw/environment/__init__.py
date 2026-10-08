"""
Environment Module
==================
Formal environment definitions, problem specifications, ground-truth state,
and partial observation models for Adaptive Graph Search.
"""

from aesw.environment.types import (
    EdgeState,
    TargetMode,
    DynamicRegime,
    DetectionOutcome,
    SearchTerminationStatus,
)
from aesw.environment.models import (
    Node,
    Edge,
    DelaySpecification,
    TargetState,
    WalkerState,
)
from aesw.environment.state import GroundTruthState
from aesw.environment.observation import (
    ObservedEdgeInfo,
    TargetSignalObservation,
    Observation,
)
from aesw.environment.problem import (
    GraphSpecification,
    DynamicsSpecification,
    TargetSpecification,
    DetectionSpecification,
    ObservationSpecification,
    WalkerSpecification,
    SimulationSpecification,
    CostSpecification,
    ProblemDefinition,
)

__all__ = [
    # Types
    "EdgeState",
    "TargetMode",
    "DynamicRegime",
    "DetectionOutcome",
    "SearchTerminationStatus",
    # Models
    "Node",
    "Edge",
    "DelaySpecification",
    "TargetState",
    "WalkerState",
    # State & Observation
    "GroundTruthState",
    "ObservedEdgeInfo",
    "TargetSignalObservation",
    "Observation",
    # Specifications
    "GraphSpecification",
    "DynamicsSpecification",
    "TargetSpecification",
    "DetectionSpecification",
    "ObservationSpecification",
    "WalkerSpecification",
    "SimulationSpecification",
    "CostSpecification",
    "ProblemDefinition",
]
