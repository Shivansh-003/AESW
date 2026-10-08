"""
Base Policy Architecture for Search Algorithms
==============================================
Defines the common abstract interface and metadata schema for all search policies.
Guarantees that every policy operates under the Controlled Observation Principle.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from aesw.baselines.types import BaselineType
from aesw.baselines.actions import SearchAction, validate_action
from aesw.environment.observation import Observation


@dataclass(frozen=True)
class BaselineMetadata:
    """Standardized metadata characterizing a search baseline for experimental reporting.

    Attributes:
        name: Human-readable algorithm name.
        baseline_type: Categorical BaselineType enum value.
        description: Brief methodological description.
        uses_memory: Whether the policy maintains state across decision steps.
        uses_detection_signals: Whether the policy reacts to binary sensor signals.
        uses_communication: Whether the policy communicates with other agents.
        is_multi_walker: Whether the baseline encapsulates multi-walker coordination.
        parameters: Configurable hyperparameter dictionary.
    """
    name: str
    baseline_type: BaselineType
    description: str
    uses_memory: bool = False
    uses_detection_signals: bool = False
    uses_communication: bool = False
    is_multi_walker: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)


class BaselinePolicy(ABC):
    """Abstract base class for all search policies and benchmark algorithms.

    Architectural Rule:
    Every policy receives ONLY an Observation object snapshot representing partial,
    local visibility. A policy has NO reference to GraphInstance, GroundTruthState,
    DynamicGraphState, or TargetState.
    """

    @property
    @abstractmethod
    def baseline_type(self) -> BaselineType:
        """Categorical baseline type."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> BaselineMetadata:
        """Descriptive metadata for benchmark evaluation."""
        ...

    @property
    @abstractmethod
    def seed(self) -> Optional[int]:
        """PRNG seed associated with this policy instance."""
        ...

    @abstractmethod
    def decide(self, observation: Observation) -> SearchAction:
        """Select an action based strictly on the walker-visible Observation snapshot.

        Args:
            observation: Partial, local observation snapshot.

        Returns:
            Validated SearchAction to be executed by the simulation engine.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal policy memory and re-seed PRNG stream reproducibly."""
        ...

    @staticmethod
    def get_available_candidates(observation: Observation) -> list[int | str]:
        """Extract all currently active, observed neighbor candidates.

        Filters checked_neighbors to those whose edge state is confirmed ON.
        Returns a deterministically sorted list.

        Args:
            observation: The walker-visible observation.

        Returns:
            Sorted list of active neighbor identifiers available for traversal.
        """
        candidates = [
            nbr for nbr in observation.checked_neighbors
            if observation.is_edge_known_active(nbr)
        ]
        return sorted(candidates, key=str)

    def _validate_action(self, action: SearchAction, observation: Observation) -> SearchAction:
        """Internal helper to validate the decided action before emitting."""
        validate_action(action, observation)
        return action
