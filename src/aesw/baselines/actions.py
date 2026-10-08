"""
Search Action Representation and Validation
===========================================
Defines formal actions emitted by search policies and enforces observation-bounded validation.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from aesw.baselines.types import ActionType
from aesw.environment.observation import Observation


@dataclass(frozen=True)
class SearchAction:
    """Formal decision emitted by a search policy at a discrete decision step.

    Attributes:
        action_type: Primitive action (MOVE, STAY, or CHECK).
        destination: Adjacent node identifier for MOVE actions. Must be None or current_node for STAY.
        metadata: Optional algorithmic audit details (e.g., candidate count, scores, policy state).
    """
    action_type: ActionType
    destination: Optional[int | str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, ActionType):
            raise TypeError(f"action_type must be an ActionType instance, got {type(self.action_type).__name__}")


def validate_action(action: SearchAction, observation: Observation) -> None:
    """Validate that an action strictly adheres to the walker's current observation.

    Defense-in-depth enforcement:
    - For MOVE: Destination must be an observed neighbor and verified active (ON).
    - For STAY: Destination must be None or match current_node.
    - Prevents arbitrary unseen node traversal or movement across OFF edges.

    Args:
        action: The candidate SearchAction chosen by the policy.
        observation: The partial Observation snapshot from which the decision was made.

    Raises:
        ValueError: If the action requests an unobserved, invalid, or inactive destination.
    """
    if action.action_type == ActionType.MOVE:
        if action.destination is None:
            raise ValueError("MOVE action must specify a non-None destination node.")

        # Destination must have been inspected within budget B
        if action.destination not in observation.checked_neighbors:
            raise ValueError(
                f"Invalid destination '{action.destination}': node was not inspected within "
                f"checked_neighbors {observation.checked_neighbors}"
            )

        # Edge to destination must be observed as active (ON)
        if not observation.is_edge_known_active(action.destination):
            raise ValueError(
                f"Invalid destination '{action.destination}': link is not currently observed as active (ON)."
            )

    elif action.action_type == ActionType.STAY:
        if action.destination is not None and action.destination != observation.current_node:
            raise ValueError(
                f"STAY action destination must be None or current node '{observation.current_node}', "
                f"got '{action.destination}'."
            )
