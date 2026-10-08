"""
Formal Types and Enums for Adaptive Graph Search
================================================
Defines strongly typed enumerations and identifiers for the computational model.
"""

from enum import Enum, auto


class EdgeState(str, Enum):
    """Discrete operational state of an edge in the dynamic graph.

    - ON: Edge is active, available for traversal, and facilitates transport.
    - OFF: Edge is inactive/broken, unavailable for traversal, blocks transport.
    """
    ON = "ON"
    OFF = "OFF"


class TargetMode(str, Enum):
    """Locomotion model of the search target.

    - STATIC: Target remains stationary at its assigned node x_t = x_0.
    - MOVING: Target relocates to an adjacent active node with probability p_move.
    """
    STATIC = "STATIC"
    MOVING = "MOVING"


class DynamicRegime(str, Enum):
    """Categorical churn regimes representing temporal edge turnover frequency."""
    STATIC = "STATIC"
    SLOW = "SLOW"
    MEDIUM = "MEDIUM"
    FAST = "FAST"
    VERY_FAST = "VERY_FAST"


class DetectionOutcome(str, Enum):
    """Sensing detection outcome classification under noisy observation models.

    - POSITIVE: True detection of target presence within signal radius s.
    - NEGATIVE: True negative detection (absence of target correctly indicated).
    - MISSED: False negative (target present within radius s, but sensor failed to trigger).
    - FALSE_POSITIVE: False alarm (target not within radius s, but sensor triggered).
    """
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MISSED = "MISSED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class SearchTerminationStatus(str, Enum):
    """Termination condition classification for a single search episode.

    - SUCCESS: Target successfully detected/acquired within budget constraints.
    - BUDGET_EXHAUSTED: Search exceeded maximum simulation steps/time budget.
    - DISCONNECTED: Search terminated due to complete graph disconnection or trap state.
    """
    SUCCESS = "SUCCESS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    DISCONNECTED = "DISCONNECTED"
