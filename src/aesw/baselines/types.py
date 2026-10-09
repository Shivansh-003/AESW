"""
Baseline Types and Enumerations
===============================
Defines categorical identifiers for baseline search algorithms and action primitives.
"""

from enum import Enum


class BaselineType(str, Enum):
    """Categorical enumeration of benchmark search algorithms."""
    RANDOM_WALK = "random_walk"
    NON_BACKTRACKING = "non_backtracking"
    INDEPENDENT_RANDOM_WALKERS = "independent_random_walkers"
    DEGREE_BASED = "degree_based"
    FLOODING = "flooding"
    ANT_COLONY = "ant_colony"
    AESW = "aesw"


class ActionType(str, Enum):
    """Primitive operational actions available to search policies."""
    MOVE = "move"
    STAY = "stay"
    CHECK = "check"
    JUMP = "jump"

