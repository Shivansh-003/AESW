"""
Dynamic Transition Models and Statistics
========================================
Immutable data structures for step-level transition counters and state snapshots.
"""

from dataclasses import dataclass, field
from typing import Mapping
from aesw.environment.types import EdgeState, DynamicRegime


@dataclass(frozen=True)
class TransitionStatistics:
    """Statistics recorded during a single synchronous transition step t -> t+1.

    Attributes:
        time: Timestamp after transition (t+1).
        on_to_off: Number of active edges that deactivated (ON -> OFF).
        off_to_on: Number of inactive edges that activated (OFF -> ON).
        on_to_on: Number of active edges that remained active (ON -> ON).
        off_to_off: Number of inactive edges that remained inactive (OFF -> OFF).
        active_edge_count: Number of active edges at time t+1.
        inactive_edge_count: Number of inactive edges at time t+1.
    """
    time: int
    on_to_off: int
    off_to_on: int
    on_to_on: int
    off_to_off: int
    active_edge_count: int
    inactive_edge_count: int

    @property
    def total_transitions(self) -> int:
        """Total number of edge state changes occurred."""
        return self.on_to_off + self.off_to_on


@dataclass(frozen=True)
class DynamicGraphSnapshot:
    """Immutable point-in-time capture of the dynamic edge configuration at step t.

    Attributes:
        time: Simulation time step t.
        edge_states: Mapping of edge endpoint frozensets to EdgeState (ON/OFF).
        active_edge_count: Total number of ON edges.
        inactive_edge_count: Total number of OFF edges.
        regime: DynamicRegime active during this snapshot.
    """
    time: int
    edge_states: Mapping[frozenset[int | str], EdgeState]
    active_edge_count: int
    inactive_edge_count: int
    regime: DynamicRegime
