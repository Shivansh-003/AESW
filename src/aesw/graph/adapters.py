"""
Graph Adapters and Temporal Extension Interfaces
================================================
Defines interfaces for static graph factories and clean extension points
for future temporal network sources (e.g. AS-733).
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional
from aesw.graph.base import GraphInstance
from aesw.environment.problem import GraphSpecification


class BaseGraphGenerator(ABC):
    """Abstract interface for static graph generators."""

    @abstractmethod
    def generate(self, spec: GraphSpecification, seed: Optional[int] = None) -> GraphInstance:
        """Generate a static graph topology instance G = (V, E)."""
        pass


class TemporalGraphSource(ABC):
    """Abstract interface for real-world or simulated temporal network snapshot sources.

    Clean extension point for AS-733 evaluation.
    Exposes a sequence of static graph snapshots G_1, G_2, ..., G_T without
    coupling the underlying generator to simulation dynamic engines.
    """

    @abstractmethod
    def num_snapshots(self) -> int:
        """Total number of temporal snapshots available."""
        pass

    @abstractmethod
    def get_snapshot(self, t: int) -> GraphInstance:
        """Retrieve static graph snapshot G_t at temporal index t."""
        pass

    @abstractmethod
    def iter_snapshots(self) -> Iterator[GraphInstance]:
        """Iterate through all temporal graph snapshots."""
        pass
