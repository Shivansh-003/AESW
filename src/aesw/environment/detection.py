"""
Target Detection and Uncertainty Engine
=======================================
Implements probabilistic target sensing under noisy observations:
- Detection probability p_d inside signal radius s (shortest-path in G_t).
- False alarm probability p_fa outside signal radius s.
- Explicit classification of detection outcomes (TP, MISS, FP, TN).

CRITICAL INVARIANT:
The detailed DetectionResult (containing ground-truth target distance and classification)
is an internal environment object. Search walkers only receive the binary/noisy observation.
"""

from dataclasses import dataclass
from typing import Optional
from collections import deque
import numpy as np

from aesw.environment.types import DetectionOutcome
from aesw.environment.problem import DetectionSpecification
from aesw.dynamics.view import ActiveGraphView
from aesw.utils.reproducibility import create_rng


@dataclass(frozen=True)
class DetectionResult:
    """Internal environment record of a sensing evaluation at observer_node.

    Contains full ground-truth context for analysis and verification.
    This full record must NOT be directly passed to search walkers.

    Attributes:
        observer_node: The vertex u where the sensor reading was taken.
        target_node: Ground-truth target location x_t.
        graph_distance: Shortest-path hop distance dist_{G_t}(u, x_t). (inf if unreachable).
        within_radius: Whether dist_{G_t}(u, x_t) <= signal_radius s.
        outcome: Classification (POSITIVE, MISSED, FALSE_POSITIVE, NEGATIVE).
        positive_signal: Binary observed signal y_t(u) in {True, False}.
        timestamp: Simulation timestamp of detection.
    """
    observer_node: int | str
    target_node: int | str
    graph_distance: float
    within_radius: bool
    outcome: DetectionOutcome
    positive_signal: bool
    timestamp: int


class DetectionEngine:
    """Engine for probabilistic sensor observation generation over G_t.

    Attributes:
        spec: DetectionSpecification with signal_radius s, p_d, and p_fa.
        seed: Random seed for the isolated detection noise RNG stream.
    """

    def __init__(
        self,
        spec: DetectionSpecification,
        seed: Optional[int] = 42,
    ) -> None:
        if not isinstance(spec, DetectionSpecification):
            raise TypeError(f"spec must be a DetectionSpecification instance, got {type(spec).__name__}")

        self._spec: DetectionSpecification = spec
        self._seed: int = 42 if seed is None else int(seed)
        self._rng: np.random.Generator = create_rng(self._seed)

    @property
    def spec(self) -> DetectionSpecification:
        """Configured detection specification."""
        return self._spec

    @property
    def signal_radius(self) -> int:
        """Signal radius threshold s."""
        return self._spec.signal_radius

    @property
    def p_d(self) -> float:
        """True positive detection probability p_d."""
        return self._spec.p_d

    @property
    def p_fa(self) -> float:
        """False alarm probability p_fa."""
        return self._spec.p_fa

    def compute_distance(
        self,
        u: int | str,
        target_node: int | str,
        active_graph: ActiveGraphView,
    ) -> float:
        """Compute shortest path hop distance in G_t using breadth-first search.

        If target is unreachable in G_t, returns float('inf').

        Args:
            u: Observer node.
            target_node: Ground-truth target node.
            active_graph: ActiveGraphView presenting G_t = (V, E_t).

        Returns:
            Shortest path hop distance in G_t.
        """
        if u == target_node:
            return 0.0

        visited = {u}
        queue = deque([(u, 0)])

        while queue:
            curr, d = queue.popleft()
            for nbr in active_graph.active_neighbors(curr):
                if nbr == target_node:
                    return float(d + 1)
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, d + 1))

        return float("inf")

    def detect(
        self,
        observer_node: int | str,
        target_node: int | str,
        active_graph: ActiveGraphView,
        timestamp: int = 0,
    ) -> DetectionResult:
        """Evaluate noisy target detection at observer_node relative to target_node in G_t.

        Probabilistic Sensing Formulation:
        Let d = dist_{G_t}(observer_node, target_node):
        1. If d <= s (Within Detection Region):
           - Signal is positive with probability p_d (TRUE_POSITIVE).
           - Signal is negative with probability 1 - p_d (MISSED).
        2. If d > s (Outside Detection Region):
           - Signal is positive with probability p_fa (FALSE_POSITIVE).
           - Signal is negative with probability 1 - p_fa (NEGATIVE).

        Args:
            observer_node: Vertex u being inspected.
            target_node: Ground-truth target vertex x_t.
            active_graph: Current active topology view G_t.
            timestamp: Simulation time step t.

        Returns:
            Structured DetectionResult with ground-truth and binary observation data.
        """
        dist = self.compute_distance(observer_node, target_node, active_graph)
        within_rad = dist <= float(self._spec.signal_radius)

        roll = float(self._rng.random())

        if within_rad:
            # Inside detection radius
            if roll < self._spec.p_d:
                positive = True
                outcome = DetectionOutcome.POSITIVE
            else:
                positive = False
                outcome = DetectionOutcome.MISSED
        else:
            # Outside detection radius
            if roll < self._spec.p_fa:
                positive = True
                outcome = DetectionOutcome.FALSE_POSITIVE
            else:
                positive = False
                outcome = DetectionOutcome.NEGATIVE

        return DetectionResult(
            observer_node=observer_node,
            target_node=target_node,
            graph_distance=dist,
            within_radius=within_rad,
            outcome=outcome,
            positive_signal=positive,
            timestamp=timestamp,
        )
