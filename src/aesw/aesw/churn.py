"""
Adaptive Churn Estimator
========================
Provides local online estimation of dynamic graph volatility (churn rate lambda_hat)
from partial, locally observed neighborhood availability changes over time.

Mathematical Model:
    lambda_hat_t = (1 - eta) * lambda_hat_(t-1) + eta * (-ln(s) / Delta_t)

where:
    s       = |N_previous ∩ N_current| / |N_previous ∪ N_current|
              (Local Jaccard neighborhood overlap on inspected active links)
    Delta_t = t_current - t_previous > 0  (Elapsed observation time)
    eta     = smoothing factor in [0.0, 1.0]
    lambda_hat >= 0.0                     (Non-negative volatility estimate)

Epistemic Boundary Principle:
The estimator operates strictly under partial visibility. It has zero access
to ground-truth transition probabilities (p_on, p_off), the complete graph topology,
active graph views, or future states. All volatility estimates are derived solely
from walker-local observations.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Union

from aesw.environment.observation import Observation


DEFAULT_ETA: float = 0.2
DEFAULT_EPSILON: float = 1e-4


class ChurnEstimator:
    """Local online estimator of graph churn rate lambda_hat.

    Operates on sequences of locally observed active neighborhoods, estimating
    the apparent rate of link turnover using exponential smoothing over raw
    negative log-overlap divided by elapsed observation time.

    Attributes:
        eta: Smoothing weight parameter in [0.0, 1.0].
        epsilon: Positive numerical stability floor applied when overlap s = 0.
        previous_neighborhood: Immutable set of active neighbors from the prior observation.
        current_neighborhood: Immutable set of active neighbors from the most recent observation.
        previous_timestamp: Simulation timestamp of the prior observation.
        current_timestamp: Simulation timestamp of the most recent observation.
        overlap: Most recently calculated Jaccard neighborhood overlap s in [0.0, 1.0].
        raw_churn_rate: Most recently calculated instantaneous churn rate (-ln(s_safe) / Delta_t).
        estimated_lambda: Current smoothed graph volatility estimate lambda_hat >= 0.0.
        update_count: Total number of observations processed by this estimator.
    """

    def __init__(
        self,
        eta: float = DEFAULT_ETA,
        epsilon: float = DEFAULT_EPSILON,
        initial_lambda: float = 0.0,
    ) -> None:
        """Initialize the churn estimator.

        Args:
            eta: Smoothing factor in [0.0, 1.0]. eta=1.0 trusts latest measurement completely;
                eta=0.0 preserves previous estimate without update. Defaults to 0.2.
            epsilon: Strictly positive numerical floor for zero-overlap situations.
                Defaults to 1e-4.
            initial_lambda: Initial prior churn rate estimate (must be >= 0.0). Defaults to 0.0.

        Raises:
            ValueError: If eta is outside [0.0, 1.0], epsilon <= 0.0, or initial_lambda < 0.0.
        """
        if not (0.0 <= float(eta) <= 1.0):
            raise ValueError(f"Smoothing parameter eta must be in [0.0, 1.0], got {eta}")
        if float(epsilon) <= 0.0:
            raise ValueError(f"Numerical floor epsilon must be strictly positive, got {epsilon}")
        if float(initial_lambda) < 0.0:
            raise ValueError(
                f"Initial churn rate initial_lambda must be non-negative, got {initial_lambda}"
            )

        self._eta: float = float(eta)
        self._epsilon: float = float(epsilon)
        self._initial_lambda: float = float(initial_lambda)

        self._previous_neighborhood: Optional[frozenset[Union[int, str]]] = None
        self._current_neighborhood: Optional[frozenset[Union[int, str]]] = None
        self._previous_timestamp: Optional[Union[int, float]] = None
        self._current_timestamp: Optional[Union[int, float]] = None

        self._overlap: Optional[float] = None
        self._raw_churn_rate: Optional[float] = None
        self._estimated_lambda: float = float(initial_lambda)
        self._update_count: int = 0

    @property
    def eta(self) -> float:
        """Exponential smoothing factor in [0.0, 1.0]."""
        return self._eta

    @property
    def epsilon(self) -> float:
        """Numerical stability floor for zero-overlap handling."""
        return self._epsilon

    @property
    def initial_lambda(self) -> float:
        """Initial prior churn rate estimate."""
        return self._initial_lambda

    @property
    def previous_neighborhood(self) -> Optional[frozenset[Union[int, str]]]:
        """Previously observed active neighborhood set, or None if fewer than 2 updates."""
        return self._previous_neighborhood

    @property
    def current_neighborhood(self) -> Optional[frozenset[Union[int, str]]]:
        """Most recently observed active neighborhood set, or None before first update."""
        return self._current_neighborhood

    @property
    def previous_timestamp(self) -> Optional[Union[int, float]]:
        """Timestamp of prior observation, or None if fewer than 2 updates."""
        return self._previous_timestamp

    @property
    def current_timestamp(self) -> Optional[Union[int, float]]:
        """Timestamp of most recent observation, or None before first update."""
        return self._current_timestamp

    @property
    def overlap(self) -> Optional[float]:
        """Most recently computed neighborhood overlap s in [0.0, 1.0], or None on first update."""
        return self._overlap

    @property
    def raw_churn_rate(self) -> Optional[float]:
        """Instantaneous raw churn rate (-ln(s_safe) / Delta_t), or None on first update."""
        return self._raw_churn_rate

    @property
    def estimated_lambda(self) -> float:
        """Smoothed churn rate estimate lambda_hat >= 0.0."""
        return self._estimated_lambda

    @property
    def lambda_hat(self) -> float:
        """Convenience alias for estimated_lambda."""
        return self._estimated_lambda

    @property
    def update_count(self) -> int:
        """Total number of observation updates processed."""
        return self._update_count

    @staticmethod
    def compute_overlap(
        nbrs1: Iterable[Union[int, str]],
        nbrs2: Iterable[Union[int, str]],
    ) -> float:
        """Compute local Jaccard neighborhood overlap between two sets of nodes.

        Mathematical Formulation:
            s = |A ∩ B| / |A ∪ B|

        Boundary Conditions:
            - If both sets are empty, s = 1.0 (no evidence of change between empty states).
            - If exactly one set is empty and the other non-empty, s = 0.0.

        Args:
            nbrs1: First collection of neighbor identifiers.
            nbrs2: Second collection of neighbor identifiers.

        Returns:
            Jaccard overlap scalar s in [0.0, 1.0].
        """
        s1 = frozenset(nbrs1)
        s2 = frozenset(nbrs2)

        if not s1 and not s2:
            return 1.0

        union = s1 | s2
        intersection = s1 & s2
        return float(len(intersection)) / float(len(union))

    def update(
        self,
        neighborhood: Union[Observation, Iterable[Union[int, str]]],
        timestamp: Optional[Union[int, float]] = None,
    ) -> float:
        """Process a new local observation and update the smoothed churn estimate.

        Accepts either an Observation instance or an explicit iterable of neighbor identifiers.

        Args:
            neighborhood: Walker-local Observation snapshot or collection of neighbor identifiers.
            timestamp: Simulation timestamp (required if neighborhood is not an Observation).

        Returns:
            Updated smoothed churn estimate lambda_hat >= 0.0.

        Raises:
            ValueError: If timestamp is missing when passing raw collection, if timestamp < 0,
                or if current timestamp <= previous timestamp on subsequent updates.
        """
        if isinstance(neighborhood, Observation):
            obs = neighborhood
            resolved_time = float(obs.time) if timestamp is None else float(timestamp)
            # Extract locally checked and confirmed active neighbors
            curr_set = frozenset(
                nbr for nbr in obs.checked_neighbors
                if obs.is_edge_known_active(nbr)
            )
        else:
            if timestamp is None:
                raise ValueError("timestamp must be provided when neighborhood is not an Observation")
            resolved_time = float(timestamp)
            curr_set = frozenset(neighborhood)

        if resolved_time < 0.0:
            raise ValueError(f"Observation timestamp must be non-negative, got {resolved_time}")

        # First observation: initialize state without manufacturing artificial churn
        if self._update_count == 0:
            self._current_neighborhood = curr_set
            self._current_timestamp = resolved_time
            self._previous_neighborhood = None
            self._previous_timestamp = None
            self._overlap = None
            self._raw_churn_rate = None
            self._update_count = 1
            return self._estimated_lambda

        # Subsequent observations: validate positive elapsed time
        assert self._current_timestamp is not None
        delta_t = resolved_time - self._current_timestamp
        if delta_t <= 0.0:
            raise ValueError(
                f"Current timestamp ({resolved_time}) must be strictly greater than "
                f"previous timestamp ({self._current_timestamp}). Elapsed time Delta_t must be positive."
            )

        # Shift prior state
        prev_set = self._current_neighborhood
        prev_time = self._current_timestamp
        assert prev_set is not None

        self._previous_neighborhood = prev_set
        self._previous_timestamp = prev_time
        self._current_neighborhood = curr_set
        self._current_timestamp = resolved_time
        self._update_count += 1

        # Case D: Both empty neighborhoods contain no evidence of topological change
        if len(prev_set) == 0 and len(curr_set) == 0:
            s = 1.0
            raw_lambda = 0.0
        else:
            intersection = prev_set & curr_set
            union = prev_set | curr_set
            s = float(len(intersection)) / float(len(union))

            # Numerical floor handling for zero overlap (Case C, E, F)
            s_safe = max(s, self._epsilon)
            raw_lambda = -math.log(s_safe) / delta_t

        self._overlap = s
        self._raw_churn_rate = raw_lambda

        # Exponential smoothing: lambda_hat_t = (1 - eta) * lambda_hat_(t-1) + eta * raw_lambda
        smoothed = (1.0 - self._eta) * self._estimated_lambda + self._eta * raw_lambda

        # Guarantee non-negativity and finiteness
        self._estimated_lambda = max(0.0, float(smoothed))
        if not math.isfinite(self._estimated_lambda):
            self._estimated_lambda = 0.0

        return self._estimated_lambda

    def update_from_observation(self, observation: Observation) -> float:
        """Update churn estimate directly from a walker's local Observation snapshot.

        Extracts only locally inspected and confirmed active neighbors, strictly preserving
        the epistemic boundary.

        Args:
            observation: Partial observation snapshot.

        Returns:
            Updated smoothed churn rate lambda_hat >= 0.0.
        """
        return self.update(observation)

    def reset(self, initial_lambda: Optional[float] = None) -> None:
        """Reset estimator state back to unobserved initial conditions.

        Args:
            initial_lambda: Optional new prior churn rate. Defaults to original initial_lambda.

        Raises:
            ValueError: If initial_lambda is negative.
        """
        self._previous_neighborhood = None
        self._current_neighborhood = None
        self._previous_timestamp = None
        self._current_timestamp = None
        self._overlap = None
        self._raw_churn_rate = None
        self._update_count = 0

        if initial_lambda is not None:
            if float(initial_lambda) < 0.0:
                raise ValueError(
                    f"Initial churn rate initial_lambda must be non-negative, got {initial_lambda}"
                )
            self._estimated_lambda = float(initial_lambda)
        else:
            self._estimated_lambda = self._initial_lambda

    def clone(self) -> ChurnEstimator:
        """Create an independent copy with identical configuration and state."""
        copied = ChurnEstimator(
            eta=self._eta,
            epsilon=self._epsilon,
            initial_lambda=self._initial_lambda,
        )
        copied._previous_neighborhood = self._previous_neighborhood
        copied._current_neighborhood = self._current_neighborhood
        copied._previous_timestamp = self._previous_timestamp
        copied._current_timestamp = self._current_timestamp
        copied._overlap = self._overlap
        copied._raw_churn_rate = self._raw_churn_rate
        copied._estimated_lambda = self._estimated_lambda
        copied._update_count = self._update_count
        return copied

    def to_dict(self) -> dict[str, Any]:
        """Export estimator state dictionary for logging and diagnostics."""
        return {
            "eta": self._eta,
            "epsilon": self._epsilon,
            "estimated_lambda": self._estimated_lambda,
            "raw_churn_rate": self._raw_churn_rate,
            "overlap": self._overlap,
            "update_count": self._update_count,
            "previous_timestamp": self._previous_timestamp,
            "current_timestamp": self._current_timestamp,
            "previous_neighborhood": (
                sorted(self._previous_neighborhood, key=str)
                if self._previous_neighborhood is not None
                else None
            ),
            "current_neighborhood": (
                sorted(self._current_neighborhood, key=str)
                if self._current_neighborhood is not None
                else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"ChurnEstimator(estimated_lambda={self._estimated_lambda:.4f}, "
            f"eta={self._eta}, updates={self._update_count})"
        )
