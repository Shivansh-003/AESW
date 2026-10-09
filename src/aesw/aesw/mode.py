"""
Adaptive Search Mode Controller
================================
Implements the AESW Adaptive Mode Controller governing transitions between
LOCAL exploration and LONG_JUMP dispersion based on recent information acquisition.

Core Decision Logic:
             Recent Information
                    │
                    ▼
        Has useful information appeared?
                 /       \
               YES        NO
                │          │
                ▼          ▼
              LOCAL     Check W-step window
                           stagnation ≥ W?
                             │          │
                            YES         NO
                             │          │
                             ▼          ▼
                         LONG_JUMP    LOCAL

Architectural Invariants:
1. Mode transitions are ADAPTIVE to information acquisition, NEVER on a predetermined time schedule.
2. Initial mode is SearchMode.LOCAL with 0 stagnation.
3. Positive / novel evidence resets stagnation to 0 and keeps/returns mode to LOCAL.
4. Consecutive stagnation for W steps triggers transition to SearchMode.LONG_JUMP.
5. Mode controller ONLY determines the search mode; it does NOT execute jumps or select destination nodes.
6. Epistemic firewall: Zero access to true target location, ground-truth graph, or future transitions.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Any, Iterable, Optional, Union

from aesw.environment.observation import Observation
from aesw.memory.models import NodeEvidence
from aesw.memory.types import EvidencePolarity


class SearchMode(str, Enum):
    """Categorical search operational modes for an AESW walker.

    - LOCAL: Exploitative / local search concentrating steps near high-confidence cues.
    - LONG_JUMP: Dispersive search triggered by information stagnation over window W.
    """
    LOCAL = "LOCAL"
    LONG_JUMP = "LONG_JUMP"

    @property
    def is_local(self) -> bool:
        """Whether current mode is LOCAL."""
        return self == SearchMode.LOCAL

    @property
    def is_long_jump(self) -> bool:
        """Whether current mode is LONG_JUMP."""
        return self == SearchMode.LONG_JUMP


def validate_search_mode(mode: Union[str, SearchMode]) -> SearchMode:
    """Validate and convert an input to a SearchMode enum.

    Args:
        mode: String or enum representation of search mode.

    Returns:
        Validated SearchMode instance.

    Raises:
        ValueError: If mode is not a valid SearchMode.
    """
    if isinstance(mode, SearchMode):
        return mode
    try:
        return SearchMode(str(mode).upper())
    except (ValueError, KeyError):
        valid = [m.value for m in SearchMode]
        raise ValueError(f"Invalid search mode: {mode}. Must be one of {valid}")


class AdaptiveModeController:
    """Adaptive mode controller governing LOCAL vs LONG_JUMP operational states.

    Tracks whether useful information has been acquired across a sliding window of
    size W steps. Detects search stagnation and adaptively switches modes without
    relying on rigid periodic schedules.

    Attributes:
        window_size: Maximum recent-information horizon W (must be >= 1).
        mode: Current operational SearchMode (LOCAL or LONG_JUMP).
        stagnation_steps: Number of consecutive steps without useful new information.
        transition_count: Total number of mode transitions executed.
        last_information_time: Simulation timestamp when useful information was last acquired.
        lambda_hat: Observability reference for current graph churn estimate.
    """

    DEFAULT_WINDOW_SIZE: int = 5
    MAX_SEEN_SIGNATURES: int = 2000

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        initial_mode: Union[str, SearchMode] = SearchMode.LOCAL,
        lambda_hat: float = 0.0,
    ) -> None:
        """Initialize the adaptive mode controller.

        Args:
            window_size: Recent-history horizon W in discrete steps (must be >= 1).
            initial_mode: Initial search mode (defaults to LOCAL).
            lambda_hat: Optional initial graph churn rate reference for diagnostics.

        Raises:
            ValueError: If window_size < 1 or lambda_hat < 0.
        """
        if int(window_size) < 1:
            raise ValueError(f"Window size W must be strictly positive (>= 1), got {window_size}")
        if float(lambda_hat) < 0.0:
            raise ValueError(f"Initial churn rate lambda_hat must be non-negative, got {lambda_hat}")

        self._window_size: int = int(window_size)
        self._mode: SearchMode = validate_search_mode(initial_mode)
        self._stagnation_steps: int = 0
        self._recent_information: deque[bool] = deque(maxlen=self._window_size)
        self._last_information_time: Optional[int] = None
        self._last_timestamp: Optional[int] = None
        self._transition_count: int = 0
        self._step_count: int = 0
        self._lambda_hat: float = float(lambda_hat)

        # Bounded signature history to detect and ignore repeated stale evidence
        self._seen_signatures_order: deque[tuple[Any, ...]] = deque(maxlen=self.MAX_SEEN_SIGNATURES)
        self._seen_signatures: set[tuple[Any, ...]] = set()

    @property
    def window_size(self) -> int:
        """Maximum recent-information horizon W."""
        return self._window_size

    @property
    def mode(self) -> SearchMode:
        """Current operational search mode."""
        return self._mode

    @property
    def current_mode(self) -> SearchMode:
        """Convenience alias for mode."""
        return self._mode

    @property
    def is_local(self) -> bool:
        """Whether current mode is SearchMode.LOCAL."""
        return self._mode == SearchMode.LOCAL

    @property
    def is_long_jump(self) -> bool:
        """Whether current mode is SearchMode.LONG_JUMP."""
        return self._mode == SearchMode.LONG_JUMP

    @property
    def stagnation_steps(self) -> int:
        """Consecutive decision steps elapsed without useful new information."""
        return self._stagnation_steps

    @property
    def recent_information(self) -> list[bool]:
        """Snapshot of the sliding W-step information history (True = useful, False = none)."""
        return list(self._recent_information)

    @property
    def last_information_time(self) -> Optional[int]:
        """Simulation timestamp when useful information was last acquired, or None."""
        return self._last_information_time

    @property
    def transition_count(self) -> int:
        """Total number of mode transitions between LOCAL and LONG_JUMP."""
        return self._transition_count

    @property
    def step_count(self) -> int:
        """Total update steps processed by this controller."""
        return self._step_count

    @property
    def lambda_hat(self) -> float:
        """Current churn rate estimate reference."""
        return self._lambda_hat

    @lambda_hat.setter
    def lambda_hat(self, value: Union[float, Any]) -> None:
        """Update the diagnostic churn estimate reference."""
        if hasattr(value, "estimated_lambda"):
            rate = float(value.estimated_lambda)
        else:
            rate = float(value)
        if rate < 0.0:
            raise ValueError(f"lambda_hat must be non-negative, got {rate}")
        self._lambda_hat = rate

    def _register_signature(self, signature: tuple[Any, ...]) -> bool:
        """Register an evidence signature, returning True if novel, False if previously seen."""
        if signature in self._seen_signatures:
            return False

        if len(self._seen_signatures_order) >= self.MAX_SEEN_SIGNATURES:
            oldest = self._seen_signatures_order.popleft()
            self._seen_signatures.discard(oldest)

        self._seen_signatures_order.append(signature)
        self._seen_signatures.add(signature)
        return True

    def evaluate_useful_information(
        self,
        evidence: Optional[Union[NodeEvidence, Iterable[NodeEvidence], Observation]] = None,
        useful: Optional[bool] = None,
    ) -> bool:
        """Determine whether the provided evidence constitutes useful new information.

        Principles:
        - If useful is explicitly provided as a bool, it takes immediate precedence.
        - Positive sensor detections or positive evidence records constitute strong useful information,
          provided they are novel (not identical repetitions of already-processed records).
        - Negative evidence (sensor non-detection or absence confirmations) does NOT automatically
          count as strong positive information.
        - Empty collections contain zero useful information.

        Args:
            evidence: NodeEvidence, collection of NodeEvidence, or Observation snapshot.
            useful: Optional explicit override boolean.

        Returns:
            True if useful new information is present, False otherwise.
        """
        if useful is not None:
            return bool(useful)

        if evidence is None:
            return False

        # Case 1: Observation snapshot
        if isinstance(evidence, Observation):
            obs = evidence
            # Positive sensor reading is strong cue (even if noisy)
            if obs.detected or (obs.target_signal and obs.target_signal.detected) or (
                obs.target_signal and obs.target_signal.signal_strength > 0.0
            ):
                sig = (obs.current_node, obs.time, True)
                return self._register_signature(sig)
            # Negative observation does not count as strong positive information
            return False

        # Case 2: Single NodeEvidence
        if isinstance(evidence, NodeEvidence):
            items = [evidence]
        elif isinstance(evidence, Iterable):
            items = list(evidence)
        else:
            return False

        if not items:
            return False

        has_novel_positive = False
        for ev in items:
            if not isinstance(ev, NodeEvidence):
                continue

            # Check polarity / positive indication
            is_pos = (
                ev.polarity == EvidencePolarity.POSITIVE
                or ev.target_found
                or ev.signal_strength > 0.0
            )
            if is_pos:
                sig = (ev.node_id, ev.timestamp, ev.confidence, ev.signal_strength)
                if self._register_signature(sig):
                    has_novel_positive = True

        return has_novel_positive

    def update(
        self,
        timestamp: int,
        evidence: Optional[Union[NodeEvidence, Iterable[NodeEvidence], Observation]] = None,
        useful: Optional[bool] = None,
        lambda_hat: Optional[Union[float, Any]] = None,
    ) -> SearchMode:
        """Process a simulation step, evaluate information presence, and update search mode.

        Decision Rules:
        - If useful new information is present:
            stagnation resets to 0.
            recent_information appends True.
            mode becomes SearchMode.LOCAL.
        - If no useful information is present:
            stagnation increments by 1.
            recent_information appends False.
            if stagnation >= W:
                mode becomes SearchMode.LONG_JUMP.
            else:
                mode remains SearchMode.LOCAL.

        Args:
            timestamp: Current discrete simulation time step t (must be >= 0).
            evidence: Evidence or observation gathered during this step.
            useful: Explicit boolean override for information presence.
            lambda_hat: Optional churn estimate update for observability.

        Returns:
            Updated SearchMode.

        Raises:
            ValueError: If timestamp < 0 or timestamp < previous timestamp.
        """
        t = int(timestamp)
        if t < 0:
            raise ValueError(f"timestamp must be non-negative, got {t}")
        if self._last_timestamp is not None and t < self._last_timestamp:
            raise ValueError(
                f"timestamp ({t}) cannot decrease below previous timestamp ({self._last_timestamp})"
            )
        self._last_timestamp = t
        self._step_count += 1

        if lambda_hat is not None:
            self.lambda_hat = lambda_hat

        is_useful = self.evaluate_useful_information(evidence=evidence, useful=useful)

        prior_mode = self._mode

        if is_useful:
            self._stagnation_steps = 0
            self._last_information_time = t
            self._recent_information.append(True)
            self._mode = SearchMode.LOCAL
        else:
            self._stagnation_steps += 1
            self._recent_information.append(False)
            if self._stagnation_steps >= self._window_size:
                self._mode = SearchMode.LONG_JUMP
            else:
                self._mode = SearchMode.LOCAL

        if self._mode != prior_mode:
            self._transition_count += 1

        return self._mode

    def reset(self) -> None:
        """Reset the controller back to initial unobserved conditions."""
        self._mode = SearchMode.LOCAL
        self._stagnation_steps = 0
        self._recent_information.clear()
        self._last_information_time = None
        self._last_timestamp = None
        self._transition_count = 0
        self._step_count = 0
        self._seen_signatures.clear()
        self._seen_signatures_order.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert controller diagnostics to a dictionary."""
        return {
            "mode": self._mode.value,
            "window_size": self._window_size,
            "stagnation_steps": self._stagnation_steps,
            "recent_information": self.recent_information,
            "last_information_time": self._last_information_time,
            "transition_count": self._transition_count,
            "step_count": self._step_count,
            "lambda_hat": self._lambda_hat,
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveModeController(mode={self._mode.value}, "
            f"stagnation={self._stagnation_steps}/{self._window_size}, "
            f"transitions={self._transition_count})"
        )
