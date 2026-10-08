"""
Non-Backtracking Walk Policy (Baseline 2)
=========================================
Search policy avoiding immediate reversal to the previous node whenever alternatives exist.
Implements strictly 1-step memory without global visited sets or target cues.
"""

from typing import Optional
import numpy as np

from aesw.utils.reproducibility import create_rng
from aesw.baselines.types import BaselineType, ActionType
from aesw.baselines.actions import SearchAction
from aesw.baselines.base import BaselinePolicy, BaselineMetadata
from aesw.environment.observation import Observation


class NonBacktrackingWalkPolicy(BaselinePolicy):
    """Non-backtracking random walk policy.

    Decision Rule:
    Let u be the current node, p be the previous node (p = previous_node), and
    N_obs(u) be the available observed active neighbors:
    - If N_obs(u) is empty: STAY at current node.
    - Let N' = N_obs(u) \\ {p}.
    - If N' is non-empty: choose uniformly at random from N'.
    - If N' is empty (dead end / degree 1): fallback to choose uniformly from N_obs(u) (i.e. backtrack to p).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng: np.random.Generator = create_rng(seed)
        self._previous_node: Optional[int | str] = None

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.NON_BACKTRACKING

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def previous_node(self) -> Optional[int | str]:
        """Preceding vertex occupied by the walker, or None if initial step."""
        return self._previous_node

    @property
    def metadata(self) -> BaselineMetadata:
        return BaselineMetadata(
            name="Non-Backtracking Walk",
            baseline_type=BaselineType.NON_BACKTRACKING,
            description="1-step memory walk avoiding immediate reversal to previous node.",
            uses_memory=True,
            uses_detection_signals=False,
            uses_communication=False,
            is_multi_walker=False,
            parameters={"memory_depth": 1},
        )

    def decide(self, observation: Observation) -> SearchAction:
        """Select next hop, filtering out immediate predecessor if possible."""
        candidates = self.get_available_candidates(observation)

        if not candidates:
            # Cannot move; stay at current node
            action = SearchAction(
                action_type=ActionType.STAY,
                destination=observation.current_node,
                metadata={"reason": "no_available_neighbors", "previous_node": self._previous_node},
            )
            return self._validate_action(action, observation)

        # Filter out immediate predecessor
        non_backtracking_candidates = [
            nbr for nbr in candidates if nbr != self._previous_node
        ]

        if non_backtracking_candidates:
            eligible = non_backtracking_candidates
            backtracked = False
        else:
            # Fallback to backtracking if no other valid neighbors exist
            eligible = candidates
            backtracked = True

        idx = int(self._rng.integers(0, len(eligible)))
        chosen_destination = eligible[idx]

        # Update 1-step memory to current position
        self._previous_node = observation.current_node

        action = SearchAction(
            action_type=ActionType.MOVE,
            destination=chosen_destination,
            metadata={
                "candidate_count": len(candidates),
                "eligible_count": len(eligible),
                "backtracked": backtracked,
                "previous_node": self._previous_node,
            },
        )
        return self._validate_action(action, observation)

    def reset(self) -> None:
        """Clear 1-step memory and reset PRNG stream."""
        self._previous_node = None
        self._rng = create_rng(self._seed)
