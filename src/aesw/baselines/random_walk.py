"""
Random Walk Policy (Baseline 1)
===============================
Standard unbiased random walk baseline on dynamic graphs under partial observability.
Selects uniformly at random among locally observed active neighbors.
"""

from typing import Optional
import numpy as np

from aesw.utils.reproducibility import create_rng
from aesw.baselines.types import BaselineType, ActionType
from aesw.baselines.actions import SearchAction
from aesw.baselines.base import BaselinePolicy, BaselineMetadata
from aesw.environment.observation import Observation


class RandomWalkPolicy(BaselinePolicy):
    """Unbiased random walk policy.

    Decision Rule:
    At each discrete decision step from node u:
    - If valid observed active neighbors N_obs(u) exist:
        P(move to v) = 1 / |N_obs(u)| for all v in N_obs(u).
    - If N_obs(u) is empty (isolated or incident links OFF):
        STAY at current node u.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng: np.random.Generator = create_rng(seed)

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.RANDOM_WALK

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def metadata(self) -> BaselineMetadata:
        return BaselineMetadata(
            name="Random Walk",
            baseline_type=BaselineType.RANDOM_WALK,
            description="Unbiased uniform random walk over locally observed active neighbors.",
            uses_memory=False,
            uses_detection_signals=False,
            uses_communication=False,
            is_multi_walker=False,
        )

    def decide(self, observation: Observation) -> SearchAction:
        """Select next hop uniformly at random from available observed neighbors."""
        candidates = self.get_available_candidates(observation)

        if not candidates:
            action = SearchAction(
                action_type=ActionType.STAY,
                destination=observation.current_node,
                metadata={"reason": "no_available_neighbors"},
            )
            return self._validate_action(action, observation)

        # Uniform random selection
        idx = int(self._rng.integers(0, len(candidates)))
        chosen_destination = candidates[idx]

        action = SearchAction(
            action_type=ActionType.MOVE,
            destination=chosen_destination,
            metadata={"candidate_count": len(candidates), "chosen_index": idx},
        )
        return self._validate_action(action, observation)

    def reset(self) -> None:
        """Reset RNG stream reproducibly."""
        self._rng = create_rng(self._seed)
