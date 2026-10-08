"""
k Independent Random Walkers (Baseline 3)
=========================================
Multi-agent baseline consisting of k independent RandomWalkPolicy instances.
Walkers operate in parallel with zero communication, zero shared memory, and
isolated PRNG streams.
"""

from typing import Mapping, Optional
from aesw.baselines.types import BaselineType
from aesw.baselines.actions import SearchAction
from aesw.baselines.base import BaselinePolicy, BaselineMetadata
from aesw.baselines.random_walk import RandomWalkPolicy
from aesw.environment.observation import Observation


class IndependentRandomWalkers(BaselinePolicy):
    """Multi-walker baseline managing k independent random walk policies.

    Architectural Rule:
    - No evidence sharing, no pheromone, no communication.
    - Each walker i maintains its own isolated RandomWalkPolicy with independent PRNG stream.
    - Tests the scaling hypothesis: does increasing searcher count without intelligence
      or coordination resolve dynamic graph search?
    """

    def __init__(self, k: int = 4, base_seed: Optional[int] = None) -> None:
        if k <= 0:
            raise ValueError(f"Number of walkers k must be strictly positive, got {k}")

        self._k = int(k)
        self._base_seed = base_seed

        # Initialize k independent policies with distinct isolated seeds
        self._walkers: tuple[RandomWalkPolicy, ...] = tuple(
            RandomWalkPolicy(seed=(base_seed + i * 1000) if base_seed is not None else None)
            for i in range(self._k)
        )

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.INDEPENDENT_RANDOM_WALKERS

    @property
    def seed(self) -> Optional[int]:
        return self._base_seed

    @property
    def k(self) -> int:
        """Total number of independent walkers in the team."""
        return self._k

    @property
    def walkers(self) -> tuple[RandomWalkPolicy, ...]:
        """Underlying individual walker policy instances."""
        return self._walkers

    @property
    def metadata(self) -> BaselineMetadata:
        return BaselineMetadata(
            name=f"{self._k} Independent Random Walkers",
            baseline_type=BaselineType.INDEPENDENT_RANDOM_WALKERS,
            description="Parallel independent random walkers with zero inter-agent communication.",
            uses_memory=False,
            uses_detection_signals=False,
            uses_communication=False,
            is_multi_walker=True,
            parameters={"k": self._k},
        )

    def decide(self, observation: Observation) -> SearchAction:
        """Default decision delegate using walker 0."""
        return self._walkers[0].decide(observation)

    def decide_walker(self, walker_index: int, observation: Observation) -> SearchAction:
        """Decide an action for a specific walker in the team.

        Args:
            walker_index: Index in [0, k-1].
            observation: The observation perceived by walker_index.

        Returns:
            Validated SearchAction chosen by that walker.
        """
        if not (0 <= walker_index < self._k):
            raise IndexError(f"walker_index {walker_index} out of bounds for k={self._k}")
        return self._walkers[walker_index].decide(observation)

    def decide_all(self, observations: Mapping[int, Observation]) -> dict[int, SearchAction]:
        """Decide actions for all walkers given their individual observations."""
        actions: dict[int, SearchAction] = {}
        for i in range(self._k):
            if i in observations:
                actions[i] = self.decide_walker(i, observations[i])
        return actions

    def reset_walker(self, walker_index: int) -> None:
        """Reset only the specified walker without altering other walkers' PRNG states."""
        if not (0 <= walker_index < self._k):
            raise IndexError(f"walker_index {walker_index} out of bounds for k={self._k}")
        self._walkers[walker_index].reset()

    def reset(self) -> None:
        """Reset all k individual walker policies."""
        for walker in self._walkers:
            walker.reset()
