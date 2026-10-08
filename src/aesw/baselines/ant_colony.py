"""
Ant Colony Walk Policy (Baseline 6)
===================================
Pheromone-guided stochastic exploration baseline under partial observability.
Maintains private local trail pheromone memory on observed directed transitions,
applying evaporation and step reinforcement.
"""

from typing import Optional, Tuple
import numpy as np

from aesw.utils.reproducibility import create_rng
from aesw.baselines.types import BaselineType, ActionType
from aesw.baselines.actions import SearchAction
from aesw.baselines.base import BaselinePolicy, BaselineMetadata
from aesw.environment.observation import Observation


class AntColonyWalkPolicy(BaselinePolicy):
    """Pheromone-guided local random walk policy.

    Decision & Pheromone Model:
    1. Evaporation:
       At each decision step, all existing pheromone values evaporate:
       tau(u, v) <- max(tau_min, (1 - rho) * tau(u, v))
       where rho in [0, 1] is the evaporation rate.
    2. Candidate Scoring:
       For each available active candidate v from current node u:
       tau_val = tau(u, v) (defaults to tau_0 = 1.0 if unseen)
       score(v) = tau_val^alpha
    3. Probability Distribution:
       P(v) = score(v) / sum_{w} score(w)
    4. Reinforcement:
       Upon selecting edge (u, v):
       tau(u, v) <- tau(u, v) + delta_tau
       If a positive detection signal is observed (detected=True), an optional
       detection reinforcement bonus delta_tau_signal can also be added.

    Pheromone memory is strictly private to this walker.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        evaporation_rate: float = 0.05,
        reinforcement: float = 1.0,
        signal_reinforcement: float = 0.0,
        alpha: float = 1.0,
        initial_pheromone: float = 1.0,
        min_pheromone: float = 1e-4,
    ) -> None:
        if not (0.0 <= evaporation_rate <= 1.0):
            raise ValueError(f"evaporation_rate rho must be in [0, 1], got {evaporation_rate}")
        if reinforcement < 0.0:
            raise ValueError(f"reinforcement delta_tau must be non-negative, got {reinforcement}")
        if signal_reinforcement < 0.0:
            raise ValueError(f"signal_reinforcement must be non-negative, got {signal_reinforcement}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if initial_pheromone <= 0.0:
            raise ValueError(f"initial_pheromone must be strictly positive, got {initial_pheromone}")
        if min_pheromone <= 0.0:
            raise ValueError(f"min_pheromone must be strictly positive, got {min_pheromone}")

        self._seed = seed
        self._rho = float(evaporation_rate)
        self._delta_tau = float(reinforcement)
        self._delta_tau_signal = float(signal_reinforcement)
        self._alpha = float(alpha)
        self._tau_0 = float(initial_pheromone)
        self._tau_min = float(min_pheromone)

        self._rng: np.random.Generator = create_rng(seed)

        # Private pheromone table: directed edge (u, v) -> pheromone intensity
        self._pheromone: dict[Tuple[int | str, int | str], float] = {}

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.ANT_COLONY

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def pheromone_table(self) -> dict[Tuple[int | str, int | str], float]:
        """Read-only copy of internal pheromone table."""
        return dict(self._pheromone)

    @property
    def metadata(self) -> BaselineMetadata:
        return BaselineMetadata(
            name="Ant Colony Walk",
            baseline_type=BaselineType.ANT_COLONY,
            description="Pheromone-guided walk with private local trail memory, evaporation, and reinforcement.",
            uses_memory=True,
            uses_detection_signals=self._delta_tau_signal > 0.0,
            uses_communication=False,
            is_multi_walker=False,
            parameters={
                "evaporation_rate": self._rho,
                "reinforcement": self._delta_tau,
                "signal_reinforcement": self._delta_tau_signal,
                "alpha": self._alpha,
                "initial_pheromone": self._tau_0,
                "min_pheromone": self._tau_min,
            },
        )

    def get_pheromone(self, u: int | str, v: int | str) -> float:
        """Query pheromone on directed edge (u, v), returning default tau_0 if unvisited."""
        return self._pheromone.get((u, v), self._tau_0)

    def set_pheromone(self, u: int | str, v: int | str, value: float) -> None:
        """Manually set pheromone value for testing or initialization."""
        self._pheromone[(u, v)] = max(self._tau_min, float(value))

    def _evaporate(self) -> None:
        """Apply evaporation across all stored pheromone trails."""
        for edge, tau in self._pheromone.items():
            self._pheromone[edge] = max(self._tau_min, (1.0 - self._rho) * tau)

    def decide(self, observation: Observation) -> SearchAction:
        """Select next hop based on local pheromone weights, then evaporate and reinforce."""
        current_node = observation.current_node
        candidates = self.get_available_candidates(observation)

        if not candidates:
            # Evaporate trails even when stationary
            self._evaporate()
            action = SearchAction(
                action_type=ActionType.STAY,
                destination=current_node,
                metadata={"reason": "no_available_neighbors"},
            )
            return self._validate_action(action, observation)

        # 1. Evaporate stored pheromones
        self._evaporate()

        # 2. Compute candidate scores: score(v) = tau(u, v)^alpha
        taus = np.array([
            self.get_pheromone(current_node, v) for v in candidates
        ], dtype=np.float64)

        scores = np.power(taus, self._alpha)
        total_score = np.sum(scores)

        if total_score <= 0.0 or not np.isfinite(total_score):
            probabilities = np.ones(len(candidates), dtype=np.float64) / len(candidates)
        else:
            probabilities = scores / total_score

        # 3. Sample destination according to pheromone probability distribution
        chosen_idx = int(self._rng.choice(len(candidates), p=probabilities))
        chosen_destination = candidates[chosen_idx]

        # 4. Reinforce selected edge
        edge_key = (current_node, chosen_destination)
        current_tau = self.get_pheromone(current_node, chosen_destination)
        reinforcement_bonus = self._delta_tau
        if observation.detected and self._delta_tau_signal > 0.0:
            reinforcement_bonus += self._delta_tau_signal

        self._pheromone[edge_key] = current_tau + reinforcement_bonus

        action = SearchAction(
            action_type=ActionType.MOVE,
            destination=chosen_destination,
            metadata={
                "candidate_count": len(candidates),
                "chosen_pheromone": self._pheromone[edge_key],
                "candidate_probabilities": dict(zip(candidates, probabilities.tolist())),
            },
        )
        return self._validate_action(action, observation)

    def reset(self) -> None:
        """Clear pheromone table and re-seed PRNG."""
        self._pheromone.clear()
        self._rng = create_rng(self._seed)
