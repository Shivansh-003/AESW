"""
Baseline Factory and Dispatcher
===============================
Provides uniform construction of baseline search policies from configuration or parameters.
"""

from typing import Any, Optional

from aesw.baselines.types import BaselineType
from aesw.baselines.base import BaselinePolicy
from aesw.baselines.random_walk import RandomWalkPolicy
from aesw.baselines.non_backtracking import NonBacktrackingWalkPolicy
from aesw.baselines.independent_walkers import IndependentRandomWalkers
from aesw.baselines.degree_based import DegreeBasedWalkPolicy
from aesw.baselines.flooding import FloodingPolicy
from aesw.baselines.ant_colony import AntColonyWalkPolicy


def create_baseline(
    baseline_type: str | BaselineType,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> BaselinePolicy:
    """Create a baseline search policy instance.

    Args:
        baseline_type: BaselineType enum or equivalent string identifier.
        seed: Isolated PRNG seed for the policy.
        **kwargs: Policy-specific hyperparameters.

    Returns:
        Configured BaselinePolicy instance.

    Raises:
        ValueError: If baseline_type is unrecognized.
    """
    if isinstance(baseline_type, str):
        try:
            b_type = BaselineType(baseline_type.lower().strip())
        except ValueError:
            valid_types = [t.value for t in BaselineType]
            raise ValueError(
                f"Unknown baseline type '{baseline_type}'. Supported types: {valid_types}"
            )
    elif isinstance(baseline_type, BaselineType):
        b_type = baseline_type
    else:
        raise TypeError(f"baseline_type must be a str or BaselineType, got {type(baseline_type).__name__}")

    if b_type == BaselineType.RANDOM_WALK:
        return RandomWalkPolicy(seed=seed)

    elif b_type == BaselineType.NON_BACKTRACKING:
        return NonBacktrackingWalkPolicy(seed=seed)

    elif b_type == BaselineType.INDEPENDENT_RANDOM_WALKERS:
        k = kwargs.get("k", 4)
        return IndependentRandomWalkers(k=k, base_seed=seed)

    elif b_type == BaselineType.DEGREE_BASED:
        epsilon = kwargs.get("epsilon", 1e-3)
        default_prior = kwargs.get("default_prior_degree", 1.0)
        return DegreeBasedWalkPolicy(seed=seed, epsilon=epsilon, default_prior_degree=default_prior)

    elif b_type == BaselineType.FLOODING:
        return FloodingPolicy(seed=seed)

    elif b_type == BaselineType.ANT_COLONY:
        evaporation_rate = kwargs.get("evaporation_rate", 0.05)
        reinforcement = kwargs.get("reinforcement", 1.0)
        signal_reinforcement = kwargs.get("signal_reinforcement", 0.0)
        alpha = kwargs.get("alpha", 1.0)
        initial_pheromone = kwargs.get("initial_pheromone", 1.0)
        min_pheromone = kwargs.get("min_pheromone", 1e-4)
        return AntColonyWalkPolicy(
            seed=seed,
            evaporation_rate=evaporation_rate,
            reinforcement=reinforcement,
            signal_reinforcement=signal_reinforcement,
            alpha=alpha,
            initial_pheromone=initial_pheromone,
            min_pheromone=min_pheromone,
        )

    raise ValueError(f"Unhandled baseline type '{b_type}'")
