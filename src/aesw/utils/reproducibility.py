"""
Reproducibility Utilities
=========================
Foundational helpers for deterministic random state management across experiments.
Avoids uncontrolled global randomness by encapsulating explicit RNG seeds.
"""

from typing import Optional
import random
import numpy as np


def seed_everything(seed: Optional[int] = None) -> int:
    """Set seeds for standard random and numpy generators.

    Args:
        seed: Integer seed value. If None, 42 is used as the default reproducible seed.

    Returns:
        The integer seed used.

    Raises:
        TypeError: If seed cannot be interpreted as an integer.
    """
    if seed is None:
        seed = 42

    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    seed_val = int(seed)
    random.seed(seed_val)
    np.random.seed(seed_val)
    return seed_val


def create_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create an explicit NumPy Generator instance for isolated deterministic streams.

    Future simulation modules should accept explicit Generator objects rather
    than calling global random methods, guaranteeing strict reproducibility.

    Args:
        seed: Optional integer seed for the generator.

    Returns:
        np.random.Generator: An isolated Generator instance.
    """
    return np.random.default_rng(seed)
