"""
Adaptive Exponential Decay Formulations
========================================
Implements the central adaptive memory decay model:
    w = exp(-lambda_hat * age)
governing information persistence under time-varying topological dynamics.
"""

from __future__ import annotations

import math
from typing import Union


def exponential_decay(age: Union[int, float], lambda_hat: float) -> float:
    """Calculate the adaptive evidence weight using the exponential decay model.

    Mathematical Formulation:
        w = exp(-lambda_hat * age)

    Behavioral Regimes:
    - Stable graph (lambda_hat -> 0): w -> 1.0 (evidence persists, slow decay).
    - Rapidly changing graph (lambda_hat >> 0): w decays rapidly towards 0.0.
    - Zero age (newly gathered evidence): w = 1.0 regardless of churn rate.

    Args:
        age: Elapsed simulation time steps since evidence acquisition (must be >= 0).
        lambda_hat: Estimated or configured graph churn / change rate (must be >= 0.0).

    Returns:
        Weight scalar w in the interval [0.0, 1.0].

    Raises:
        ValueError: If age is negative or lambda_hat is negative.
    """
    if age < 0:
        raise ValueError(f"Evidence age must be non-negative, got {age}")
    if lambda_hat < 0.0:
        raise ValueError(f"Estimated churn rate lambda_hat must be non-negative, got {lambda_hat}")

    # Boundary conditions
    if age == 0 or lambda_hat == 0.0:
        return 1.0

    exponent = -float(lambda_hat) * float(age)

    # Prevent numerical floating point underflow
    if exponent < -700.0:
        return 0.0

    return float(math.exp(exponent))


def compute_half_life(lambda_hat: float) -> float:
    """Compute the evidence half-life t_half corresponding to churn rate lambda_hat.

    Mathematical Formulation:
        t_half = ln(2) / lambda_hat

    Args:
        lambda_hat: Estimated churn rate (must be >= 0.0).

    Returns:
        Duration in time steps until evidence weight halves (inf if lambda_hat == 0.0).

    Raises:
        ValueError: If lambda_hat is negative.
    """
    if lambda_hat < 0.0:
        raise ValueError(f"Estimated churn rate lambda_hat must be non-negative, got {lambda_hat}")
    if lambda_hat == 0.0:
        return float("inf")

    return float(math.log(2.0) / lambda_hat)


def decay_rate_from_half_life(half_life: float) -> float:
    """Compute the churn decay rate lambda_hat required to produce a given half-life.

    Mathematical Formulation:
        lambda_hat = ln(2) / half_life

    Args:
        half_life: Desired half-life duration in steps (must be > 0.0, or inf).

    Returns:
        Corresponding churn rate lambda_hat (0.0 if half_life == inf).

    Raises:
        ValueError: If half_life is non-positive.
    """
    if half_life == float("inf"):
        return 0.0
    if half_life <= 0.0:
        raise ValueError(f"Half-life must be strictly positive, got {half_life}")

    return float(math.log(2.0) / half_life)
