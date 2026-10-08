"""
Dynamic Graph Types and Initialization Policies
================================================
Defines enumeration types for edge initialization policies and transition outcomes.
"""

from enum import Enum


class InitializationPolicy(str, Enum):
    """Policy for setting edge operational states at simulation onset t = 0.

    - ALL_ON: All underlying edges start active (s_0(e) = ON).
    - ALL_OFF: All underlying edges start inactive (s_0(e) = OFF).
    - STATIONARY: Each edge is sampled independently from the stationary distribution
      of the two-state Markov chain: pi_ON = p_on / (p_on + p_off).
      (If p_on == p_off == 0, defaults to ALL_ON).
    """
    ALL_ON = "ALL_ON"
    ALL_OFF = "ALL_OFF"
    STATIONARY = "STATIONARY"
