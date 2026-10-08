"""
Dynamics Module
===============
Dynamic graph behavior and temporal edge evolution subsystem for Adaptive Graph Search.

Provides stochastic discrete-time edge transition modeling G_t = (V, E_t) where
edges transition independently between active (ON) and inactive (OFF) states according to:
- P(s_{t+1}(e) = OFF | s_t(e) = ON) = p_off
- P(s_{t+1}(e) = ON  | s_t(e) = OFF) = p_on

Preserves underlying static GraphInstance G = (V, E) immutability.
"""

from aesw.environment.types import DynamicRegime
from aesw.dynamics.types import InitializationPolicy
from aesw.dynamics.models import TransitionStatistics, DynamicGraphSnapshot
from aesw.dynamics.engine import DynamicGraphState
from aesw.dynamics.view import ActiveGraphView
from aesw.dynamics.factory import (
    REGIME_PRESETS,
    create_dynamic_graph,
    create_dynamic_graph_from_spec,
    create_dynamic_graph_from_config,
)

__all__ = [
    # Types & Models
    "DynamicRegime",
    "InitializationPolicy",
    "TransitionStatistics",
    "DynamicGraphSnapshot",
    # Core Engine & View
    "DynamicGraphState",
    "ActiveGraphView",
    # Factory & Presets
    "REGIME_PRESETS",
    "create_dynamic_graph",
    "create_dynamic_graph_from_spec",
    "create_dynamic_graph_from_config",
]
