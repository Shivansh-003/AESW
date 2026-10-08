"""
Dynamics Configuration Loader and Factory
=========================================
Factory functions to instantiate DynamicGraphState instances directly from
DynamicsSpecification or configuration dictionaries.
"""

from typing import Any, Mapping, Optional
from aesw.environment.types import DynamicRegime
from aesw.environment.problem import DynamicsSpecification
from aesw.graph.base import GraphInstance
from aesw.dynamics.types import InitializationPolicy
from aesw.dynamics.engine import DynamicGraphState


# Default experimental transition rate presets for named regimes
# Note: These values serve as configurable defaults in configs/dynamics.yaml
REGIME_PRESETS: dict[DynamicRegime, tuple[float, float]] = {
    DynamicRegime.STATIC: (0.0, 0.0),
    DynamicRegime.SLOW: (0.01, 0.005),
    DynamicRegime.MEDIUM: (0.05, 0.025),
    DynamicRegime.FAST: (0.1, 0.05),
    DynamicRegime.VERY_FAST: (0.2, 0.1),
}


def create_dynamic_graph(
    static_graph: GraphInstance,
    regime: DynamicRegime | str = DynamicRegime.MEDIUM,
    p_on: Optional[float] = None,
    p_off: Optional[float] = None,
    initialization_policy: InitializationPolicy = InitializationPolicy.ALL_ON,
    seed: Optional[int] = 42,
) -> DynamicGraphState:
    """Create and initialize a DynamicGraphState instance.

    If p_on or p_off are not explicitly provided, they are resolved from the
    preset parameters for the selected regime.

    Args:
        static_graph: Permanent underlying GraphInstance.
        regime: Named churn regime (STATIC, SLOW, MEDIUM, FAST, VERY_FAST).
        p_on: Optional explicit override for p_on.
        p_off: Optional explicit override for p_off.
        initialization_policy: Edge state initialization policy at t = 0.
        seed: Random seed for deterministic transitions.

    Returns:
        Configured DynamicGraphState instance.
    """
    if isinstance(regime, str):
        try:
            regime_enum = DynamicRegime(regime.upper())
        except ValueError:
            raise ValueError(f"Unknown dynamic regime: '{regime}'. Valid regimes: {[r.value for r in DynamicRegime]}")
    else:
        regime_enum = regime

    preset_p_on, preset_p_off = REGIME_PRESETS.get(regime_enum, (0.05, 0.025))
    resolved_p_on = preset_p_on if p_on is None else float(p_on)
    resolved_p_off = preset_p_off if p_off is None else float(p_off)

    return DynamicGraphState(
        static_graph=static_graph,
        p_on=resolved_p_on,
        p_off=resolved_p_off,
        regime=regime_enum,
        initialization_policy=initialization_policy,
        seed=seed,
    )


def create_dynamic_graph_from_spec(
    static_graph: GraphInstance,
    spec: DynamicsSpecification,
    initialization_policy: InitializationPolicy = InitializationPolicy.ALL_ON,
    seed: Optional[int] = 42,
) -> DynamicGraphState:
    """Create a DynamicGraphState instance from a validated DynamicsSpecification."""
    return create_dynamic_graph(
        static_graph=static_graph,
        regime=spec.regime,
        p_on=spec.p_on,
        p_off=spec.p_off,
        initialization_policy=initialization_policy,
        seed=seed,
    )


def create_dynamic_graph_from_config(
    static_graph: GraphInstance,
    dynamics_config: Mapping[str, Any],
    initialization_policy: InitializationPolicy = InitializationPolicy.ALL_ON,
    seed: Optional[int] = 42,
) -> DynamicGraphState:
    """Create a DynamicGraphState instance from parsed dynamics.yaml dictionary."""
    d_data = dynamics_config.get("dynamics", {})
    regime_str = d_data.get("regime", "MEDIUM")
    p_on = d_data.get("p_on")
    p_off = d_data.get("p_off")

    return create_dynamic_graph(
        static_graph=static_graph,
        regime=regime_str,
        p_on=p_on,
        p_off=p_off,
        initialization_policy=initialization_policy,
        seed=seed,
    )
