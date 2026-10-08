"""
Formal Problem Definition Specification
=======================================
Central specification aggregating and validating all environment, dynamics,
target, walker, observation, and cost specifications.

Acts as the immutable computational contract for search algorithms and simulation components.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping
from aesw.environment.types import TargetMode, DynamicRegime
from aesw.environment.models import DelaySpecification


@dataclass(frozen=True)
class GraphSpecification:
    """Formal parameters defining the base graph topology."""
    graph_type: str = "er"
    num_nodes: int = 100
    seed: int = 42
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {self.num_nodes}")
        if not self.graph_type or not self.graph_type.strip():
            raise ValueError("graph_type must not be empty")


@dataclass(frozen=True)
class DynamicsSpecification:
    """Formal parameters defining temporal edge transitions and churn regimes."""
    regime: DynamicRegime = DynamicRegime.MEDIUM
    p_on: float = 0.05
    p_off: float = 0.025
    edge_update_behavior: str = "markovian"

    def __post_init__(self) -> None:
        if not isinstance(self.regime, DynamicRegime):
            raise TypeError(f"regime must be a DynamicRegime enum, got {type(self.regime).__name__}")
        if not (0.0 <= self.p_on <= 1.0):
            raise ValueError(f"p_on must be in [0.0, 1.0], got {self.p_on}")
        if not (0.0 <= self.p_off <= 1.0):
            raise ValueError(f"p_off must be in [0.0, 1.0], got {self.p_off}")


@dataclass(frozen=True)
class TargetSpecification:
    """Formal parameters defining target locality and locomotion."""
    mode: TargetMode = TargetMode.MOVING
    p_move: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TargetMode):
            raise TypeError(f"mode must be a TargetMode enum, got {type(self.mode).__name__}")
        if not (0.0 <= self.p_move <= 1.0):
            raise ValueError(f"p_move must be in [0.0, 1.0], got {self.p_move}")


@dataclass(frozen=True)
class DetectionSpecification:
    """Formal parameters defining noisy target sensing."""
    signal_radius: int = 1
    p_d: float = 0.9
    p_fa: float = 0.05

    def __post_init__(self) -> None:
        if self.signal_radius < 0:
            raise ValueError(f"signal_radius s must be non-negative, got {self.signal_radius}")
        if not (0.0 <= self.p_d <= 1.0):
            raise ValueError(f"p_d must be in [0.0, 1.0], got {self.p_d}")
        if not (0.0 <= self.p_fa <= 1.0):
            raise ValueError(f"p_fa must be in [0.0, 1.0], got {self.p_fa}")


@dataclass(frozen=True)
class ObservationSpecification:
    """Formal parameters defining walker observation budget."""
    neighbor_budget: int = 4

    def __post_init__(self) -> None:
        if self.neighbor_budget <= 0:
            raise ValueError(f"neighbor_budget B must be strictly positive, got {self.neighbor_budget}")


@dataclass(frozen=True)
class WalkerSpecification:
    """Formal parameters defining multi-agent searcher team."""
    count: int = 5

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError(f"walker count must be strictly positive, got {self.count}")


@dataclass(frozen=True)
class SimulationSpecification:
    """Formal parameters defining simulation execution bounds."""
    time_horizon: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.time_horizon <= 0:
            raise ValueError(f"time_horizon must be strictly positive, got {self.time_horizon}")


@dataclass(frozen=True)
class CostSpecification:
    """Formal parameters defining the cost model C_total = M + c_m * Q."""
    c_m: float = 0.1

    def __post_init__(self) -> None:
        if self.c_m < 0.0:
            raise ValueError(f"communication cost coefficient c_m must be non-negative, got {self.c_m}")

    def compute_cost(self, num_moves: int, num_messages: int) -> float:
        """Compute total cost according to C_total = M + c_m * Q.

        Args:
            num_moves: Number of physical movements M.
            num_messages: Number of exchanged messages Q.

        Returns:
            Total evaluated cost.
        """
        if num_moves < 0:
            raise ValueError(f"num_moves must be non-negative, got {num_moves}")
        if num_messages < 0:
            raise ValueError(f"num_messages must be non-negative, got {num_messages}")
        return float(num_moves + self.c_m * num_messages)


@dataclass(frozen=True)
class ProblemDefinition:
    """Complete, immutable formal computational specification for dynamic graph search.

    This object serves as the formal contract that all future simulation engines,
    graph generators, baseline algorithms, and AESW implementations must obey.

    Attributes:
        graph: Base graph topology specification.
        dynamics: Temporal edge transition parameters.
        target: Target locomotion specification.
        detection: Noisy sensor model parameters.
        observation: Observation horizon and neighbor check budget B.
        delay: Traversal delay specification.
        walker: Searcher team specification.
        simulation: Time horizon and random seed specification.
        cost: Cost model specification.
    """
    graph: GraphSpecification = field(default_factory=GraphSpecification)
    dynamics: DynamicsSpecification = field(default_factory=DynamicsSpecification)
    target: TargetSpecification = field(default_factory=TargetSpecification)
    detection: DetectionSpecification = field(default_factory=DetectionSpecification)
    observation: ObservationSpecification = field(default_factory=ObservationSpecification)
    delay: DelaySpecification = field(default_factory=DelaySpecification)
    walker: WalkerSpecification = field(default_factory=WalkerSpecification)
    simulation: SimulationSpecification = field(default_factory=SimulationSpecification)
    cost: CostSpecification = field(default_factory=CostSpecification)

    def to_dict(self) -> dict[str, Any]:
        """Convert specification into a deterministic dictionary representation."""
        data = asdict(self)
        # Ensure enums are converted to primitive strings
        data["dynamics"]["regime"] = self.dynamics.regime.value
        data["target"]["mode"] = self.target.mode.value
        return data

    @classmethod
    def from_configs(
        cls,
        default_config: Mapping[str, Any],
        graph_config: Mapping[str, Any],
        dynamics_config: Mapping[str, Any],
        experiments_config: Mapping[str, Any],
    ) -> "ProblemDefinition":
        """Construct and validate a ProblemDefinition from parsed YAML configuration dictionaries.

        Args:
            default_config: Content from default.yaml.
            graph_config: Content from graph.yaml.
            dynamics_config: Content from dynamics.yaml.
            experiments_config: Content from experiments.yaml.

        Returns:
            Validated, immutable ProblemDefinition instance.
        """
        # 1. Graph spec
        g_data = graph_config.get("graph", {})
        g_type = g_data.get("type", "er")
        graph_spec = GraphSpecification(
            graph_type=g_type,
            num_nodes=g_data.get("num_nodes", 100),
            seed=g_data.get("seed", 42),
            parameters=g_data.get(g_type, {}),
        )

        # 2. Dynamics spec
        d_data = dynamics_config.get("dynamics", {})
        regime_str = d_data.get("regime", "MEDIUM").upper()
        dynamics_spec = DynamicsSpecification(
            regime=DynamicRegime(regime_str),
            p_on=float(d_data.get("p_on", 0.05)),
            p_off=float(d_data.get("p_off", 0.025)),
            edge_update_behavior=d_data.get("edge_update_behavior", "markovian"),
        )

        # 3. Target spec
        exp_data = experiments_config.get("experiments", {})
        t_data = exp_data.get("target", {})
        mode_str = t_data.get("mode", "moving").upper()
        target_spec = TargetSpecification(
            mode=TargetMode(mode_str),
            p_move=float(t_data.get("movement_rate", 0.05)),
        )

        # 4. Detection spec
        sensor_noise = float(t_data.get("sensor_noise", 0.1))
        p_d = max(0.0, min(1.0, 1.0 - sensor_noise))
        p_fa = float(t_data.get("false_alarm_rate", 0.05))
        detection_spec = DetectionSpecification(
            signal_radius=1,
            p_d=p_d,
            p_fa=p_fa,
        )

        # 5. Observation spec
        observation_spec = ObservationSpecification(neighbor_budget=4)

        # 6. Delay spec
        delay_spec = DelaySpecification(
            distribution_type="constant",
            min_delay=1,
            max_delay=1,
            mean_delay=1.0,
        )

        # 7. Walker spec
        w_data = exp_data.get("walker", {})
        walker_spec = WalkerSpecification(count=int(w_data.get("count", 5)))

        # 8. Simulation spec
        time_horizon = int(d_data.get("time_horizon", 1000))
        seed = int(default_config.get("reproducibility", {}).get("seed", 42))
        simulation_spec = SimulationSpecification(
            time_horizon=time_horizon,
            seed=seed,
        )

        # 9. Cost spec
        cost_spec = CostSpecification(c_m=0.1)

        return cls(
            graph=graph_spec,
            dynamics=dynamics_spec,
            target=target_spec,
            detection=detection_spec,
            observation=observation_spec,
            delay=delay_spec,
            walker=walker_spec,
            simulation=simulation_spec,
            cost=cost_spec,
        )
