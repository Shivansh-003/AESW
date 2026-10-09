"""
AESW Autonomous Agent and Multi-Walker Policy
=============================================
Orchestrates the Adaptive Evidence-Sharing Walkers (AESW) search algorithm by
composing the verified subcomponents:
- Evidence Memory & Exponential Decay Cache (EvidenceCache)
- Online Adaptive Churn Estimator (ChurnEstimator)
- Node-Mediated Information Sharing (NodeMediatedExchange)
- Adaptive Mode Controller (AdaptiveModeController)
- Next-Hop Candidate Scorer & Softmax Selector (NextHopDecisionEngine)

Architectural Invariants:
1. Epistemic Firewall: Agents operate strictly on local partial observations.
   No access to hidden dynamic edge states, transition probabilities (p_on, p_off),
   or ground-truth target coordinates.
2. Independent Multi-Walker State: Each walker maintains its own isolated EvidenceCache,
   ChurnEstimator, AdaptiveModeController, visit counts, and PRNG stream.
3. Node-Mediated Communication: Walkers never communicate directly point-to-point.
   All shared clues are published to and retrieved from localized node caches
   (SharedNodeCache via NodeMediatedExchange).
4. Environment-Mediated Long Jumps: When stagnation occurs, the agent requests a JUMP
   action (destination=None), and the simulation environment coordinator executes the
   global hop using its isolated PRNG stream.
5. Deterministic & Reproducible: Isolated seed streams guarantee bit-for-bit repeatability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np

from aesw.baselines.actions import SearchAction
from aesw.baselines.base import BaselineMetadata, BaselinePolicy
from aesw.baselines.types import ActionType, BaselineType
from aesw.environment.observation import Observation
from aesw.memory.cache import EvidenceCache
from aesw.memory.models import NodeEvidence
from aesw.memory.types import EvidencePolarity, EvidenceSource
from aesw.aesw.churn import DEFAULT_EPSILON, DEFAULT_ETA, ChurnEstimator
from aesw.aesw.communication import (
    CommunicationMode,
    NodeMediatedExchange,
    validate_communication_mode,
)
from aesw.aesw.mode import AdaptiveModeController, SearchMode
from aesw.aesw.next_hop import NextHopDecision, NextHopDecisionEngine


class AESWAgent:
    """Autonomous single-walker AESW search agent.

    Maintains walker-private cognitive state including local decaying evidence memory,
    online churn estimation, stagnation detection, and multi-criteria candidate scoring.
    """

    def __init__(
        self,
        walker_id: Union[int, str] = 0,
        seed: Optional[int] = None,
        eta: float = DEFAULT_ETA,
        epsilon: float = DEFAULT_EPSILON,
        window_size: int = AdaptiveModeController.DEFAULT_WINDOW_SIZE,
        weight_positive: float = 1.0,
        weight_novelty: float = 0.5,
        weight_revisit: float = 0.5,
        weight_delay: float = 0.2,
        temperature: float = 1.0,
        mode_weight_scaling: bool = True,
        default_lambda_hat: float = 0.05,
        max_cache_capacity: Optional[int] = None,
    ) -> None:
        """Initialize an autonomous AESW walker agent.

        Args:
            walker_id: Unique identifier for this walker.
            seed: Isolated PRNG seed for stochastic action selection.
            eta: Churn estimator exponential smoothing weight in [0, 1].
            epsilon: Numerical stability floor for churn estimation.
            window_size: Stagnation detection window W >= 1.
            weight_positive: Candidate scorer weight alpha for decayed positive evidence.
            weight_novelty: Candidate scorer weight beta for vertex novelty.
            weight_revisit: Candidate scorer penalty weight gamma for frequent revisits.
            weight_delay: Candidate scorer penalty weight delta for edge traversal delay.
            temperature: Softmax exploration temperature T > 0.
            mode_weight_scaling: Whether to scale scoring weights during LONG_JUMP mode.
            default_lambda_hat: Fallback churn rate when insufficient history is observed.
            max_cache_capacity: Optional upper bound on local evidence cache capacity.
        """
        self._walker_id = walker_id
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Private cognitive state
        self._default_lambda_hat = float(default_lambda_hat)
        self._cache = EvidenceCache(
            default_lambda_hat=self._default_lambda_hat,
            max_capacity=max_cache_capacity,
        )
        self._churn_estimator = ChurnEstimator(
            eta=eta,
            epsilon=epsilon,
            initial_lambda=self._default_lambda_hat,
        )
        self._mode_controller = AdaptiveModeController(
            window_size=window_size,
            initial_mode=SearchMode.LOCAL,
            lambda_hat=self._default_lambda_hat,
        )
        self._decision_engine = NextHopDecisionEngine(
            weight_positive=weight_positive,
            weight_novelty=weight_novelty,
            weight_revisit=weight_revisit,
            weight_delay=weight_delay,
            temperature=temperature,
            mode_weight_scaling=mode_weight_scaling,
        )

        # Local visit counts and traversal history (walker-private)
        self._visit_counts: dict[Union[int, str], int] = {}
        self._traversal_delays: dict[Union[int, str], float] = {}
        self._step_trace: list[dict[str, Any]] = []
        self._last_decision: Optional[NextHopDecision] = None

    @property
    def walker_id(self) -> Union[int, str]:
        """Walker identifier."""
        return self._walker_id

    @property
    def cache(self) -> EvidenceCache:
        """Private local evidence memory cache."""
        return self._cache

    @property
    def churn_estimator(self) -> ChurnEstimator:
        """Private online graph churn estimator."""
        return self._churn_estimator

    @property
    def mode_controller(self) -> AdaptiveModeController:
        """Private adaptive search mode controller."""
        return self._mode_controller

    @property
    def decision_engine(self) -> NextHopDecisionEngine:
        """Private next-hop candidate scoring and selection engine."""
        return self._decision_engine

    @property
    def visit_counts(self) -> Mapping[Union[int, str], int]:
        """Local visit frequency map."""
        return dict(self._visit_counts)

    @property
    def step_trace(self) -> Sequence[dict[str, Any]]:
        """Audit trace of step decisions."""
        return tuple(self._step_trace)

    @property
    def last_decision(self) -> Optional[NextHopDecision]:
        """Most recent structured NextHopDecision, if applicable."""
        return self._last_decision

    def update_evidence(self, observation: Observation) -> NodeEvidence:
        """Incorporate a direct partial observation snapshot into private evidence memory.

        Args:
            observation: Local observation perceived at the walker's current node.

        Returns:
            The created and stored NodeEvidence record.
        """
        detected = bool(observation.detected)
        sig_val = (
            float(observation.target_signal.signal_strength)
            if (observation.target_signal and observation.target_signal.signal_strength > 0.0)
            else (1.0 if detected else 0.0)
        )

        conf = sig_val if detected else 0.8

        return self._cache.store_from_observation(
            obs=observation,
            current_time=observation.time,
            target_found=False,
            confidence=conf,
            walker_id=self._walker_id,
            source=EvidenceSource.DIRECT_VISIT,
        )

    def update_churn(self, observation: Observation) -> float:
        """Update the online churn estimator with observed active neighborhood.

        Args:
            observation: Current local observation snapshot.

        Returns:
            Updated smoothed churn estimate lambda_hat.
        """
        self._churn_estimator.update(observation)
        current_lambda = max(0.001, self._churn_estimator.lambda_hat)
        self._cache.default_lambda_hat = current_lambda
        return current_lambda

    def exchange_information(
        self,
        observation: Observation,
        exchange: Optional[NodeMediatedExchange] = None,
    ) -> list[NodeEvidence]:
        """Publish local evidence and retrieve shared evidence at the current node.

        Args:
            observation: Local observation at current location.
            exchange: Optional shared NodeMediatedExchange substrate.

        Returns:
            List of newly pulled NodeEvidence records, if any.
        """
        if exchange is None:
            return []

        curr_node = observation.current_node
        t = observation.time

        # Push local evidence if permitted by mode
        if exchange.mode.can_push:
            exchange.push_from_cache(
                walker_id=self._walker_id,
                at_node=curr_node,
                local_cache=self._cache,
                timestamp=t,
            )

        # Pull shared evidence if permitted by mode
        pulled_records: list[NodeEvidence] = []
        if exchange.mode.can_pull:
            pulled_records = exchange.pull(
                walker_id=self._walker_id,
                at_node=curr_node,
                timestamp=t,
                target_cache=self._cache,
                exclude_self=True,
            )

        return pulled_records

    def update_mode(
        self,
        observation: Observation,
        pulled_evidence: Optional[Sequence[NodeEvidence]] = None,
        lambda_hat: float = 0.0,
    ) -> SearchMode:
        """Evaluate information gain and update search mode (LOCAL vs LONG_JUMP).

        Args:
            observation: Current step observation.
            pulled_evidence: Optional evidence retrieved via node exchange during this step.
            lambda_hat: Current graph volatility estimate.

        Returns:
            Updated SearchMode.
        """
        # Determine whether useful new information arrived
        is_useful = self._mode_controller.evaluate_useful_information(observation)
        if not is_useful and pulled_evidence:
            is_useful = self._mode_controller.evaluate_useful_information(pulled_evidence)

        return self._mode_controller.update(
            timestamp=observation.time,
            useful=is_useful,
            lambda_hat=lambda_hat,
        )

    def step(
        self,
        observation: Observation,
        exchange: Optional[NodeMediatedExchange] = None,
    ) -> SearchAction:
        """Execute a full decision step for this walker.

        Orchestrates:
        1. Local visit tracking
        2. Churn estimation update
        3. Local evidence memory storage
        4. Node-mediated communication exchange
        5. Mode controller update (stagnation detection)
        6. Candidate scoring and action selection

        Args:
            observation: Local observation perceived at current time step.
            exchange: Optional NodeMediatedExchange for inter-walker evidence sharing.

        Returns:
            Legal SearchAction to be executed by the simulation coordinator.
        """
        curr = observation.current_node
        self._visit_counts[curr] = self._visit_counts.get(curr, 0) + 1

        # 1. Update online churn estimator
        lambda_hat = self.update_churn(observation)

        # 2. Store direct observation in local evidence cache
        self.update_evidence(observation)

        # 3. Information exchange via node cache
        pulled = self.exchange_information(observation, exchange=exchange)

        # 4. Update adaptive mode controller
        mode = self.update_mode(observation, pulled_evidence=pulled, lambda_hat=lambda_hat)

        # 5. Decide next hop action
        if mode == SearchMode.LONG_JUMP:
            action = SearchAction(
                action_type=ActionType.JUMP,
                destination=None,
                metadata={
                    "mode": SearchMode.LONG_JUMP.value,
                    "walker_id": self._walker_id,
                    "reason": "stagnation_long_jump",
                    "lambda_hat": lambda_hat,
                },
            )
            self._last_decision = None
        else:
            decision = self._decision_engine.decide(
                observation=observation,
                evidence_cache=self._cache,
                current_time=observation.time,
                mode=mode,
                visit_counts=self._visit_counts,
                traversal_delays=self._traversal_delays,
                lambda_hat=lambda_hat,
                rng=self._rng,
            )
            self._last_decision = decision
            action = decision.action

        # Record step diagnostic trace
        self._step_trace.append({
            "time": observation.time,
            "walker_id": self._walker_id,
            "current_node": curr,
            "mode": mode.value,
            "lambda_hat": lambda_hat,
            "action_type": action.action_type.value,
            "destination": action.destination,
            "stagnation_steps": self._mode_controller.stagnation_steps,
        })

        return action

    def decide(self, observation: Observation) -> SearchAction:
        """Standalone decision interface without shared communication exchange."""
        return self.step(observation, exchange=None)

    def reset(self) -> None:
        """Reset private cognitive state while preserving configuration."""
        self._rng = np.random.default_rng(self._seed)
        self._cache.clear()
        self._churn_estimator.reset()
        self._mode_controller.reset()
        self._visit_counts.clear()
        self._traversal_delays.clear()
        self._step_trace.clear()
        self._last_decision = None


class AESWPolicy(BaselinePolicy):
    """Multi-walker AESW Search Policy conforming to the BaselinePolicy interface.

    Orchestrates k autonomous AESWAgent instances and one shared NodeMediatedExchange
    substrate, enabling evaluation within the experiment harness alongside baselines.
    """

    def __init__(
        self,
        k: int = 4,
        seed: Optional[int] = None,
        communication_mode: Union[str, CommunicationMode] = CommunicationMode.PUSH_PULL,
        node_capacity: Optional[int] = None,
        eta: float = DEFAULT_ETA,
        epsilon: float = DEFAULT_EPSILON,
        window_size: int = AdaptiveModeController.DEFAULT_WINDOW_SIZE,
        weight_positive: float = 1.0,
        weight_novelty: float = 0.5,
        weight_revisit: float = 0.5,
        weight_delay: float = 0.2,
        temperature: float = 1.0,
        mode_weight_scaling: bool = True,
        default_lambda_hat: float = 0.05,
        max_cache_capacity: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize multi-walker AESW search policy.

        Args:
            k: Number of walkers in the team (k >= 1).
            seed: Master random seed for walker PRNG stream isolation.
            communication_mode: Node-mediated communication mode (NO_SHARING, PUSH, PULL, PUSH_PULL).
            node_capacity: Maximum capacity of vertex-anchored shared caches.
            eta: Churn estimator exponential smoothing parameter.
            epsilon: Churn estimator numerical stability floor.
            window_size: Mode controller stagnation window W.
            weight_positive: Weight alpha for positive evidence.
            weight_novelty: Weight beta for vertex novelty.
            weight_revisit: Penalty gamma for vertex revisits.
            weight_delay: Penalty delta for edge traversal delay.
            temperature: Softmax exploration temperature T > 0.
            mode_weight_scaling: Whether to scale exploration weights under LONG_JUMP mode.
            default_lambda_hat: Baseline churn rate reference.
            max_cache_capacity: Capacity bound for each walker's local evidence cache.
            **kwargs: Additional parameters.
        """
        if k <= 0:
            raise ValueError(f"Number of walkers k must be strictly positive, got {k}")

        self._k = int(k)
        self._base_seed = seed
        self._comm_mode = validate_communication_mode(communication_mode)
        self._node_capacity = node_capacity

        # Shared node-mediated communication exchange
        self._exchange = NodeMediatedExchange(
            mode=self._comm_mode,
            node_capacity=node_capacity,
        )

        # Initialize k independent walker agents with isolated PRNG seeds
        self._agents: tuple[AESWAgent, ...] = tuple(
            AESWAgent(
                walker_id=i,
                seed=(seed + i * 1000) if seed is not None else None,
                eta=eta,
                epsilon=epsilon,
                window_size=window_size,
                weight_positive=weight_positive,
                weight_novelty=weight_novelty,
                weight_revisit=weight_revisit,
                weight_delay=weight_delay,
                temperature=temperature,
                mode_weight_scaling=mode_weight_scaling,
                default_lambda_hat=default_lambda_hat,
                max_cache_capacity=max_cache_capacity,
            )
            for i in range(self._k)
        )

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.AESW

    @property
    def seed(self) -> Optional[int]:
        return self._base_seed

    @property
    def k(self) -> int:
        """Number of searchers in the team."""
        return self._k

    @property
    def communication_mode(self) -> CommunicationMode:
        """Configured communication mode."""
        return self._comm_mode

    @property
    def exchange(self) -> NodeMediatedExchange:
        """Shared node-mediated communication exchange."""
        return self._exchange

    @property
    def total_messages(self) -> int:
        """Total communication messages Q incurred across all walkers."""
        return self._exchange.total_messages

    @property
    def agents(self) -> tuple[AESWAgent, ...]:
        """Underlying autonomous walker agents."""
        return self._agents

    @property
    def metadata(self) -> BaselineMetadata:
        return BaselineMetadata(
            name=f"AESW (k={self._k}, {self._comm_mode.value})",
            baseline_type=BaselineType.AESW,
            description="Adaptive Evidence-Sharing Walkers with adaptive churn decay and node-mediated sharing.",
            uses_memory=True,
            uses_detection_signals=True,
            uses_communication=(self._comm_mode != CommunicationMode.NO_SHARING),
            is_multi_walker=True,
            parameters={
                "k": self._k,
                "communication_mode": self._comm_mode.value,
            },
        )

    def decide(self, observation: Observation) -> SearchAction:
        """Decision delegate for single-walker execution using agent 0."""
        return self._agents[0].step(observation, exchange=self._exchange)

    def decide_walker(self, walker_index: int, observation: Observation) -> SearchAction:
        """Decide next action for a specific walker in the team.

        Args:
            walker_index: Index in [0, k-1].
            observation: Partial observation perceived by walker_index.

        Returns:
            Validated SearchAction chosen by that walker.
        """
        if not (0 <= walker_index < self._k):
            raise IndexError(f"walker_index {walker_index} out of bounds for k={self._k}")
        return self._agents[walker_index].step(observation, exchange=self._exchange)

    def decide_all(self, observations: Mapping[int, Observation]) -> dict[int, SearchAction]:
        """Decide actions for all walkers in deterministic index order.

        Args:
            observations: Mapping from walker ID to Observation snapshot.

        Returns:
            Dictionary mapping walker ID to its selected SearchAction.
        """
        actions: dict[int, SearchAction] = {}
        for w_id in sorted(observations.keys()):
            if 0 <= w_id < self._k:
                actions[w_id] = self.decide_walker(w_id, observations[w_id])
        return actions

    def get_traces(self) -> dict[int, list[dict[str, Any]]]:
        """Retrieve decision audit traces for all walkers."""
        return {agent.walker_id: list(agent.step_trace) for agent in self._agents}

    def reset_walker(self, walker_index: int) -> None:
        """Reset state for a specific walker."""
        if not (0 <= walker_index < self._k):
            raise IndexError(f"walker_index {walker_index} out of bounds for k={self._k}")
        self._agents[walker_index].reset()

    def reset(self) -> None:
        """Reset all walkers and the shared communication exchange."""
        self._exchange.reset()
        for agent in self._agents:
            agent.reset()
