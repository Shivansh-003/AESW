"""
Target Locomotion and State Engine
==================================
Manages ground-truth state and Markovian locomotion of the target x_t ∈ V
across currently active edges in G_t = (V, E_t).

CRITICAL INVARIANT:
The target state produced and managed here is unobserved ground-truth state.
It must NEVER be directly exposed to search walkers or observation objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence
import numpy as np

from aesw.environment.types import TargetMode
from aesw.environment.models import TargetState
from aesw.environment.problem import TargetSpecification
from aesw.utils.reproducibility import create_rng

if TYPE_CHECKING:
    from aesw.dynamics.view import ActiveGraphView


class TargetEngine:
    """Manages ground-truth target locality and stochastic locomotion over G_t.

    Attributes:
        spec: TargetSpecification containing locomotion mode, p_move, and optional initial_node.
        seed: Random seed for the isolated target movement RNG stream.
    """

    def __init__(
        self,
        spec: TargetSpecification,
        initial_node: Optional[int | str] = None,
        candidate_nodes: Optional[Sequence[int | str]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        if not isinstance(spec, TargetSpecification):
            raise TypeError(f"spec must be a TargetSpecification instance, got {type(spec).__name__}")

        self._spec: TargetSpecification = spec
        self._seed: int = 42 if seed is None else int(seed)
        self._rng: np.random.Generator = create_rng(self._seed)

        # Resolve initial node
        resolved_initial: Optional[int | str] = initial_node if initial_node is not None else spec.initial_node
        if resolved_initial is None:
            if not candidate_nodes:
                raise ValueError("Must provide either initial_node, spec.initial_node, or non-empty candidate_nodes")
            # Deterministic uniform selection from available nodes using target RNG stream
            idx = int(self._rng.integers(0, len(candidate_nodes)))
            resolved_initial = candidate_nodes[idx]

        self._state: TargetState = TargetState(
            current_node=resolved_initial,
            mode=spec.mode,
            p_move=spec.p_move,
            previous_node=None,
            time=0,
            move_attempted=False,
            move_succeeded=False,
        )
        self._trajectory: list[int | str] = [resolved_initial]

    @property
    def state(self) -> TargetState:
        """Current ground-truth TargetState (unobserved)."""
        return self._state

    @property
    def current_node(self) -> int | str:
        """Current vertex occupied by the target x_t."""
        return self._state.current_node

    @property
    def time(self) -> int:
        """Current timestamp of the target engine."""
        return self._state.time

    @property
    def trajectory(self) -> list[int | str]:
        """Sequence of nodes occupied by the target up to step t."""
        return list(self._trajectory)

    def step(self, active_graph: ActiveGraphView) -> TargetState:
        """Execute one simulation step of target locomotion over G_t.

        Locomotion Rules:
        1. If mode == STATIC or p_move == 0:
           Target remains stationary at x_t.
        2. With probability p_move:
           Target attempts movement.
           - Obtains active neighbors N_t(x_t) = {v : (x_t, v) in E_t}.
           - If active neighbors exist:
             Uniformly selects one neighbor v in N_t(x_t).
             Transition succeeds: x_(t+1) = v.
           - If no active neighbors exist (target is trapped/isolated by dynamic edge churn):
             Target remains stationary: x_(t+1) = x_t.
        3. With probability 1 - p_move:
           Target remains stationary: x_(t+1) = x_t.

        Args:
            active_graph: ActiveGraphView presenting operational edges at time t.

        Returns:
            Updated TargetState at step t+1.
        """
        curr = self._state.current_node
        next_time = self._state.time + 1

        if self._spec.mode == TargetMode.STATIC or self._spec.p_move == 0.0:
            self._state = TargetState(
                current_node=curr,
                mode=self._spec.mode,
                p_move=self._spec.p_move,
                previous_node=curr,
                time=next_time,
                move_attempted=False,
                move_succeeded=False,
            )
            self._trajectory.append(curr)
            return self._state

        # Sample movement attempt using target isolated RNG stream
        attempt_roll = float(self._rng.random())
        attempt_movement = attempt_roll < self._spec.p_move

        if not attempt_movement:
            self._state = TargetState(
                current_node=curr,
                mode=self._spec.mode,
                p_move=self._spec.p_move,
                previous_node=curr,
                time=next_time,
                move_attempted=False,
                move_succeeded=False,
            )
            self._trajectory.append(curr)
            return self._state

        # Movement attempted: query active neighbors in G_t
        active_nbrs = active_graph.active_neighbors(curr)
        if not active_nbrs:
            # Trapped: no active incident links
            self._state = TargetState(
                current_node=curr,
                mode=self._spec.mode,
                p_move=self._spec.p_move,
                previous_node=curr,
                time=next_time,
                move_attempted=True,
                move_succeeded=False,
            )
            self._trajectory.append(curr)
            return self._state

        # Uniform selection among currently active neighbors
        nbr_idx = int(self._rng.integers(0, len(active_nbrs)))
        chosen_node = active_nbrs[nbr_idx]

        self._state = TargetState(
            current_node=chosen_node,
            mode=self._spec.mode,
            p_move=self._spec.p_move,
            previous_node=curr,
            time=next_time,
            move_attempted=True,
            move_succeeded=True,
        )
        self._trajectory.append(chosen_node)
        return self._state
