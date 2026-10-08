"""
Flooding Search Policy (Baseline 5)
===================================
Aggressive frontier-expanding search policy under partial observability.
Maintains a local discovery frontier and prioritizes moving toward unexplored,
newly discovered local neighbors before revisiting previously explored nodes.
"""

from typing import Optional, Set
import numpy as np

from aesw.utils.reproducibility import create_rng
from aesw.baselines.types import BaselineType, ActionType
from aesw.baselines.actions import SearchAction
from aesw.baselines.base import BaselinePolicy, BaselineMetadata
from aesw.environment.observation import Observation


class FloodingPolicy(BaselinePolicy):
    """Local frontier propagation / flooding search policy.

    Information Boundary:
    The walker has NO access to global graph topology. Nodes and edges are discovered
    strictly as they enter the walker's local observation window.

    Decision Rule:
    1. Register the current node as visited: V_visited <- V_visited U {current_node}.
    2. Register all currently observed active neighbors as discovered:
       V_disc <- V_disc U N_obs(u).
    3. Partition available candidates into:
       - Unvisited candidates: N_unvisited = N_obs(u) \\ V_visited.
       - Visited candidates: N_visited = N_obs(u) ∩ V_visited.
    4. If N_unvisited is non-empty:
       Uniformly sample next hop from N_unvisited (aggressive frontier expansion).
    5. If all available neighbors are already visited:
       Uniformly sample next hop from N_obs(u) (continuation / fallback traversal).
    6. If N_obs(u) is empty:
       STAY at current node.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng: np.random.Generator = create_rng(seed)
        self._visited_nodes: Set[int | str] = set()
        self._discovered_nodes: Set[int | str] = set()

    @property
    def baseline_type(self) -> BaselineType:
        return BaselineType.FLOODING

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def visited_nodes(self) -> set[int | str]:
        """Read-only copy of physically visited nodes."""
        return set(self._visited_nodes)

    @property
    def discovered_nodes(self) -> set[int | str]:
        """Read-only copy of locally discovered nodes."""
        return set(self._discovered_nodes)

    @property
    def metadata(self) -> BaselineMetadata:
        return BaselineMetadata(
            name="Flooding / Frontier Search",
            baseline_type=BaselineType.FLOODING,
            description="Aggressive frontier exploration prioritizing unvisited locally discovered neighbors.",
            uses_memory=True,
            uses_detection_signals=False,
            uses_communication=False,
            is_multi_walker=False,
        )

    def decide(self, observation: Observation) -> SearchAction:
        """Select next hop prioritizing unvisited observed neighbors."""
        current_node = observation.current_node
        self._visited_nodes.add(current_node)

        candidates = self.get_available_candidates(observation)

        # Update discovered nodes
        for nbr in candidates:
            self._discovered_nodes.add(nbr)

        if not candidates:
            action = SearchAction(
                action_type=ActionType.STAY,
                destination=current_node,
                metadata={"reason": "no_available_neighbors"},
            )
            return self._validate_action(action, observation)

        # Partition into unvisited vs visited candidates
        unvisited = [nbr for nbr in candidates if nbr not in self._visited_nodes]

        if unvisited:
            eligible = unvisited
            prioritized_unvisited = True
        else:
            eligible = candidates
            prioritized_unvisited = False

        idx = int(self._rng.integers(0, len(eligible)))
        chosen_destination = eligible[idx]

        action = SearchAction(
            action_type=ActionType.MOVE,
            destination=chosen_destination,
            metadata={
                "candidate_count": len(candidates),
                "unvisited_count": len(unvisited),
                "prioritized_unvisited": prioritized_unvisited,
                "total_visited": len(self._visited_nodes),
                "total_discovered": len(self._discovered_nodes),
            },
        )
        return self._validate_action(action, observation)

    def reset(self) -> None:
        """Clear visited and discovered sets and re-seed PRNG."""
        self._visited_nodes.clear()
        self._discovered_nodes.clear()
        self._rng = create_rng(self._seed)
