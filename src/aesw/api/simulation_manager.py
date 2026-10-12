"""
Simulation Session and Lifecycle Manager
========================================
Manages discrete-time simulation sessions, coordinating the underlying
ExperimentInstance, SimulationCoordinator, and search policies (AESW and baselines)
with deterministic layouts, thread-safe stepping, and audit traces.
"""

from __future__ import annotations

import math
import threading
import uuid
from typing import Any, Dict, List, Mapping, Optional, Union
import networkx as nx
import numpy as np

from aesw.api.models import (
    CandidateScoreDto,
    ChurnItemDto,
    DecisionTraceDto,
    EvidenceItemDto,
    GraphEdgeDto,
    GraphNodeDto,
    GraphResponse,
    MetricsDto,
    SimulationConfig,
    SimulationStatusResponse,
    StepResultResponse,
    TrajectoryDto,
    WalkerStateDto,
)
from aesw.aesw.agent import AESWPolicy
from aesw.baselines.base import BaselinePolicy
from aesw.baselines.factory import create_baseline
from aesw.baselines.independent_walkers import IndependentRandomWalkers
from aesw.baselines.types import BaselineType
from aesw.environment.coordinator import SimulationCoordinator
from aesw.environment.types import EdgeState, SearchTerminationStatus
from aesw.evaluation.experiment import (
    ExperimentInstance,
    _resolve_baseline_type,
    generate_experiment,
)


class SimulationSession:
    """Encapsulates a live, interactive simulation session."""

    def __init__(self, simulation_id: str, config: SimulationConfig) -> None:
        self.simulation_id = simulation_id
        self.config = config
        self._lock = threading.RLock()

        t_mode_str = config.target_mode.upper()
        if t_mode_str in ("RANDOM_WALK", "MOVE", "MOVING"):
            t_mode = "MOVING"
        elif t_mode_str in ("STATIC", "STATIONARY"):
            t_mode = "STATIC"
        else:
            t_mode = config.target_mode

        overrides: Dict[str, Any] = {
            "graph_type": config.graph_type,
            "num_nodes": config.num_nodes,
            "regime": config.regime,
            "p_move": config.p_move,
            "target_mode": t_mode,
            "signal_radius": config.signal_radius,
            "p_d": config.p_d,
            "p_fa": config.p_fa,
            "neighbor_budget": config.neighbor_budget,
            "walker_count": config.walker_count,
            "time_horizon": config.time_horizon,
            "c_m": config.c_m,
        }
        if config.p_on is not None:
            overrides["p_on"] = config.p_on
        if config.p_off is not None:
            overrides["p_off"] = config.p_off
        if config.graph_parameters:
            overrides.update(config.graph_parameters)

        self.experiment: ExperimentInstance = generate_experiment(
            seed=config.seed,
            **overrides,
        )

        # Precompute stable deterministic 2D layout coordinates for nodes
        self._node_positions = self._compute_stable_layout()

        # Track history
        self._churn_history: Dict[int, List[Dict[str, Any]]] = {
            w_id: [] for w_id in range(self.config.walker_count)
        }
        self._latest_decisions: Dict[int, DecisionTraceDto] = {}
        self._previous_edge_states: Dict[tuple[Any, Any], EdgeState] = {}

        # Initialize coordinator and policies
        self.coordinator: SimulationCoordinator = self.experiment.create_coordinator()
        self.is_aesw = False
        self.policy: Optional[Union[AESWPolicy, IndependentRandomWalkers, Dict[int, BaselinePolicy]]] = None
        self._processed_comm_event_index = 0
        self._init_policy()

    def _sync_communication(self) -> None:
        """Synchronize node-mediated communication events to coordinator per-walker counters."""
        if not (self.is_aesw and isinstance(self.policy, AESWPolicy)):
            return
        events = self.policy.exchange.metrics.events
        if len(events) > self._processed_comm_event_index:
            new_events = events[self._processed_comm_event_index:]
            for ev in new_events:
                self.coordinator.record_messages(count=ev.message_cost, walker_id=ev.walker_id)
            self._processed_comm_event_index = len(events)

    def _compute_stable_layout(self) -> Dict[Union[int, str], tuple[float, float]]:
        """Compute stable 2D node coordinates in range [-1.0, 1.0]."""
        graph_inst = self.experiment.static_graph
        nx_graph = graph_inst._nx_graph
        pos_dict: Dict[Union[int, str], tuple[float, float]] = {}

        # 1. Check if nodes already have pos or coordinates (e.g. RGG or Grid)
        has_pos = all("pos" in data for _, data in nx_graph.nodes(data=True))
        if has_pos:
            raw_pos = {n: data["pos"] for n, data in nx_graph.nodes(data=True)}
            # Normalize to [-1.0, 1.0]
            xs = [p[0] for p in raw_pos.values()]
            ys = [p[1] for p in raw_pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = (max_x - min_x) if max_x > min_x else 1.0
            span_y = (max_y - min_y) if max_y > min_y else 1.0
            for n, (x, y) in raw_pos.items():
                norm_x = ((x - min_x) / span_x) * 1.8 - 0.9
                norm_y = ((y - min_y) / span_y) * 1.8 - 0.9
                pos_dict[n] = (float(norm_x), float(norm_y))
            return pos_dict

        # 2. Check 2D grid coordinates (x, y)
        has_xy = all("x" in data and "y" in data for _, data in nx_graph.nodes(data=True))
        if has_xy:
            raw_pos = {n: (data["x"], data["y"]) for n, data in nx_graph.nodes(data=True)}
            xs = [p[0] for p in raw_pos.values()]
            ys = [p[1] for p in raw_pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = (max_x - min_x) if max_x > min_x else 1.0
            span_y = (max_y - min_y) if max_y > min_y else 1.0
            for n, (x, y) in raw_pos.items():
                norm_x = ((x - min_x) / span_x) * 1.8 - 0.9
                norm_y = ((y - min_y) / span_y) * 1.8 - 0.9
                pos_dict[n] = (float(norm_x), float(norm_y))
            return pos_dict

        # 3. Compute deterministic spring layout
        try:
            spring_pos = nx.spring_layout(
                nx_graph,
                seed=self.config.seed,
                iterations=50,
                k=1.2 / math.sqrt(max(1, len(nx_graph.nodes))),
            )
            for n, arr in spring_pos.items():
                pos_dict[n] = (float(arr[0]), float(arr[1]))
        except Exception:
            # Fallback circle layout
            nodes = list(nx_graph.nodes())
            total = len(nodes)
            for idx, n in enumerate(nodes):
                angle = 2 * math.pi * idx / max(1, total)
                pos_dict[n] = (0.85 * math.cos(angle), 0.85 * math.sin(angle))

        return pos_dict

    def _init_policy(self) -> None:
        """Instantiate policy according to selected algorithm."""
        num_walkers = self.config.walker_count
        algo_seed = self.experiment.sub_seeds["algorithm"]
        algo_type = _resolve_baseline_type(self.config.algorithm)

        if algo_type == BaselineType.AESW:
            self.is_aesw = True
            self.policy = AESWPolicy(
                k=num_walkers,
                seed=algo_seed,
                communication_mode=self.config.communication_mode,
            )
        elif algo_type == BaselineType.INDEPENDENT_RANDOM_WALKERS:
            self.is_aesw = False
            self.policy = IndependentRandomWalkers(k=num_walkers, base_seed=algo_seed)
        else:
            self.is_aesw = False
            policies: Dict[int, BaselinePolicy] = {}
            for w_id in range(num_walkers):
                w_seed = algo_seed + w_id * 1000
                policies[w_id] = create_baseline(algo_type, seed=w_seed)
            self.policy = policies

    def step(self) -> StepResultResponse:
        """Advance the simulation by exactly one discrete step."""
        with self._lock:
            if self.coordinator.is_terminated:
                return StepResultResponse(
                    simulation_id=self.simulation_id,
                    time=self.coordinator.time,
                    is_terminated=True,
                    termination_status=self.coordinator.termination_status.value if self.coordinator.termination_status else None,
                    metrics=self.get_metrics(),
                    walkers=self.get_walkers(),
                    latest_decisions=list(self._latest_decisions.values()),
                )

            # Record current active edge states to detect transitions
            ground_truth = self.coordinator.get_ground_truth_state()
            self._previous_edge_states = dict(ground_truth.active_edge_states)

            observations = self.coordinator.get_observations()
            actions: Dict[int, Any] = {}

            if self.is_aesw and isinstance(self.policy, AESWPolicy):
                actions = self.policy.decide_all(observations)
                # Capture decision traces and churn history
                t = self.coordinator.time
                for w_id, agent in enumerate(self.policy.agents):
                    last_dec = agent.last_decision
                    est_lambda = agent.churn_estimator.estimated_lambda
                    stagnation = agent.mode_controller.stagnation_steps
                    mode = agent.mode_controller.mode.value

                    c_scores: List[CandidateScoreDto] = []
                    sel_node = None
                    if last_dec is not None:
                        sel_node = last_dec.selected_node
                        for c_id, score in last_dec.scores.items():
                            c_scores.append(
                                CandidateScoreDto(
                                    node_id=c_id,
                                    positive_evidence=float(score.positive_evidence),
                                    novelty=float(score.novelty),
                                    revisit_penalty=float(score.revisit_penalty),
                                    delay=float(score.delay),
                                    total_score=float(score.total_score),
                                    probability=float(last_dec.probabilities.get(c_id, 0.0)),
                                )
                            )

                    act = actions.get(w_id)
                    trace = DecisionTraceDto(
                        walker_id=w_id,
                        time=t,
                        current_node=observations[w_id].current_node,
                        mode=mode,
                        estimated_lambda=float(est_lambda),
                        stagnation_steps=int(stagnation),
                        selected_node=sel_node,
                        action_type=act.action_type.value if act else "UNKNOWN",
                        destination=act.destination if act else None,
                        candidate_scores=c_scores,
                    )
                    self._latest_decisions[w_id] = trace

                    # Record churn history point
                    ov = agent.churn_estimator.overlap
                    raw_c = agent.churn_estimator.raw_churn_rate
                    self._churn_history[w_id].append({
                        "time": t,
                        "lambda_hat": float(est_lambda),
                        "overlap": float(ov) if ov is not None else 1.0,
                        "raw_churn": float(raw_c) if raw_c is not None else 0.0,
                    })

                # Sync communication messages
                self._sync_communication()

            elif isinstance(self.policy, IndependentRandomWalkers):
                actions = self.policy.decide_all(observations)
                t = self.coordinator.time
                for w_id, obs in observations.items():
                    act = actions.get(w_id)
                    candidates = self.policy.get_available_candidates(obs)
                    prob = 1.0 / len(candidates) if candidates else 0.0
                    c_scores = [
                        CandidateScoreDto(
                            node_id=c_id,
                            positive_evidence=0.0,
                            novelty=0.0,
                            revisit_penalty=0.0,
                            delay=1.0,
                            total_score=1.0,
                            probability=float(prob),
                        )
                        for c_id in candidates
                    ]
                    self._latest_decisions[w_id] = DecisionTraceDto(
                        walker_id=w_id,
                        time=t,
                        current_node=obs.current_node,
                        mode="BASELINE",
                        estimated_lambda=0.0,
                        stagnation_steps=0,
                        selected_node=act.destination if act and act.action_type.value == "move" else None,
                        action_type=act.action_type.value if act else "UNKNOWN",
                        destination=act.destination if act else None,
                        candidate_scores=c_scores,
                    )

            elif isinstance(self.policy, dict):
                t = self.coordinator.time
                for w_id, obs in observations.items():
                    if w_id in self.policy:
                        pol = self.policy[w_id]
                        act = pol.decide(obs)
                        actions[w_id] = act

                        candidates = pol.get_available_candidates(obs) if hasattr(pol, "get_available_candidates") else []
                        c_scores: List[CandidateScoreDto] = []

                        # Extract policy-specific candidate metrics
                        if pol.baseline_type.value == "random_walk":
                            prob = 1.0 / len(candidates) if candidates else 0.0
                            for c_id in candidates:
                                c_scores.append(
                                    CandidateScoreDto(
                                        node_id=c_id,
                                        positive_evidence=0.0,
                                        novelty=0.0,
                                        revisit_penalty=0.0,
                                        delay=1.0,
                                        total_score=1.0,
                                        probability=float(prob),
                                    )
                                )
                        elif pol.baseline_type.value == "non_backtracking":
                            prev_n = getattr(pol, "_previous_node", None)
                            nb_candidates = [n for n in candidates if n != prev_n]
                            eligible = nb_candidates if nb_candidates else candidates
                            prob = 1.0 / len(eligible) if eligible else 0.0
                            for c_id in candidates:
                                is_eligible = c_id in eligible
                                c_scores.append(
                                    CandidateScoreDto(
                                        node_id=c_id,
                                        positive_evidence=0.0,
                                        novelty=1.0 if c_id != prev_n else 0.0,
                                        revisit_penalty=1.0 if c_id == prev_n else 0.0,
                                        delay=1.0,
                                        total_score=1.0 if is_eligible else 0.0,
                                        probability=float(prob if is_eligible else 0.0),
                                    )
                                )
                        elif pol.baseline_type.value == "degree_based":
                            scores_map = act.metadata.get("candidate_scores", {}) if act and act.metadata else {}
                            total_s = sum(scores_map.values()) if scores_map else 0.0
                            for c_id in candidates:
                                deg_score = float(scores_map.get(c_id, getattr(pol, "_default_prior", 1.0)))
                                prob = deg_score / total_s if total_s > 0 else (1.0 / len(candidates) if candidates else 0.0)
                                c_scores.append(
                                    CandidateScoreDto(
                                        node_id=c_id,
                                        positive_evidence=0.0,
                                        novelty=deg_score,
                                        revisit_penalty=0.0,
                                        delay=1.0,
                                        total_score=deg_score,
                                        probability=float(prob),
                                    )
                                )
                        elif pol.baseline_type.value == "ant_colony":
                            prob_map = act.metadata.get("candidate_probabilities", {}) if act and act.metadata else {}
                            for c_id in candidates:
                                p_val = float(prob_map.get(c_id, 1.0 / len(candidates) if candidates else 0.0))
                                tau = float(pol.get_pheromone(obs.current_node, c_id)) if hasattr(pol, "get_pheromone") else 1.0
                                c_scores.append(
                                    CandidateScoreDto(
                                        node_id=c_id,
                                        positive_evidence=tau,
                                        novelty=0.0,
                                        revisit_penalty=0.0,
                                        delay=1.0,
                                        total_score=tau,
                                        probability=p_val,
                                    )
                                )
                        elif pol.baseline_type.value == "flooding":
                            visited_set = getattr(pol, "_visited_nodes", set())
                            unvis = [n for n in candidates if n not in visited_set]
                            eligible = unvis if unvis else candidates
                            prob = 1.0 / len(eligible) if eligible else 0.0
                            for c_id in candidates:
                                is_unvis = c_id not in visited_set
                                is_el = c_id in eligible
                                c_scores.append(
                                    CandidateScoreDto(
                                        node_id=c_id,
                                        positive_evidence=0.0,
                                        novelty=1.0 if is_unvis else 0.0,
                                        revisit_penalty=1.0 if not is_unvis else 0.0,
                                        delay=1.0,
                                        total_score=1.0 if is_el else 0.0,
                                        probability=float(prob if is_el else 0.0),
                                    )
                                )
                        else:
                            prob = 1.0 / len(candidates) if candidates else 0.0
                            for c_id in candidates:
                                c_scores.append(
                                    CandidateScoreDto(
                                        node_id=c_id,
                                        positive_evidence=0.0,
                                        novelty=0.0,
                                        revisit_penalty=0.0,
                                        delay=1.0,
                                        total_score=1.0,
                                        probability=float(prob),
                                    )
                                )

                        self._latest_decisions[w_id] = DecisionTraceDto(
                            walker_id=w_id,
                            time=t,
                            current_node=obs.current_node,
                            mode="BASELINE",
                            estimated_lambda=0.0,
                            stagnation_steps=0,
                            selected_node=act.destination if act and act.action_type.value == "move" else None,
                            action_type=act.action_type.value if act else "UNKNOWN",
                            destination=act.destination if act else None,
                            candidate_scores=c_scores,
                        )

            # Step environment coordinator
            self.coordinator.step(actions)

            # Sync communication messages again if any
            if self.is_aesw and isinstance(self.policy, AESWPolicy):
                self._sync_communication()

            is_term = self.coordinator.is_terminated
            term_status = (
                self.coordinator.termination_status.value
                if self.coordinator.termination_status
                else None
            )

            return StepResultResponse(
                simulation_id=self.simulation_id,
                time=self.coordinator.time,
                is_terminated=is_term,
                termination_status=term_status,
                metrics=self.get_metrics(),
                walkers=self.get_walkers(),
                latest_decisions=list(self._latest_decisions.values()),
            )

    def reset(self) -> SimulationStatusResponse:
        """Reset the simulation to its initial deterministic realization."""
        with self._lock:
            self.coordinator = self.experiment.create_coordinator()
            self._churn_history = {
                w_id: [] for w_id in range(self.config.walker_count)
            }
            self._latest_decisions.clear()
            self._previous_edge_states.clear()
            self._processed_comm_event_index = 0
            self._init_policy()

            return self.get_status()

    def get_status(self) -> SimulationStatusResponse:
        """Get simulation status summary."""
        term_status = (
            self.coordinator.termination_status.value
            if self.coordinator.termination_status
            else None
        )
        status_str = (
            "TERMINATED"
            if self.coordinator.is_terminated
            else ("INITIALIZED" if self.coordinator.time == 0 else "RUNNING")
        )

        return SimulationStatusResponse(
            simulation_id=self.simulation_id,
            status=status_str,
            termination_status=term_status,
            time=self.coordinator.time,
            time_horizon=self.config.time_horizon,
            algorithm=self.config.algorithm,
            is_aesw=self.is_aesw,
            seed=self.config.seed,
            walker_count=self.config.walker_count,
            config=self.config,
            metrics=self.get_metrics(),
        )

    def get_metrics(self) -> MetricsDto:
        """Get formal search evaluation metrics."""
        run_metrics = self.coordinator.get_run_metrics(
            algorithm_name=self.config.algorithm,
            seed=self.config.seed,
        )
        if self.coordinator.is_terminated:
            status_val = run_metrics.status.value
        else:
            status_val = "INITIALIZED" if self.coordinator.time == 0 else "RUNNING"

        return MetricsDto(
            algorithm=run_metrics.algorithm,
            search_time=run_metrics.search_time,
            status=status_val,
            success=run_metrics.success,
            total_moves=run_metrics.total_moves,
            node_revisits=run_metrics.node_revisits,
            nodes_visited=run_metrics.nodes_visited,
            total_messages=run_metrics.total_messages,
            total_cost=float(run_metrics.total_cost),
            c_m=float(self.config.c_m),
        )

    def get_graph(self) -> GraphResponse:
        """Serialize current graph topology with active/inactive edges and walker locations."""
        with self._lock:
            ground_truth = self.coordinator.get_ground_truth_state()
            target_node = self.coordinator.target_state.current_node
            visited = self.coordinator.visited_nodes

            # Gather walker locations
            walker_locs: Dict[Union[int, str], List[int]] = {}
            for w_id, w_state in self.coordinator.walker_states.items():
                walker_locs.setdefault(w_state.current_node, []).append(w_id)

            # Gather evidence on nodes if AESW
            node_evidence_summary: Dict[Union[int, str], tuple[str, float]] = {}
            if self.is_aesw and isinstance(self.policy, AESWPolicy):
                curr_t = self.coordinator.time
                for agent in self.policy.agents:
                    all_ev = agent._cache.all_entries(current_time=curr_t)
                    for n_id, w_ev in all_ev.items():
                        pol = w_ev.evidence.polarity.value
                        eff_w = w_ev.weight
                        node_evidence_summary[n_id] = (pol, float(eff_w))

            # Build node list
            nodes_dto: List[GraphNodeDto] = []
            for n_id, n_obj in ground_truth.nodes.items():
                deg = self.experiment.static_graph._nx_graph.degree(n_id)
                pos = self._node_positions.get(n_id, (0.0, 0.0))
                ev_info = node_evidence_summary.get(n_id)

                nodes_dto.append(
                    GraphNodeDto(
                        id=n_id,
                        label=str(n_id),
                        degree=int(deg),
                        is_target=(n_id == target_node),
                        is_visited=(n_id in visited),
                        walker_ids=walker_locs.get(n_id, []),
                        has_evidence=(ev_info is not None),
                        evidence_polarity=ev_info[0] if ev_info else None,
                        evidence_weight=ev_info[1] if ev_info else None,
                        x=float(pos[0]),
                        y=float(pos[1]),
                    )
                )

            # Build edge list
            edges_dto: List[GraphEdgeDto] = []
            active_count = 0
            for edge in ground_truth.edges:
                is_on = (edge.state == EdgeState.ON)
                if is_on:
                    active_count += 1

                # Check if state changed from previous step
                prev_state = self._previous_edge_states.get(
                    (edge.source, edge.target),
                    self._previous_edge_states.get((edge.target, edge.source), None),
                )
                recent_transition = (prev_state is not None and prev_state != edge.state)

                edges_dto.append(
                    GraphEdgeDto(
                        source=edge.source,
                        target=edge.target,
                        is_active=is_on,
                        state=edge.state.value,
                        is_recent_transition=recent_transition,
                    )
                )

            return GraphResponse(
                simulation_id=self.simulation_id,
                nodes=nodes_dto,
                edges=edges_dto,
                target_node=target_node,
                visited_nodes=list(visited),
                active_edge_count=active_count,
                total_edge_count=len(edges_dto),
            )

    def get_walkers(self) -> List[WalkerStateDto]:
        """Serialize current walker states with cognitive properties if AESW."""
        with self._lock:
            states_list: List[WalkerStateDto] = []
            for w_id, w_state in self.coordinator.walker_states.items():
                mode = None
                est_lambda = None
                stagnation = None
                latest_dec = self._latest_decisions.get(w_id)
                recent_act = None
                recent_dest = None

                if latest_dec is not None:
                    recent_act = latest_dec.action_type
                    recent_dest = latest_dec.destination

                visit_counts_str: Dict[str, int] = {}
                if self.is_aesw and isinstance(self.policy, AESWPolicy) and w_id < len(self.policy.agents):
                    agent = self.policy.agents[w_id]
                    mode = agent.mode_controller.mode.value
                    est_lambda = float(agent.churn_estimator.estimated_lambda)
                    stagnation = int(agent.mode_controller.stagnation_steps)
                    visit_counts_str = {str(k): v for k, v in agent.visit_counts.items()}
                else:
                    mode = "BASELINE"
                    traj = self.coordinator.walker_trajectories.get(w_id, ())
                    counts: Dict[str, int] = {}
                    for node in traj:
                        node_str = str(node)
                        counts[node_str] = counts.get(node_str, 0) + 1
                    visit_counts_str = counts

                states_list.append(
                    WalkerStateDto(
                        walker_id=w_id,
                        current_node=w_state.current_node,
                        step_count=w_state.step_count,
                        message_count=w_state.message_count,
                        mode=mode,
                        estimated_lambda=est_lambda,
                        stagnation_steps=stagnation,
                        recent_action=recent_act,
                        recent_destination=recent_dest,
                        visit_counts=visit_counts_str,
                        latest_decision=latest_dec,
                    )
                )
            return states_list

    def get_evidence(self) -> List[EvidenceItemDto]:
        """Retrieve stored evidence records from all AESW agents."""
        if not self.is_aesw or not isinstance(self.policy, AESWPolicy):
            return []

        curr_t = self.coordinator.time
        items: List[EvidenceItemDto] = []

        with self._lock:
            for w_id, agent in enumerate(self.policy.agents):
                all_weighted = agent._cache.all_entries(current_time=curr_t)
                for n_id, w_ev in all_weighted.items():
                    ev = w_ev.evidence
                    items.append(
                        EvidenceItemDto(
                            node_id=n_id,
                            walker_id=w_id,
                            polarity=ev.polarity.value,
                            confidence=float(ev.confidence),
                            effective_weight=float(w_ev.weight),
                            effective_confidence=float(w_ev.effective_confidence),
                            timestamp=int(ev.timestamp),
                            age=int(w_ev.age),
                            source=ev.source.value,
                        )
                    )
        return items

    def get_churn(self) -> List[ChurnItemDto]:
        """Retrieve churn estimator status and history per walker."""
        if not self.is_aesw or not isinstance(self.policy, AESWPolicy):
            return []

        items: List[ChurnItemDto] = []
        with self._lock:
            for w_id, agent in enumerate(self.policy.agents):
                estimator = agent.churn_estimator
                ov = estimator.overlap
                raw_c = estimator.raw_churn_rate
                items.append(
                    ChurnItemDto(
                        walker_id=w_id,
                        estimated_lambda=float(estimator.estimated_lambda),
                        overlap=float(ov) if ov is not None else 1.0,
                        raw_churn_rate=float(raw_c) if raw_c is not None else 0.0,
                        update_count=int(estimator.update_count),
                        history=list(self._churn_history.get(w_id, [])),
                    )
                )
        return items

    def get_decisions(self) -> List[DecisionTraceDto]:
        """Get latest decision traces across all walkers."""
        return list(self._latest_decisions.values())

    def get_trajectories(self) -> TrajectoryDto:
        """Get full trajectories for walkers and target."""
        with self._lock:
            return TrajectoryDto(
                time=self.coordinator.time,
                target_trajectory=list(self.coordinator.target_trajectory),
                walker_trajectories={
                    w_id: list(traj)
                    for w_id, traj in self.coordinator.walker_trajectories.items()
                },
            )


class SimulationManager:
    """Central in-memory manager for active simulation sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SimulationSession] = {}
        self._lock = threading.RLock()

    def create_simulation(self, config: SimulationConfig) -> SimulationSession:
        """Create and register a new simulation session."""
        with self._lock:
            sim_id = str(uuid.uuid4())[:8]
            session = SimulationSession(simulation_id=sim_id, config=config)
            self._sessions[sim_id] = session
            return session

    def get_session(self, simulation_id: str) -> Optional[SimulationSession]:
        """Retrieve an existing simulation session."""
        with self._lock:
            return self._sessions.get(simulation_id)

    def delete_session(self, simulation_id: str) -> bool:
        """Remove a simulation session."""
        with self._lock:
            return self._sessions.pop(simulation_id, None) is not None

    def list_sessions(self) -> List[SimulationStatusResponse]:
        """List statuses of all active simulation sessions."""
        with self._lock:
            return [session.get_status() for session in self._sessions.values()]

    def clear(self) -> None:
        """Clear all simulation sessions."""
        with self._lock:
            self._sessions.clear()


# Global singleton instance for the FastAPI application
default_manager = SimulationManager()
