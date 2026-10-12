/**
 * AESW Typed API Client
 * =====================
 * Communicates with the FastAPI simulation backend conforming to exact schemas.
 */

export interface SimulationConfig {
  algorithm: string;
  graph_type: string;
  num_nodes: number;
  graph_parameters?: Record<string, any>;
  regime: string;
  p_on?: number | null;
  p_off?: number | null;
  p_move: number;
  target_mode: string;
  signal_radius: number;
  p_d: number;
  p_fa: number;
  neighbor_budget: number;
  walker_count: number;
  seed: number;
  time_horizon: number;
  communication_mode: string;
  c_m: number;
}

export interface MetricsDto {
  algorithm: string;
  search_time: number;
  status: string;
  success: boolean;
  total_moves: number;
  node_revisits: number;
  nodes_visited: number;
  total_messages: number;
  total_cost: number;
  c_m: number;
}

export interface SimulationStatusResponse {
  simulation_id: string;
  status: string;
  termination_status: string | null;
  time: number;
  time_horizon: number;
  algorithm: string;
  is_aesw: boolean;
  seed: number;
  walker_count: number;
  config: SimulationConfig;
  metrics: MetricsDto;
}

export interface GraphNodeDto {
  id: number | string;
  label: string;
  degree: number;
  is_target: boolean;
  is_visited: boolean;
  walker_ids: number[];
  has_evidence: boolean;
  evidence_polarity?: string | null;
  evidence_weight?: number | null;
  x: number;
  y: number;
}

export interface GraphEdgeDto {
  source: number | string;
  target: number | string;
  is_active: boolean;
  state: string;
  is_recent_transition: boolean;
}

export interface GraphResponse {
  simulation_id: string;
  nodes: GraphNodeDto[];
  edges: GraphEdgeDto[];
  target_node: number | string;
  visited_nodes: (number | string)[];
  active_edge_count: number;
  total_edge_count: number;
}

export interface CandidateScoreDto {
  node_id: number | string;
  positive_evidence: number;
  novelty: number;
  revisit_penalty: number;
  delay: number;
  total_score: number;
  probability: number;
}

export interface DecisionTraceDto {
  walker_id: number;
  time: number;
  current_node: number | string;
  mode: string;
  estimated_lambda: number;
  stagnation_steps: number;
  selected_node?: number | string | null;
  action_type: string;
  destination?: number | string | null;
  candidate_scores: CandidateScoreDto[];
}

export interface WalkerStateDto {
  walker_id: number;
  current_node: number | string;
  step_count: number;
  message_count: number;
  mode?: string | null;
  estimated_lambda?: number | null;
  stagnation_steps?: number | null;
  recent_action?: string | null;
  recent_destination?: number | string | null;
  visit_counts: Record<string, number>;
  latest_decision?: DecisionTraceDto | null;
}

export interface EvidenceItemDto {
  node_id: number | string;
  walker_id: number;
  polarity: string;
  confidence: number;
  effective_weight: number;
  effective_confidence: number;
  timestamp: number;
  age: number;
  source: string;
}

export interface ChurnItemDto {
  walker_id: number;
  estimated_lambda: number;
  overlap: number;
  raw_churn_rate: number;
  update_count: number;
  history: Array<{
    time: number;
    lambda_hat: number;
    overlap: number;
    raw_churn: number;
  }>;
}

export interface TrajectoryDto {
  time: number;
  target_trajectory: (number | string)[];
  walker_trajectories: Record<string | number, (number | string)[]>;
}

export interface StepResultResponse {
  simulation_id: string;
  time: number;
  is_terminated: boolean;
  termination_status: string | null;
  metrics: MetricsDto;
  walkers: WalkerStateDto[];
  latest_decisions: DecisionTraceDto[];
}

export interface AlgorithmInfo {
  id: string;
  name: string;
  description: string;
  is_aesw: boolean;
}

export interface GraphTypeInfo {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Trans-Type': 'application/json',
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!res.ok) {
      let errorMsg = `HTTP Error ${res.status}: ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorMsg = errorJson.detail;
        }
      } catch {
        // ignore json parse error
      }
      throw new Error(errorMsg);
    }

    return (await res.json()) as T;
  } catch (err: any) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error(`Unable to connect to backend at ${url}. Ensure the FastAPI server is running.`);
    }
    throw err;
  }
}

export const api = {
  checkHealth: () => request<{ status: string; version: string }>('/api/health'),

  getAlgorithms: () => request<AlgorithmInfo[]>('/api/algorithms'),

  getGraphTypes: () => request<GraphTypeInfo[]>('/api/graph-types'),

  createSimulation: (config: SimulationConfig) =>
    request<SimulationStatusResponse>('/api/simulations', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  getSimulation: (id: string) => request<SimulationStatusResponse>(`/api/simulations/${id}`),

  stepSimulation: (id: string) =>
    request<StepResultResponse>(`/api/simulations/${id}/step`, {
      method: 'POST',
    }),

  resetSimulation: (id: string) =>
    request<SimulationStatusResponse>(`/api/simulations/${id}/reset`, {
      method: 'POST',
    }),

  getSimulationGraph: (id: string) => request<GraphResponse>(`/api/simulations/${id}/graph`),

  getSimulationWalkers: (id: string) => request<WalkerStateDto[]>(`/api/simulations/${id}/walkers`),

  getSimulationMetrics: (id: string) => request<MetricsDto>(`/api/simulations/${id}/metrics`),

  getSimulationEvidence: (id: string) => request<EvidenceItemDto[]>(`/api/simulations/${id}/evidence`),

  getSimulationChurn: (id: string) => request<ChurnItemDto[]>(`/api/simulations/${id}/churn`),

  getSimulationDecisions: (id: string) => request<DecisionTraceDto[]>(`/api/simulations/${id}/decisions`),

  getSimulationTrajectory: (id: string) => request<TrajectoryDto>(`/api/simulations/${id}/trajectory`),

  deleteSimulation: (id: string) =>
    request<{ status: string; simulation_id: string }>(`/api/simulations/${id}`, {
      method: 'DELETE',
    }),
};
