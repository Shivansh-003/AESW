import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { App } from '../App';
import { api, WalkerStateDto } from '../lib/api';

vi.mock('../lib/api', () => {
  const mockApi = {
    checkHealth: vi.fn(),
    getAlgorithms: vi.fn(),
    getGraphTypes: vi.fn(),
    createSimulation: vi.fn(),
    getSimulation: vi.fn(),
    stepSimulation: vi.fn(),
    resetSimulation: vi.fn(),
    getSimulationGraph: vi.fn(),
    getSimulationWalkers: vi.fn(),
    getSimulationMetrics: vi.fn(),
    getSimulationEvidence: vi.fn(),
    getSimulationChurn: vi.fn(),
    getSimulationDecisions: vi.fn(),
    getSimulationTrajectory: vi.fn(),
    deleteSimulation: vi.fn(),
  };
  return { api: mockApi };
});

const sampleSimulation = {
  simulation_id: 'test-sim-123',
  status: 'INITIALIZED',
  termination_status: null,
  time: 0,
  time_horizon: 100,
  algorithm: 'aesw',
  is_aesw: true,
  seed: 42,
  walker_count: 2,
  config: {
    algorithm: 'aesw',
    graph_type: 'er',
    num_nodes: 10,
    regime: 'MEDIUM',
    p_move: 0.25,
    target_mode: 'MOVING',
    signal_radius: 1,
    p_d: 0.9,
    p_fa: 0.05,
    neighbor_budget: 8,
    walker_count: 2,
    seed: 42,
    time_horizon: 100,
    communication_mode: 'PUSH_PULL',
    c_m: 0.1,
  },
  metrics: {
    algorithm: 'aesw',
    search_time: 0,
    status: 'INITIALIZED',
    success: false,
    total_moves: 0,
    node_revisits: 0,
    nodes_visited: 2,
    total_messages: 0,
    total_cost: 0.0,
    c_m: 0.1,
  },
};

const sampleGraph = {
  simulation_id: 'test-sim-123',
  nodes: [
    { id: 0, label: '0', degree: 2, is_target: false, is_visited: true, walker_ids: [0], has_evidence: true, x: 0.2, y: 0.1 },
    { id: 1, label: '1', degree: 3, is_target: true, is_visited: false, walker_ids: [], has_evidence: false, x: -0.3, y: 0.4 },
    { id: 2, label: '2', degree: 1, is_target: false, is_visited: true, walker_ids: [1], has_evidence: false, x: 0.5, y: -0.2 },
  ],
  edges: [
    { source: 0, target: 1, is_active: true, state: 'ON', is_recent_transition: false },
    { source: 1, target: 2, is_active: false, state: 'OFF', is_recent_transition: true },
  ],
  target_node: 1,
  visited_nodes: [0, 2],
  active_edge_count: 1,
  total_edge_count: 2,
};

const sampleWalkers: WalkerStateDto[] = [
  {
    walker_id: 0,
    current_node: 0,
    step_count: 0,
    message_count: 0,
    mode: 'LOCAL',
    estimated_lambda: 0.045,
    stagnation_steps: 0,
    recent_action: 'MOVE',
    recent_destination: 1,
    visit_counts: { '0': 1 },
    latest_decision: null,
  },
  {
    walker_id: 1,
    current_node: 2,
    step_count: 0,
    message_count: 0,
    mode: 'LOCAL',
    estimated_lambda: 0.05,
    stagnation_steps: 0,
    recent_action: 'STAY',
    recent_destination: null,
    visit_counts: { '2': 1 },
    latest_decision: null,
  },
];

const sampleEvidence = [
  {
    node_id: 0,
    walker_id: 0,
    polarity: 'POSITIVE',
    confidence: 1.0,
    effective_weight: 0.95,
    effective_confidence: 0.95,
    timestamp: 0,
    age: 0,
    source: 'DIRECT_VISIT',
  },
];

const sampleChurn = [
  {
    walker_id: 0,
    estimated_lambda: 0.045,
    overlap: 0.75,
    raw_churn_rate: 0.04,
    update_count: 1,
    history: [{ time: 0, lambda_hat: 0.045, overlap: 0.75, raw_churn: 0.04 }],
  },
];

const sampleDecisions = [
  {
    walker_id: 0,
    time: 0,
    current_node: 0,
    mode: 'LOCAL',
    estimated_lambda: 0.045,
    stagnation_steps: 0,
    selected_node: 1,
    action_type: 'MOVE',
    destination: 1,
    candidate_scores: [
      {
        node_id: 1,
        positive_evidence: 0.8,
        novelty: 1.0,
        revisit_penalty: 0.0,
        delay: 1.0,
        total_score: 1.1,
        probability: 0.85,
      },
    ],
  },
];

const sampleTrajectory = {
  time: 0,
  target_trajectory: [1],
  walker_trajectories: { 0: [0], 1: [2] },
};

describe('AESW Research Workstation Frontend', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.checkHealth).mockResolvedValue({ status: 'ok', version: '0.1.0' });
    vi.mocked(api.getAlgorithms).mockResolvedValue([
      { id: 'aesw', name: 'AESW', description: 'AESW policy', is_aesw: true },
      { id: 'random_walk', name: 'Random Walk', description: 'RW policy', is_aesw: false },
    ]);
    vi.mocked(api.getGraphTypes).mockResolvedValue([
      { id: 'er', name: 'Erdos-Renyi', description: 'ER graph', parameters: {} },
    ]);
    vi.mocked(api.createSimulation).mockResolvedValue(sampleSimulation);
    vi.mocked(api.getSimulationGraph).mockResolvedValue(sampleGraph);
    vi.mocked(api.getSimulationWalkers).mockResolvedValue(sampleWalkers);
    vi.mocked(api.getSimulationMetrics).mockResolvedValue(sampleSimulation.metrics);
    vi.mocked(api.getSimulationEvidence).mockResolvedValue(sampleEvidence);
    vi.mocked(api.getSimulationChurn).mockResolvedValue(sampleChurn);
    vi.mocked(api.getSimulationDecisions).mockResolvedValue(sampleDecisions);
    vi.mocked(api.getSimulationTrajectory).mockResolvedValue(sampleTrajectory);
  });

  it('1. Application renders correctly with header and title', async () => {
    render(<App />);
    expect(screen.getAllByText(/AESW/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Dynamic Search Under Uncertainty/i)).toBeInTheDocument();
  });

  it('2. API connection status is displayed', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/API ONLINE/i)).toBeInTheDocument();
    });
  });

  it('3. Generate button triggers createSimulation with parameters and stores session ID', async () => {
    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(api.createSimulation).toHaveBeenCalled();
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });
  });

  it('4. Graph data and nodes render upon generation', async () => {
    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/Topology Explorer/i)).toBeInTheDocument();
      expect(screen.getByText(/\|V\| = 3/i)).toBeInTheDocument();
    });
  });

  it('5. Metrics strip displays backend values accurately', async () => {
    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/Moves \(M\):/i)).toBeInTheDocument();
      expect(screen.getByText(/Coverage:/i)).toBeInTheDocument();
      expect(screen.getByText(/Total Cost \(C\):/i)).toBeInTheDocument();
    });
  });

  it('6. Manual Step advances simulation by 1 step', async () => {
    const stepResponse = {
      simulation_id: 'test-sim-123',
      time: 1,
      is_terminated: false,
      termination_status: null,
      metrics: {
        ...sampleSimulation.metrics,
        search_time: 1,
        total_moves: 2,
        total_cost: 2.0,
      },
      walkers: sampleWalkers.map((w) => ({ ...w, step_count: 1 })),
      latest_decisions: sampleDecisions,
    };
    vi.mocked(api.stepSimulation).mockResolvedValue(stepResponse);

    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });

    const stepButton = screen.getByRole('button', { name: /STEP \(1\)/i });
    fireEvent.click(stepButton);

    await waitFor(() => {
      expect(api.stepSimulation).toHaveBeenCalledWith('test-sim-123');
    });
  });

  it('7. Reset button calls resetSimulation and restores initial state', async () => {
    vi.mocked(api.resetSimulation).mockResolvedValue({
      ...sampleSimulation,
      time: 0,
    });

    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });

    const resetButton = screen.getByRole('button', { name: /^RESET$/i });
    fireEvent.click(resetButton);

    await waitFor(() => {
      expect(api.resetSimulation).toHaveBeenCalledWith('test-sim-123');
    });
  });

  it('8. Start and Pause controls toggle automatic stepping', async () => {
    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });

    const startButton = screen.getByRole('button', { name: /START/i });
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /PAUSE/i })).toBeInTheDocument();
    });

    const pauseButton = screen.getByRole('button', { name: /PAUSE/i });
    fireEvent.click(pauseButton);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /START/i })).toBeInTheDocument();
    });
  });

  it('9. AESW decision inspector displays candidate scores and probabilities', async () => {
    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/AESW Decision Inspector/i)).toBeInTheDocument();
      expect(screen.getByText(/Active Neighbor Candidate Ranking/i)).toBeInTheDocument();
      expect(screen.getByText(/85.0%/i)).toBeInTheDocument();
    });
  });

  it('10. Non-AESW algorithm adapts UI to baseline inspector and renders candidate ranking when stepped', async () => {
    const nonAeswSim = {
      ...sampleSimulation,
      algorithm: 'random_walk',
      is_aesw: false,
    };
    const baselineDecisions = [
      {
        walker_id: 0,
        time: 1,
        current_node: 0,
        mode: 'BASELINE',
        estimated_lambda: 0.0,
        stagnation_steps: 0,
        selected_node: 1,
        action_type: 'MOVE',
        destination: 1,
        candidate_scores: [
          {
            node_id: 1,
            positive_evidence: 0.0,
            novelty: 0.0,
            revisit_penalty: 0.0,
            delay: 1.0,
            total_score: 1.0,
            probability: 0.5,
          },
          {
            node_id: 2,
            positive_evidence: 0.0,
            novelty: 0.0,
            revisit_penalty: 0.0,
            delay: 1.0,
            total_score: 1.0,
            probability: 0.5,
          },
        ],
      },
    ];
    const baselineWalkers = [
      {
        walker_id: 0,
        current_node: 1,
        step_count: 1,
        message_count: 0,
        mode: 'BASELINE',
        estimated_lambda: 0.0,
        stagnation_steps: 0,
        recent_action: 'MOVE',
        recent_destination: 1,
        visit_counts: { '0': 1, '1': 1 },
        latest_decision: baselineDecisions[0],
      },
    ];
    vi.mocked(api.createSimulation).mockResolvedValue(nonAeswSim);
    vi.mocked(api.getSimulationDecisions).mockResolvedValue(baselineDecisions);
    vi.mocked(api.getSimulationWalkers).mockResolvedValue(baselineWalkers);
    vi.mocked(api.stepSimulation).mockResolvedValue({
      simulation_id: 'test-sim-123',
      time: 1,
      is_terminated: false,
      termination_status: null,
      metrics: { ...sampleSimulation.metrics, search_time: 1, total_moves: 1 },
      walkers: baselineWalkers,
      latest_decisions: baselineDecisions,
    });

    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/Algorithm Inspector/i)).toBeInTheDocument();
      expect(screen.getByText(/Unbiased Random Walk Policy/i)).toBeInTheDocument();
      expect(screen.getByText(/Observed Neighbor Inspection/i)).toBeInTheDocument();
      expect(screen.getAllByText(/50.0%/i).length).toBeGreaterThan(0);
    });

    // Check Walker Activity tab displays baseline data
    const activityTab = screen.getByRole('button', { name: /Walker Activity/i });
    fireEvent.click(activityTab);

    await waitFor(() => {
      expect(screen.getAllByText(/Walker #0/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/BASELINE/i).length).toBeGreaterThan(0);
    });
  });

  it('11. Displays API error banner when an error occurs', async () => {
    vi.mocked(api.createSimulation).mockRejectedValue(new Error('Network failure'));

    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/Network failure/i)).toBeInTheDocument();
    });
  });

  it('12. Auto-stepping stops automatically when simulation finishes (is_terminated: true)', async () => {
    vi.mocked(api.stepSimulation).mockResolvedValue({
      simulation_id: 'test-sim-123',
      time: 10,
      is_terminated: true,
      termination_status: 'SUCCESS',
      metrics: {
        ...sampleSimulation.metrics,
        search_time: 10,
        success: true,
        status: 'SUCCESS',
      },
      walkers: sampleWalkers,
      latest_decisions: sampleDecisions,
    });

    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });

    // Start auto stepping
    const startButton = screen.getByRole('button', { name: /START/i });
    fireEvent.click(startButton);

    // After step terminates, autoStepping turns off and START button reappears
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /START/i })).toBeInTheDocument();
      expect(screen.getByText(/ACQUIRED \(SUCCESS\)/i)).toBeInTheDocument();
    });
  });

  it('13. Reset cancels active auto-stepping before issuing reset', async () => {
    vi.mocked(api.stepSimulation).mockResolvedValue({
      simulation_id: 'test-sim-123',
      time: 1,
      is_terminated: false,
      termination_status: null,
      metrics: sampleSimulation.metrics,
      walkers: sampleWalkers,
      latest_decisions: sampleDecisions,
    });
    vi.mocked(api.resetSimulation).mockResolvedValue({
      ...sampleSimulation,
      time: 0,
    });

    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });

    // Start auto-stepping
    const startButton = screen.getByRole('button', { name: /START/i });
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /PAUSE/i })).toBeInTheDocument();
    });

    // Click RESET
    const resetButton = screen.getByRole('button', { name: /^RESET$/i });
    fireEvent.click(resetButton);

    await waitFor(() => {
      expect(api.resetSimulation).toHaveBeenCalledWith('test-sim-123');
      // Auto-stepping must be stopped (showing START instead of PAUSE)
      expect(screen.getByRole('button', { name: /START/i })).toBeInTheDocument();
    });
  });

  it('14. Decision Inspector shows LONG_JUMP explanation when candidate scores are bypassed', async () => {
    const longJumpDecision = [
      {
        walker_id: 0,
        time: 5,
        current_node: 18,
        mode: 'LONG_JUMP',
        estimated_lambda: 0.045,
        stagnation_steps: 5,
        selected_node: null,
        action_type: 'JUMP',
        destination: null,
        candidate_scores: [],
      },
    ];

    vi.mocked(api.stepSimulation).mockResolvedValue({
      simulation_id: 'test-sim-123',
      time: 5,
      is_terminated: false,
      termination_status: null,
      metrics: sampleSimulation.metrics,
      walkers: sampleWalkers,
      latest_decisions: longJumpDecision,
    });

    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });

    const stepButton = screen.getByRole('button', { name: /STEP \(1\)/i });
    fireEvent.click(stepButton);

    await waitFor(() => {
      expect(
        screen.getByText(/Walker in LONG_JUMP mode: candidate scoring bypassed for environment-mediated dispersion/i)
      ).toBeInTheDocument();
    });
  });

  it('15. Decision Inspector updates candidate ranking when changing selected walker tab', async () => {
    const multiWalkerDecisions = [
      {
        walker_id: 0,
        time: 7,
        current_node: 18,
        mode: 'LONG_JUMP',
        estimated_lambda: 0.045,
        stagnation_steps: 5,
        selected_node: null,
        action_type: 'JUMP',
        destination: null,
        candidate_scores: [],
      },
      {
        walker_id: 1,
        time: 7,
        current_node: 12,
        mode: 'LOCAL',
        estimated_lambda: 0.042,
        stagnation_steps: 0,
        selected_node: 10,
        action_type: 'MOVE',
        destination: 10,
        candidate_scores: [
          {
            node_id: 10,
            positive_evidence: 0.9,
            novelty: 0.8,
            revisit_penalty: 0.0,
            delay: 1.0,
            total_score: 1.3,
            probability: 0.92,
          },
        ],
      },
    ];

    vi.mocked(api.stepSimulation).mockResolvedValue({
      simulation_id: 'test-sim-123',
      time: 7,
      is_terminated: false,
      termination_status: null,
      metrics: sampleSimulation.metrics,
      walkers: sampleWalkers,
      latest_decisions: multiWalkerDecisions,
    });

    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });

    const stepButton = screen.getByRole('button', { name: /STEP \(1\)/i });
    fireEvent.click(stepButton);

    // Initially Walker #0 is selected, showing LONG_JUMP message
    await waitFor(() => {
      expect(
        screen.getByText(/Walker in LONG_JUMP mode: candidate scoring bypassed for environment-mediated dispersion/i)
      ).toBeInTheDocument();
    });

    // Switch to Walker #1
    const walker1Tab = screen.getByRole('button', { name: /Walker #1/i });
    fireEvent.click(walker1Tab);

    // Now Walker #1 candidate ranking table is visible with probability 92.0%
    await waitFor(() => {
      expect(screen.getByText(/92.0%/i)).toBeInTheDocument();
      expect(screen.getAllByText(/Node 10/i).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('16. Metrics strip displays SEARCHING... during intermediate states and BUDGET EXHAUSTED upon true budget exhaustion', async () => {
    // 1. Initialized simulation with status: 'INITIALIZED'
    render(<App />);
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(screen.getByText(/test-sim-123/i)).toBeInTheDocument();
    });

    // When status is INITIALIZED, metrics outcome displays SEARCHING...
    expect(screen.getByText(/SEARCHING\.\.\./i)).toBeInTheDocument();
    expect(screen.queryByText(/BUDGET EXHAUSTED/i)).not.toBeInTheDocument();

    // 2. Active running step with status: 'RUNNING'
    vi.mocked(api.stepSimulation).mockResolvedValueOnce({
      simulation_id: 'test-sim-123',
      time: 1,
      is_terminated: false,
      termination_status: null,
      metrics: {
        ...sampleSimulation.metrics,
        search_time: 1,
        status: 'RUNNING',
      },
      walkers: sampleWalkers,
      latest_decisions: sampleDecisions,
    });

    const stepButton = screen.getByRole('button', { name: /STEP \(1\)/i });
    fireEvent.click(stepButton);

    await waitFor(() => {
      expect(screen.getByText(/SEARCHING\.\.\./i)).toBeInTheDocument();
      expect(screen.queryByText(/BUDGET EXHAUSTED/i)).not.toBeInTheDocument();
    });

    // 3. Genuine budget exhaustion with status: 'BUDGET_EXHAUSTED'
    vi.mocked(api.stepSimulation).mockResolvedValueOnce({
      simulation_id: 'test-sim-123',
      time: 100,
      is_terminated: true,
      termination_status: 'BUDGET_EXHAUSTED',
      metrics: {
        ...sampleSimulation.metrics,
        search_time: 100,
        status: 'BUDGET_EXHAUSTED',
        success: false,
      },
      walkers: sampleWalkers,
      latest_decisions: sampleDecisions,
    });

    fireEvent.click(stepButton);

    await waitFor(() => {
      expect(screen.getByText(/BUDGET EXHAUSTED/i)).toBeInTheDocument();
      expect(screen.queryByText(/SEARCHING\.\.\./i)).not.toBeInTheDocument();
    });
  });

  it('17. Allows direct typing for integers and decimal values in configuration panel and submits updated config', async () => {
    render(<App />);

    // Target integer input: Node Count (|V|)
    const nodeCountInput = screen.getByDisplayValue('30');
    // Simulate user selecting and typing replacement integer '45'
    fireEvent.change(nodeCountInput, { target: { value: '45' } });
    expect(nodeCountInput).toHaveValue(45);

    // Target decimal input: Detection (p_d)
    const pdInput = screen.getByDisplayValue('0.9');
    // Simulate user typing replacement decimal '0.75'
    fireEvent.change(pdInput, { target: { value: '0.75' } });
    expect(pdInput).toHaveValue(0.75);

    // Target decimal input with intermediate empty / dot keystroke: p_move
    const pMoveInput = screen.getByDisplayValue('0.25');
    // User clears field
    fireEvent.change(pMoveInput, { target: { value: '' } });
    // User types dot
    fireEvent.change(pMoveInput, { target: { value: '.' } });
    // User completes typing '0.4'
    fireEvent.change(pMoveInput, { target: { value: '0.4' } });
    expect(pMoveInput).toHaveValue(0.4);

    // Submit configuration
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(api.createSimulation).toHaveBeenCalledWith(
        expect.objectContaining({
          num_nodes: 45,
          p_d: 0.75,
          p_move: 0.4,
        })
      );
    });
  });

  it('18. Allows typing 0.05 into Detection Probability (p_d) and retains value on blur and submission', async () => {
    render(<App />);

    // Detection (p_d) has default value 0.9
    const pdInput = screen.getByDisplayValue('0.9');

    // Simulate clearing and typing 0.05
    fireEvent.change(pdInput, { target: { value: '' } });
    fireEvent.change(pdInput, { target: { value: '0.05' } });
    expect(pdInput).toHaveValue(0.05);

    // Simulate blur event (clicking elsewhere)
    fireEvent.blur(pdInput);

    // Verify it remains 0.05 and does not reset to 0.1
    expect(pdInput).toHaveValue(0.05);

    // Submit configuration
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(api.createSimulation).toHaveBeenCalledWith(
        expect.objectContaining({
          p_d: 0.05,
        })
      );
    });
  });

  it('19. Switching to 2D Grid passes correct parameters without hardcoded rows and cols', async () => {
    render(<App />);

    // Switch graph type to 2D Grid
    const graphSelect = screen.getAllByRole('combobox')[1]; // graph family select
    fireEvent.change(graphSelect, { target: { value: 'grid' } });

    // Submit configuration
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(api.createSimulation).toHaveBeenCalledWith(
        expect.objectContaining({
          graph_type: 'grid',
          graph_parameters: { obstacle_ratio: 0.1 },
        })
      );
    });
  });

  it('20. Switching to RGG submits radius 0.28 without step mismatch', async () => {
    render(<App />);

    // Switch graph type to RGG
    const graphSelect = screen.getAllByRole('combobox')[1];
    fireEvent.change(graphSelect, { target: { value: 'rgg' } });

    const radiusInput = screen.getByDisplayValue('0.28');
    expect(radiusInput).toBeInTheDocument();
    expect(radiusInput).toHaveAttribute('step', '0.01');
    expect(radiusInput).toHaveAttribute('min', '0.01');

    // Submit configuration
    const genButton = screen.getByRole('button', { name: /Generate Simulation/i });
    fireEvent.click(genButton);

    await waitFor(() => {
      expect(api.createSimulation).toHaveBeenCalledWith(
        expect.objectContaining({
          graph_type: 'rgg',
          graph_parameters: { radius: 0.28 },
        })
      );
    });
  });
});
