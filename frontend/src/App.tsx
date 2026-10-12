import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  api,
  SimulationConfig,
  SimulationStatusResponse,
  GraphResponse,
  GraphNodeDto,
  MetricsDto,
  WalkerStateDto,
  EvidenceItemDto,
  ChurnItemDto,
  DecisionTraceDto,
  TrajectoryDto,
} from './lib/api';
import { Header } from './components/Header';
import { MetricsStrip } from './components/MetricsStrip';
import { ConfigPanel } from './components/ConfigPanel';
import { SimulationToolbar } from './components/SimulationToolbar';
import { GraphExplorer } from './components/GraphExplorer';
import { NodeDetailModal } from './components/NodeDetailModal';
import { DecisionInspector } from './components/DecisionInspector';
import { AnalysisPanels } from './components/AnalysisPanels';
import { AlertCircle, X } from 'lucide-react';

export const App: React.FC = () => {
  // Backend & session state
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [simulation, setSimulation] = useState<SimulationStatusResponse | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsDto | null>(null);
  const [walkers, setWalkers] = useState<WalkerStateDto[]>([]);
  const [evidence, setEvidence] = useState<EvidenceItemDto[]>([]);
  const [churn, setChurn] = useState<ChurnItemDto[]>([]);
  const [latestDecisions, setLatestDecisions] = useState<DecisionTraceDto[]>([]);
  const [trajectory, setTrajectory] = useState<TrajectoryDto | null>(null);

  // Interaction state
  const [selectedNode, setSelectedNode] = useState<GraphNodeDto | null>(null);
  const [targetVisible, setTargetVisible] = useState<boolean>(false);
  const [autoStepping, setAutoStepping] = useState<boolean>(false);
  const [stepSpeed, setStepSpeed] = useState<number>(350);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [requestInFlight, setRequestInFlight] = useState<boolean>(false);

  // Active simulation ID ref to guard against stale responses
  const activeSimIdRef = useRef<string | null>(null);
  const autoSteppingRef = useRef<boolean>(false);
  autoSteppingRef.current = autoStepping;

  // Health check polling
  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      try {
        await api.checkHealth();
        if (isMounted) setBackendOnline(true);
      } catch {
        if (isMounted) setBackendOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 5000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Fetch full simulation details
  const refreshSimulationState = useCallback(async (simId: string) => {
    try {
      const [gData, wData, mData, eData, cData, dData, tData] = await Promise.all([
        api.getSimulationGraph(simId),
        api.getSimulationWalkers(simId),
        api.getSimulationMetrics(simId),
        api.getSimulationEvidence(simId),
        api.getSimulationChurn(simId),
        api.getSimulationDecisions(simId),
        api.getSimulationTrajectory(simId),
      ]);

      if (activeSimIdRef.current === simId) {
        setGraph(gData);
        setWalkers(wData);
        setMetrics(mData);
        setEvidence(eData);
        setChurn(cData);
        setLatestDecisions(dData);
        setTrajectory(tData);
      }
    } catch (err: any) {
      setError(err.message || 'Error refreshing simulation state');
    }
  }, []);

  // Create/Generate Simulation
  const handleGenerate = async (config: SimulationConfig) => {
    setAutoStepping(false);
    setLoading(true);
    setError(null);
    setSelectedNode(null);

    try {
      const newSim = await api.createSimulation(config);
      activeSimIdRef.current = newSim.simulation_id;
      setSimulation(newSim);
      await refreshSimulationState(newSim.simulation_id);
    } catch (err: any) {
      setError(err.message || 'Failed to generate simulation');
    } finally {
      setLoading(false);
    }
  };

  // Perform single discrete step
  const executeStep = useCallback(async (): Promise<boolean> => {
    const simId = activeSimIdRef.current;
    if (!simId || requestInFlight) return false;

    setRequestInFlight(true);
    try {
      const stepRes = await api.stepSimulation(simId);
      if (activeSimIdRef.current === simId) {
        setMetrics(stepRes.metrics);
        setWalkers(stepRes.walkers);
        setLatestDecisions(stepRes.latest_decisions);

        // Update status in simulation state
        setSimulation((prev) =>
          prev
            ? {
                ...prev,
                time: stepRes.time,
                status: stepRes.is_terminated ? 'TERMINATED' : 'RUNNING',
                termination_status: stepRes.termination_status,
                metrics: stepRes.metrics,
              }
            : null
        );

        // Refresh graph, evidence, churn, trajectory concurrently
        const [gData, eData, cData, tData] = await Promise.all([
          api.getSimulationGraph(simId),
          api.getSimulationEvidence(simId),
          api.getSimulationChurn(simId),
          api.getSimulationTrajectory(simId),
        ]);

        if (activeSimIdRef.current === simId) {
          setGraph(gData);
          setEvidence(eData);
          setChurn(cData);
          setTrajectory(tData);

          // Update selected node if currently open
          if (selectedNode) {
            const updated = gData.nodes.find((n) => n.id === selectedNode.id);
            if (updated) setSelectedNode(updated);
          }
        }

        if (stepRes.is_terminated) {
          setAutoStepping(false);
          return false;
        }
        return true;
      }
      return false;
    } catch (err: any) {
      setError(err.message || 'Error executing simulation step');
      setAutoStepping(false);
      return false;
    } finally {
      setRequestInFlight(false);
    }
  }, [requestInFlight, selectedNode]);

  // Step trigger
  const handleStep = async () => {
    if (autoStepping || !simulation || simulation.termination_status) return;
    await executeStep();
  };

  // Reset trigger
  const handleReset = async () => {
    setAutoStepping(false);
    const simId = activeSimIdRef.current;
    if (!simId) return;

    setRequestInFlight(true);
    setError(null);
    try {
      const resetSim = await api.resetSimulation(simId);
      if (activeSimIdRef.current === simId) {
        setSimulation(resetSim);
        setSelectedNode(null);
        await refreshSimulationState(simId);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to reset simulation');
    } finally {
      setRequestInFlight(false);
    }
  };

  // Start automatic stepping loop
  const handleStart = () => {
    if (!simulation || simulation.termination_status) return;
    setAutoStepping(true);
  };

  // Pause automatic stepping
  const handlePause = () => {
    setAutoStepping(false);
  };

  // Auto-stepping loop timer
  useEffect(() => {
    if (!autoStepping) return;

    let isTimerActive = true;
    const runLoop = async () => {
      if (!isTimerActive || !autoSteppingRef.current) return;
      const canContinue = await executeStep();
      if (isTimerActive && autoSteppingRef.current && canContinue) {
        setTimeout(runLoop, stepSpeed);
      }
    };

    const timer = setTimeout(runLoop, stepSpeed);
    return () => {
      isTimerActive = false;
      clearTimeout(timer);
    };
  }, [autoStepping, stepSpeed, executeStep]);

  return (
    <div className="min-h-screen bg-graphite-950 text-slate-100 flex flex-col font-sans">
      {/* 1. Application Header */}
      <Header
        backendOnline={backendOnline}
        simulation={simulation}
        autoStepping={autoStepping}
      />

      {/* 2. Metrics Strip */}
      <MetricsStrip
        metrics={metrics}
        totalNodes={graph?.nodes.length || simulation?.config.num_nodes || 0}
      />

      {/* Error Banner */}
      {error && (
        <div className="bg-rose-950/80 border-b border-rose-800 text-rose-200 px-5 py-2 text-xs font-mono flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="p-1 hover:bg-rose-900 rounded text-rose-300"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* 3. Main Workstation Body */}
      <main className="flex-1 p-4 grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Sidebar: Experiment Configuration */}
        <aside className="lg:col-span-3 flex flex-col h-full min-h-[580px]">
          <ConfigPanel
            onGenerate={handleGenerate}
            loading={loading}
            disabled={autoStepping || requestInFlight}
          />
        </aside>

        {/* Center & Right Column: Simulation Controls, Graph Explorer, Decision Inspector, Analytics */}
        <section className="lg:col-span-9 flex flex-col space-y-4">
          {/* Top Control Toolbar */}
          <SimulationToolbar
            simulation={simulation}
            autoStepping={autoStepping}
            stepSpeed={stepSpeed}
            onStart={handleStart}
            onPause={handlePause}
            onStep={handleStep}
            onReset={handleReset}
            onSpeedChange={setStepSpeed}
            disabled={loading}
            requestInFlight={requestInFlight}
          />

          {/* Center Split: Interactive Graph Explorer (60%) & Decision Inspector (40%) */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 min-h-[460px]">
            {/* Interactive Graph Explorer */}
            <div className="xl:col-span-7 h-full min-h-[440px] relative">
              <GraphExplorer
                graph={graph}
                selectedNodeId={selectedNode ? selectedNode.id : null}
                onSelectNode={setSelectedNode}
                targetVisible={targetVisible}
                onToggleTargetVisible={setTargetVisible}
              />

              {/* Node Detail Floating Modal */}
              <NodeDetailModal
                node={selectedNode}
                evidence={evidence}
                targetVisible={targetVisible}
                onClose={() => setSelectedNode(null)}
              />
            </div>

            {/* Decision & Cognitive State Inspector */}
            <div className="xl:col-span-5 h-full min-h-[440px]">
              <DecisionInspector
                isAesw={simulation?.is_aesw ?? true}
                algorithmName={simulation?.algorithm ?? 'aesw'}
                walkers={walkers}
                latestDecisions={latestDecisions}
              />
            </div>
          </div>

          {/* Lower Empirical Analysis Subsystem (Evidence, Churn, Activity, Trajectory) */}
          <div className="h-72 min-h-[260px]">
            <AnalysisPanels
              evidence={evidence}
              churn={churn}
              walkers={walkers}
              trajectory={trajectory}
              targetVisible={targetVisible}
              isAesw={simulation?.is_aesw ?? true}
            />
          </div>
        </section>
      </main>
    </div>
  );
};
