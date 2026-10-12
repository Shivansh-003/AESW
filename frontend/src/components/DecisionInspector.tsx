import React, { useState } from 'react';
import { WalkerStateDto, DecisionTraceDto } from '../lib/api';
import { Brain, ArrowRight, Table } from 'lucide-react';

interface DecisionInspectorProps {
  isAesw: boolean;
  algorithmName: string;
  walkers: WalkerStateDto[];
  latestDecisions: DecisionTraceDto[];
}

export const DecisionInspector: React.FC<DecisionInspectorProps> = ({
  isAesw,
  algorithmName,
  walkers,
  latestDecisions,
}) => {
  const [selectedWalkerId, setSelectedWalkerId] = useState<number>(0);

  // Selected walker and decision for inspection (both AESW and baselines)
  const walker = walkers.find((w) => w.walker_id === selectedWalkerId) || walkers[0];
  const decision =
    latestDecisions.find((d) => d.walker_id === selectedWalkerId) || walker?.latest_decision;

  // If non-AESW algorithm, display baseline policy explanation AND candidate inspection
  if (!isAesw) {
    const isDegree = algorithmName.includes('degree');
    const isAnt = algorithmName.includes('ant_colony');
    const isFlooding = algorithmName.includes('flooding');
    const isNb = algorithmName.includes('non_backtracking');

    return (
      <div className="bg-graphite-900 border border-slate-800 rounded-lg p-4 h-full flex flex-col text-xs font-mono">
        {/* Header & Walker Tabs */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-slate-400" />
            <h2 className="font-semibold text-slate-200 uppercase tracking-wider text-xs">
              Algorithm Inspector
            </h2>
          </div>

          {/* Walker Selector Tabs */}
          {walkers.length > 1 && (
            <div className="flex items-center bg-graphite-950 p-0.5 rounded border border-slate-800">
              {walkers.map((w) => (
                <button
                  key={`tab-baseline-${w.walker_id}`}
                  type="button"
                  onClick={() => setSelectedWalkerId(w.walker_id)}
                  className={`px-2 py-0.5 rounded text-[11px] ${
                    selectedWalkerId === w.walker_id
                      ? 'bg-cyan-600 text-slate-950 font-bold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Walker #{w.walker_id}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-3 flex-1 overflow-y-auto pr-1">
          {/* Policy Overview Card */}
          <div className="bg-graphite-950 p-3 rounded border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Active Baseline:</span>
              <span className="text-cyan-400 font-bold uppercase">{algorithmName}</span>
            </div>

            <div className="text-slate-300 text-[11px] leading-relaxed pt-1 border-t border-slate-800">
              {algorithmName.includes('random_walk') && (
                <p>
                  <strong>Unbiased Random Walk Policy</strong>: At each step, the walker inspects all locally observable active incident edges and transitions to an active neighbor chosen uniformly at random without cognitive memory or evidence decay.
                </p>
              )}
              {algorithmName.includes('non_backtracking') && (
                <p>
                  <strong>Non-Backtracking Walk Policy</strong>: Avoids immediate backtracking to the previous vertex when alternative active incident links exist, effectively reducing cyclic 2-step traps.
                </p>
              )}
              {algorithmName.includes('k_walk') && (
                <p>
                  <strong>k Independent Random Walkers</strong>: Disperses k uncoordinated random walkers throughout the network simultaneously, evaluating coverage speedup without information sharing.
                </p>
              )}
              {algorithmName.includes('degree') && (
                <p>
                  <strong>Degree-Based Heuristic Walk</strong>: Preferentially transitions toward the active neighbor vertex exhibiting the highest locally observed degree.
                </p>
              )}
              {algorithmName.includes('flooding') && (
                <p>
                  <strong>Flooding / Frontier Discovery</strong>: Maintains a local boundary of uninspected vertices, stepping toward unseen network frontiers to maximize topological exploration.
                </p>
              )}
              {algorithmName.includes('ant_colony') && (
                <p>
                  <strong>Ant Colony Optimization Walk</strong>: Employs local synthetic pheromone trails with exponential evaporation and reinforcement upon discovering proximity cues.
                </p>
              )}
            </div>
          </div>

          {/* Action / Next Hop Status */}
          {walker && (
            <div className="bg-graphite-950 p-2.5 rounded border border-slate-800 flex items-center justify-between">
              <div>
                <span className="text-[10px] text-slate-500 block mb-0.5">CURRENT LOCATION</span>
                <span className="font-bold text-slate-200">Node {walker.current_node}</span>
              </div>
              <div className="text-right">
                <span className="text-[10px] text-slate-500 block mb-0.5">SELECTED ACTION</span>
                <div className="flex items-center gap-1 font-bold text-cyan-400 text-xs justify-end">
                  <span>{decision?.action_type || walker.recent_action || 'STAY'}</span>
                  {decision?.destination !== null && decision?.destination !== undefined ? (
                    <>
                      <ArrowRight className="w-3 h-3 text-slate-500" />
                      <span className="text-slate-100">Node {String(decision.destination)}</span>
                    </>
                  ) : walker.recent_destination !== null && walker.recent_destination !== undefined ? (
                    <>
                      <ArrowRight className="w-3 h-3 text-slate-500" />
                      <span className="text-slate-100">Node {String(walker.recent_destination)}</span>
                    </>
                  ) : null}
                </div>
              </div>
            </div>
          )}

          {/* Baseline Candidate Table */}
          {decision && decision.candidate_scores && decision.candidate_scores.length > 0 ? (
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] text-slate-300 font-semibold flex items-center gap-1.5">
                  <Table className="w-3.5 h-3.5 text-cyan-400" />
                  Observed Neighbor Inspection
                </span>
                <span className="text-[10px] text-slate-500">
                  {isDegree
                    ? 'Score = Observed Degree d_obs(v) + ε'
                    : isAnt
                    ? 'Score = Pheromone τ(u,v)^α'
                    : isFlooding
                    ? 'Status = Frontier Priority'
                    : isNb
                    ? 'Filter = Non-Predecessor'
                    : 'Probability = Uniform (1 / k)'}
                </span>
              </div>

              <div className="bg-graphite-950 rounded border border-slate-800 overflow-x-auto">
                <table className="w-full text-[11px] text-left">
                  <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800 text-[10px]">
                    <tr>
                      <th className="py-1 px-2">Neighbor</th>
                      {isDegree && <th className="py-1 px-2 text-right">Observed Degree</th>}
                      {isAnt && <th className="py-1 px-2 text-right">Pheromone (τ)</th>}
                      {isFlooding && <th className="py-1 px-2 text-right">Frontier Status</th>}
                      {isNb && <th className="py-1 px-2 text-right">Predecessor</th>}
                      <th className="py-1 px-2 text-right">Heuristic Weight</th>
                      <th className="py-1 px-2 text-right">Selection Prob</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {decision.candidate_scores.map((cand) => {
                      const isChosen = String(cand.node_id) === String(decision.selected_node);
                      return (
                        <tr
                          key={`cand-${cand.node_id}`}
                          className={`hover:bg-slate-900/50 ${
                            isChosen ? 'bg-cyan-950/40 text-cyan-300 font-semibold' : 'text-slate-300'
                          }`}
                        >
                          <td className="py-1 px-2 flex items-center gap-1.5">
                            {isChosen && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 inline-block" />}
                            <span>Node {cand.node_id}</span>
                          </td>
                          {isDegree && (
                            <td className="py-1 px-2 text-right text-cyan-300 font-mono">
                              {cand.novelty > 0 ? cand.novelty.toFixed(1) : '—'}
                            </td>
                          )}
                          {isAnt && (
                            <td className="py-1 px-2 text-right text-emerald-400 font-mono">
                              {cand.positive_evidence.toFixed(2)}
                            </td>
                          )}
                          {isFlooding && (
                            <td className="py-1 px-2 text-right font-mono">
                              <span
                                className={`px-1.5 py-0.5 rounded text-[10px] ${
                                  cand.novelty > 0
                                    ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/60'
                                    : 'bg-slate-900 text-slate-500'
                                }`}
                              >
                                {cand.novelty > 0 ? 'UNVISITED' : 'VISITED'}
                              </span>
                            </td>
                          )}
                          {isNb && (
                            <td className="py-1 px-2 text-right font-mono text-slate-400">
                              {cand.revisit_penalty > 0 ? 'BACKTRACK' : 'VALID'}
                            </td>
                          )}
                          <td className="py-1 px-2 text-right font-mono text-slate-200">
                            {cand.total_score.toFixed(3)}
                          </td>
                          <td className="py-1 px-2 text-right font-mono">
                            <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-cyan-300">
                              {(cand.probability * 100).toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="bg-graphite-950 p-3 rounded border border-slate-800/80 text-center text-slate-500 text-[11px] italic">
              Advance a simulation step to observe locally evaluated neighbors.
            </div>
          )}

          <div className="pt-2 text-slate-500 text-[10px]">
            Notice: Adaptive Evidence-Sharing Walkers (AESW) specific cognitive inspection (churn decay, candidate softmax scoring, mode controller) is only active when running AESW.
          </div>
        </div>
      </div>
    );
  }

  // AESW cognitive inspection
  return (
    <div className="bg-graphite-900 border border-slate-800 rounded-lg p-4 h-full flex flex-col text-xs font-mono">
      {/* Header & Walker Tabs */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
        <div className="flex items-center space-x-2">
          <Brain className="w-4 h-4 text-cyan-400" />
          <h2 className="font-semibold text-slate-100 uppercase tracking-wider text-xs">
            AESW Decision Inspector
          </h2>
        </div>

        {/* Walker Selector Tabs */}
        {walkers.length > 1 && (
          <div className="flex items-center bg-graphite-950 p-0.5 rounded border border-slate-800">
            {walkers.map((w) => (
              <button
                key={`tab-${w.walker_id}`}
                type="button"
                onClick={() => setSelectedWalkerId(w.walker_id)}
                className={`px-2 py-0.5 rounded text-[11px] ${
                  selectedWalkerId === w.walker_id
                    ? 'bg-cyan-600 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Walker #{w.walker_id}
              </button>
            ))}
          </div>
        )}
      </div>

      {!walker ? (
        <div className="text-slate-500 text-center py-6">Waiting for walker instantiation...</div>
      ) : (
        <div className="space-y-3 flex-1 overflow-y-auto pr-1">
          {/* Cognitive State Badges */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {/* Mode */}
            <div className="bg-graphite-950 p-2 rounded border border-slate-800">
              <span className="text-[10px] text-slate-500 block mb-0.5">SEARCH MODE</span>
              <span
                className={`font-bold px-1.5 py-0.5 rounded text-[11px] inline-block ${
                  decision?.mode === 'LONG_JUMP'
                    ? 'bg-amber-950/70 border border-amber-600 text-amber-300'
                    : 'bg-cyan-950/70 border border-cyan-600 text-cyan-300'
                }`}
              >
                {decision?.mode || walker.mode || 'LOCAL'}
              </span>
            </div>

            {/* Churn lambda_hat */}
            <div className="bg-graphite-950 p-2 rounded border border-slate-800">
              <span className="text-[10px] text-slate-500 block mb-0.5">CHURN ESTIMATE (λ̂)</span>
              <span className="font-bold text-slate-200 text-sm">
                {(decision?.estimated_lambda ?? walker.estimated_lambda ?? 0.0).toFixed(4)}
              </span>
            </div>

            {/* Stagnation Counter */}
            <div className="bg-graphite-950 p-2 rounded border border-slate-800">
              <span className="text-[10px] text-slate-500 block mb-0.5">STAGNATION STEPS</span>
              <span className="font-bold text-slate-200 text-sm">
                {decision?.stagnation_steps ?? walker.stagnation_steps ?? 0}
              </span>
            </div>

            {/* Next Hop Action */}
            <div className="bg-graphite-950 p-2 rounded border border-slate-800">
              <span className="text-[10px] text-slate-500 block mb-0.5">SELECTED ACTION</span>
              <div className="flex items-center gap-1 font-bold text-cyan-400 text-xs">
                <span>{decision?.action_type || walker.recent_action || 'MOVE'}</span>
                {decision?.destination !== null && decision?.destination !== undefined && (
                  <>
                    <ArrowRight className="w-3 h-3 text-slate-500" />
                    <span className="text-slate-100">Node {String(decision.destination)}</span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Candidate Neighbor Ranking Table */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] text-slate-300 font-semibold flex items-center gap-1.5">
                <Table className="w-3.5 h-3.5 text-cyan-400" />
                Active Neighbor Candidate Ranking
              </span>
              <span className="text-[10px] text-slate-500">
                S(u) = α·P(u) + β·N(u) - γ·R(u) - δ·D(u)
              </span>
            </div>

            {!decision || decision.candidate_scores.length === 0 ? (
              <div className="bg-graphite-950 p-3 rounded border border-slate-800/80 text-center text-slate-500 text-[11px] italic">
                {decision?.mode === 'LONG_JUMP'
                  ? 'Walker in LONG_JUMP mode: candidate scoring bypassed for environment-mediated dispersion.'
                  : 'No candidate scores recorded for this step yet (advance a step to observe decisions).'}
              </div>
            ) : (
              <div className="bg-graphite-950 rounded border border-slate-800 overflow-x-auto">
                <table className="w-full text-[11px] text-left">
                  <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800 text-[10px]">
                    <tr>
                      <th className="py-1 px-2">Node</th>
                      <th className="py-1 px-2 text-right">Evidence P(u)</th>
                      <th className="py-1 px-2 text-right">Novelty N(u)</th>
                      <th className="py-1 px-2 text-right">Revisit R(u)</th>
                      <th className="py-1 px-2 text-right">Delay D(u)</th>
                      <th className="py-1 px-2 text-right">Score S(u)</th>
                      <th className="py-1 px-2 text-right">Prob P_select</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {decision.candidate_scores.map((cand) => {
                      const isChosen = String(cand.node_id) === String(decision.selected_node);
                      return (
                        <tr
                          key={`cand-${cand.node_id}`}
                          className={`hover:bg-slate-900/50 ${
                            isChosen ? 'bg-cyan-950/40 text-cyan-300 font-semibold' : 'text-slate-300'
                          }`}
                        >
                          <td className="py-1 px-2 flex items-center gap-1.5">
                            {isChosen && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 inline-block" />}
                            <span>Node {cand.node_id}</span>
                          </td>
                          <td className="py-1 px-2 text-right font-mono text-emerald-400">
                            {cand.positive_evidence.toFixed(3)}
                          </td>
                          <td className="py-1 px-2 text-right font-mono text-cyan-400">
                            {cand.novelty.toFixed(3)}
                          </td>
                          <td className="py-1 px-2 text-right font-mono text-amber-400">
                            {cand.revisit_penalty.toFixed(1)}
                          </td>
                          <td className="py-1 px-2 text-right font-mono text-slate-400">
                            {cand.delay.toFixed(1)}
                          </td>
                          <td className="py-1 px-2 text-right font-mono font-bold">
                            {cand.total_score.toFixed(3)}
                          </td>
                          <td className="py-1 px-2 text-right font-mono">
                            <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-cyan-300">
                              {(cand.probability * 100).toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
