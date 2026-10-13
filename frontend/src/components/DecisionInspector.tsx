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
            <div className="bg-graphite-950 p-2 rounded border border-slate-800" title="Operational Search Mode: LOCAL explores around high-confidence cues; LONG_JUMP executes dispersion when information stagnates.">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-[10px] text-slate-500">SEARCH MODE</span>
                <span className="text-[9px] text-slate-600 font-normal">W=5</span>
              </div>
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
            <div className="bg-graphite-950 p-2 rounded border border-slate-800" title="Exponential moving average of observed link churn rate λ̂_t, governing temporal decay of evidence.">
              <span className="text-[10px] text-slate-500 block mb-0.5">CHURN ESTIMATE (λ̂)</span>
              <span className="font-bold text-slate-200 text-sm">
                {(decision?.estimated_lambda ?? walker.estimated_lambda ?? 0.0).toFixed(4)}
              </span>
            </div>

            {/* Stagnation Counter */}
            <div className="bg-graphite-950 p-2 rounded border border-slate-800" title="Consecutive decision steps without useful positive/novel evidence. Triggers LONG_JUMP when stagnation ≥ W (W=5).">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-[10px] text-slate-500">STAGNATION</span>
                <span className="text-[9px] text-slate-600">Threshold: 5</span>
              </div>
              <div className="flex items-baseline gap-1.5">
                <span className="font-bold text-slate-200 text-sm">
                  {decision?.stagnation_steps ?? walker.stagnation_steps ?? 0}
                </span>
                <span className="text-[10px] text-slate-500">/ 5 steps</span>
              </div>
            </div>

            {/* Next Hop Action */}
            <div className="bg-graphite-950 p-2 rounded border border-slate-800" title="Action evaluated and dispatched for physical execution on the environment.">
              <span className="text-[10px] text-slate-500 block mb-0.5">SELECTED ACTION</span>
              <div className="flex items-center gap-1 font-bold text-cyan-400 text-xs">
                <span>{decision?.action_type || walker.recent_action || 'MOVE'}</span>
                {decision?.destination !== null && decision?.destination !== undefined ? (
                  <>
                    <ArrowRight className="w-3 h-3 text-slate-500" />
                    <span className="text-slate-100">Node {String(decision.destination)}</span>
                  </>
                ) : (
                  walker.recent_destination !== null && walker.recent_destination !== undefined && (
                    <>
                      <ArrowRight className="w-3 h-3 text-slate-500" />
                      <span className="text-slate-100">Node {String(walker.recent_destination)}</span>
                    </>
                  )
                )}
              </div>
            </div>
          </div>

          {/* Search Mode Transition & Reason Explanation */}
          {decision?.mode === 'LONG_JUMP' ? (
            <div className="bg-amber-950/30 border border-amber-800/60 rounded p-2.5 text-[11px] text-amber-200/90 space-y-1">
              <div className="font-bold flex items-center justify-between text-amber-300">
                <span>Mode Transition: LONG_JUMP Dispersion Active</span>
                <span className="text-[10px] font-mono bg-amber-900/60 px-1.5 py-0.5 rounded text-amber-200">
                  Stagnation: {decision.stagnation_steps} ≥ 5
                </span>
              </div>
              <p className="text-[10px] text-amber-300/80 leading-relaxed">
                Walker exceeded the information horizon window W=5 without novel evidence. Local candidate scoring is bypassed in favor of uniform topological dispersion across non-local vertices to escape search stagnation.
              </p>
            </div>
          ) : (
            <div className="bg-slate-950 p-2 rounded border border-slate-800/80 text-[10px] text-slate-400 flex items-center justify-between">
              <span>
                <strong>Mode State:</strong> Exploiting local neighborhood via multi-factor candidate scoring.
              </span>
              <span className="text-slate-500 font-mono">
                {5 - (decision?.stagnation_steps ?? walker.stagnation_steps ?? 0)} steps until dispersion
              </span>
            </div>
          )}

          {/* Selected Candidate Explanation & Multi-Criteria Breakdown */}
          {decision && decision.candidate_scores.length > 0 && decision.selected_node !== null && decision.selected_node !== undefined && (
            (() => {
              const selectedCand = decision.candidate_scores.find(
                (c) => String(c.node_id) === String(decision.selected_node)
              );
              if (!selectedCand) return null;

              // Find top alternative by score
              const alternatives = decision.candidate_scores
                .filter((c) => String(c.node_id) !== String(decision.selected_node))
                .sort((a, b) => b.total_score - a.total_score);
              const topAlt = alternatives[0];
              const scoreDiff = topAlt ? selectedCand.total_score - topAlt.total_score : 0;

              return (
                <div className="bg-graphite-950 rounded border border-cyan-900/50 p-2.5 space-y-2">
                  <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5">
                    <span className="text-[11px] font-bold text-cyan-300 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block" />
                      Selected Destination: Node {selectedCand.node_id}
                    </span>
                    <div className="flex items-center gap-2 font-mono text-[10px]">
                      <span className="text-slate-400">Score S(u): <strong className="text-slate-100">{selectedCand.total_score.toFixed(3)}</strong></span>
                      <span className="px-1.5 py-0.5 rounded bg-cyan-950 border border-cyan-800 text-cyan-300 font-bold">
                        P={(selectedCand.probability).toFixed(3)}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-4 gap-1.5 text-[10px] font-mono text-center">
                    <div className="bg-slate-900/60 p-1 rounded border border-slate-800" title="Positive evidence strength decayed by churn rate λ̂">
                      <span className="text-slate-500 block text-[9px]">+Evidence P(u)</span>
                      <span className="text-emerald-400 font-bold">{selectedCand.positive_evidence.toFixed(3)}</span>
                    </div>
                    <div className="bg-slate-900/60 p-1 rounded border border-slate-800" title="Novelty value inversely proportional to visit count: 1 / (1 + visits)">
                      <span className="text-slate-500 block text-[9px]">+Novelty N(u)</span>
                      <span className="text-cyan-400 font-bold">{selectedCand.novelty.toFixed(3)}</span>
                    </div>
                    <div className="bg-slate-900/60 p-1 rounded border border-slate-800" title="Local revisit count penalty subtracted from score">
                      <span className="text-slate-500 block text-[9px]">-Revisit R(u)</span>
                      <span className="text-amber-400 font-bold">-{selectedCand.revisit_penalty.toFixed(1)}</span>
                    </div>
                    <div className="bg-slate-900/60 p-1 rounded border border-slate-800" title="Observable edge traversal delay subtracted from score">
                      <span className="text-slate-500 block text-[9px]">-Delay D(u)</span>
                      <span className="text-slate-300 font-bold">-{selectedCand.delay.toFixed(1)}</span>
                    </div>
                  </div>

                  {topAlt && (
                    <div className="text-[10px] text-slate-400 pt-1 flex items-center justify-between border-t border-slate-800/60">
                      <span>
                        Outranked nearest alternative (<strong>Node {topAlt.node_id}</strong>, S={topAlt.total_score.toFixed(3)}) by{' '}
                        <strong className="text-cyan-300">
                          {scoreDiff >= 0 ? `+${scoreDiff.toFixed(3)}` : scoreDiff.toFixed(3)}
                        </strong>
                      </span>
                      <span className="text-slate-500 text-[9px]">
                        Alt prob: {(topAlt.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
              );
            })()
          )}

          {/* Candidate Neighbor Ranking Table */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] text-slate-300 font-semibold flex items-center gap-1.5">
                <Table className="w-3.5 h-3.5 text-cyan-400" />
                Active Neighbor Candidate Ranking
              </span>
              <span className="text-[10px] text-slate-500 font-mono" title="Scoring weights: α=1.0, β=0.5, γ=0.5, δ=0.2 with exploration temperature T=1.0. Unvisited candidates without positive evidence legitimately have identical scores.">
                S(u) = 1.0·P(u) + 0.5·N(u) - 0.5·R(u) - 0.2·D(u)
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
                      <th className="py-1 px-2 text-right" title="Decayed positive evidence strength in [0.0, 1.0]">Evidence P(u)</th>
                      <th className="py-1 px-2 text-right" title="Novelty value 1 / (1 + visits) in (0.0, 1.0]">Novelty N(u)</th>
                      <th className="py-1 px-2 text-right" title="Local visit count frequency">Revisit R(u)</th>
                      <th className="py-1 px-2 text-right" title="Traversal duration cost (base=1)">Delay D(u)</th>
                      <th className="py-1 px-2 text-right" title="Composite score S(u)">Score S(u)</th>
                      <th className="py-1 px-2 text-right" title="Softmax selection probability with temperature T=1.0">Prob P_select</th>
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
                            {isChosen && (
                              <span className="text-[9px] font-sans px-1 rounded bg-cyan-900/80 text-cyan-200">
                                SELECTED
                              </span>
                            )}
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
                            <span className={`px-1.5 py-0.5 rounded border text-[10px] ${
                              isChosen
                                ? 'bg-cyan-950 border-cyan-700 text-cyan-300 font-bold'
                                : 'bg-slate-900 border-slate-700 text-slate-300'
                            }`}>
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
