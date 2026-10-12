import React, { useState } from 'react';
import {
  EvidenceItemDto,
  ChurnItemDto,
  WalkerStateDto,
  TrajectoryDto,
} from '../lib/api';
import {
  Database,
  TrendingUp,
  Footprints,
  Compass,
  Radio,
  ArrowRight,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';

interface AnalysisPanelsProps {
  evidence: EvidenceItemDto[];
  churn: ChurnItemDto[];
  walkers: WalkerStateDto[];
  trajectory: TrajectoryDto | null;
  targetVisible: boolean;
  isAesw: boolean;
}

export const AnalysisPanels: React.FC<AnalysisPanelsProps> = ({
  evidence,
  churn,
  walkers,
  trajectory,
  targetVisible,
  isAesw,
}) => {
  const [activeTab, setActiveTab] = useState<'evidence' | 'churn' | 'activity' | 'trajectories'>(
    'evidence'
  );

  return (
    <div className="bg-graphite-900 border border-slate-800 rounded-lg p-3 flex flex-col h-full text-xs font-mono">
      {/* Tab Navigation */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-3">
        <div className="flex items-center space-x-1.5">
          <button
            type="button"
            onClick={() => setActiveTab('evidence')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded transition-colors ${
              activeTab === 'evidence'
                ? 'bg-cyan-950 border border-cyan-800 text-cyan-300 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Evidence Memory ({evidence.length})</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('churn')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded transition-colors ${
              activeTab === 'churn'
                ? 'bg-cyan-950 border border-cyan-800 text-cyan-300 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5" />
            <span>Churn Volatility ({churn.length})</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('activity')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded transition-colors ${
              activeTab === 'activity'
                ? 'bg-cyan-950 border border-cyan-800 text-cyan-300 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Footprints className="w-3.5 h-3.5" />
            <span>Walker Activity ({walkers.length})</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('trajectories')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded transition-colors ${
              activeTab === 'trajectories'
                ? 'bg-cyan-950 border border-cyan-800 text-cyan-300 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Trajectory History</span>
          </button>
        </div>

        <span className="text-[10px] text-slate-500 uppercase">
          Empirical Analytics Subsystem
        </span>
      </div>

      {/* Tab 1: Evidence Memory */}
      {activeTab === 'evidence' && (
        <div className="flex-1 overflow-y-auto">
          {!isAesw ? (
            <div className="text-slate-500 text-center py-8 italic">
              Evidence memory caching is an AESW-specific mechanism. Selected baseline algorithm does not store evidence records.
            </div>
          ) : evidence.length === 0 ? (
            <div className="text-slate-500 text-center py-8 italic">
              No evidence records stored in walker memory yet. Advance steps to observe local clue acquisition.
            </div>
          ) : (
            <div className="border border-slate-800 rounded overflow-hidden">
              <table className="w-full text-left text-[11px]">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 text-[10px]">
                  <tr>
                    <th className="py-1.5 px-2.5">Node</th>
                    <th className="py-1.5 px-2.5">Walker</th>
                    <th className="py-1.5 px-2.5">Polarity</th>
                    <th className="py-1.5 px-2.5 text-right">Base Conf</th>
                    <th className="py-1.5 px-2.5 text-right">Weight w(t)</th>
                    <th className="py-1.5 px-2.5 text-right">Effective Conf</th>
                    <th className="py-1.5 px-2.5 text-right">Age</th>
                    <th className="py-1.5 px-2.5">Source</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {evidence.map((item, idx) => (
                    <tr key={`ev-row-${idx}`} className="hover:bg-slate-900/40">
                      <td className="py-1.5 px-2.5 font-bold text-slate-200">
                        Node {item.node_id}
                      </td>
                      <td className="py-1.5 px-2.5 text-cyan-400">
                        Walker #{item.walker_id}
                      </td>
                      <td className="py-1.5 px-2.5">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] ${
                            item.polarity === 'POSITIVE'
                              ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/60'
                              : 'bg-slate-900 text-slate-400 border border-slate-800'
                          }`}
                        >
                          {item.polarity}
                        </span>
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-slate-300">
                        {item.confidence.toFixed(2)}
                      </td>
                      <td className="py-1.5 px-2.5 text-right font-bold text-cyan-300">
                        {item.effective_weight.toFixed(3)}
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-emerald-300">
                        {item.effective_confidence.toFixed(3)}
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-slate-400">
                        {item.age}
                      </td>
                      <td className="py-1.5 px-2.5 text-slate-500 text-[10px]">
                        {item.source}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Churn Volatility */}
      {activeTab === 'churn' && (
        <div className="flex-1 flex flex-col space-y-3 overflow-y-auto">
          {!isAesw ? (
            <div className="text-slate-500 text-center py-8 italic">
              Dynamic churn estimation is an AESW-specific component. Baseline algorithms do not infer graph turnover.
            </div>
          ) : churn.length === 0 ? (
            <div className="text-slate-500 text-center py-8 italic">
              No churn estimates recorded yet. Advance steps to track online neighborhood turnover.
            </div>
          ) : (
            <>
              {/* Walker Churn Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {churn.map((item) => (
                  <div
                    key={`churn-card-${item.walker_id}`}
                    className="bg-graphite-950 p-2.5 rounded border border-slate-800 text-[11px] space-y-1"
                  >
                    <div className="flex justify-between items-center text-slate-400">
                      <span>Walker #{item.walker_id}</span>
                      <span className="text-[10px] text-slate-500">{item.update_count} obs</span>
                    </div>
                    <div className="flex justify-between items-baseline">
                      <span className="text-slate-500 text-[10px]">λ̂ Smoothed:</span>
                      <span className="text-cyan-300 font-bold text-sm">
                        {item.estimated_lambda.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline text-[10px]">
                      <span className="text-slate-500">Jaccard Overlap s:</span>
                      <span className="text-slate-300">{item.overlap.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between items-baseline text-[10px]">
                      <span className="text-slate-500">Raw Churn Rate:</span>
                      <span className="text-slate-300">{item.raw_churn_rate.toFixed(3)}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Churn History Line Chart */}
              {churn.some((c) => c.history.length > 0) && (
                <div className="bg-graphite-950 p-3 rounded border border-slate-800 flex-1 flex flex-col min-h-[180px]">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-slate-300 font-medium text-[11px]">
                      Estimated Churn Volatility (λ̂_t) over Time Steps
                    </span>
                    <span className="text-[10px] text-slate-500">
                      λ̂_t = (1-η)λ̂_(t-1) + η·(-ln(s)/Δt)
                    </span>
                  </div>
                  <div className="flex-1 w-full min-h-[140px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={churn[0].history}
                        margin={{ top: 5, right: 10, left: -20, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis
                          dataKey="time"
                          stroke="#64748b"
                          tick={{ fill: '#64748b', fontSize: 10 }}
                        />
                        <YAxis
                          stroke="#64748b"
                          tick={{ fill: '#64748b', fontSize: 10 }}
                          domain={[0, 'auto']}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#0b0f17',
                            borderColor: '#334155',
                            fontSize: '11px',
                            fontFamily: 'monospace',
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="lambda_hat"
                          name="Walker 0 λ̂"
                          stroke="#06b6d4"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Tab 3: Walker Activity */}
      {activeTab === 'activity' && (
        <div className="flex-1 overflow-y-auto">
          {walkers.length === 0 ? (
            <div className="text-slate-500 text-center py-8 italic">
              No walkers instantiated yet.
            </div>
          ) : (
            <div className="border border-slate-800 rounded overflow-hidden">
              <table className="w-full text-left text-[11px]">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 text-[10px]">
                  <tr>
                    <th className="py-1.5 px-2.5">Walker ID</th>
                    <th className="py-1.5 px-2.5">Current Node</th>
                    <th className="py-1.5 px-2.5">Mode</th>
                    <th className="py-1.5 px-2.5 text-right">Step Count</th>
                    <th className="py-1.5 px-2.5 text-right">Messages</th>
                    <th className="py-1.5 px-2.5 text-right">Unique Visits</th>
                    <th className="py-1.5 px-2.5">Recent Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {walkers.map((w) => (
                    <tr key={`w-row-${w.walker_id}`} className="hover:bg-slate-900/40">
                      <td className="py-1.5 px-2.5 font-bold text-cyan-400">
                        Walker #{w.walker_id}
                      </td>
                      <td className="py-1.5 px-2.5 font-bold text-slate-200">
                        Node {w.current_node}
                      </td>
                      <td className="py-1.5 px-2.5">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] ${
                            w.mode === 'LONG_JUMP'
                              ? 'bg-amber-950/60 text-amber-300 border border-amber-700/60'
                              : 'bg-cyan-950/60 text-cyan-300 border border-cyan-700/60'
                          }`}
                        >
                          {w.mode || 'BASELINE'}
                        </span>
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-slate-200 font-bold">
                        {w.step_count}
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-teal-300">
                        {w.message_count}
                      </td>
                      <td className="py-1.5 px-2.5 text-right text-slate-300">
                        {Object.keys(w.visit_counts || {}).length}
                      </td>
                      <td className="py-1.5 px-2.5 text-slate-400">
                        {w.recent_action ? (
                          <span className="flex items-center gap-1">
                            <span>{w.recent_action}</span>
                            {w.recent_destination !== null && w.recent_destination !== undefined && (
                              <>
                                <ArrowRight className="w-3 h-3 text-slate-600" />
                                <span>{String(w.recent_destination)}</span>
                              </>
                            )}
                          </span>
                        ) : (
                          'N/A'
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Tab 4: Trajectories */}
      {activeTab === 'trajectories' && (
        <div className="flex-1 overflow-y-auto space-y-3">
          {!trajectory ? (
            <div className="text-slate-500 text-center py-8 italic">
              No trajectory history available.
            </div>
          ) : (
            <div className="space-y-2.5">
              {/* Target Trajectory (Ground Truth) */}
              {targetVisible && (
                <div className="bg-graphite-950 p-2.5 rounded border border-amber-900/40">
                  <div className="flex items-center justify-between text-amber-400 font-semibold mb-1">
                    <span className="flex items-center gap-1.5">
                      <Radio className="w-3.5 h-3.5 text-amber-400" />
                      Target Trajectory (Ground Truth View)
                    </span>
                    <span className="text-[10px] text-amber-500/80">
                      {trajectory.target_trajectory.length} steps
                    </span>
                  </div>
                  <div className="text-slate-300 text-[11px] overflow-x-auto whitespace-nowrap py-1">
                    {trajectory.target_trajectory.map((pos, idx) => (
                      <span key={`t-traj-${idx}`}>
                        <span className="font-bold text-amber-300">{pos}</span>
                        {idx < trajectory.target_trajectory.length - 1 && (
                          <span className="text-slate-600 mx-1">→</span>
                        )}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Walker Trajectories */}
              {Object.entries(trajectory.walker_trajectories).map(([wId, traj]) => (
                <div
                  key={`w-traj-${wId}`}
                  className="bg-graphite-950 p-2.5 rounded border border-slate-800"
                >
                  <div className="flex items-center justify-between text-cyan-400 font-semibold mb-1">
                    <span className="flex items-center gap-1.5">
                      <Footprints className="w-3.5 h-3.5 text-cyan-400" />
                      Walker #{wId} Trajectory
                    </span>
                    <span className="text-[10px] text-slate-500">{traj.length} steps</span>
                  </div>
                  <div className="text-slate-300 text-[11px] overflow-x-auto whitespace-nowrap py-1">
                    {traj.map((pos, idx) => (
                      <span key={`traj-${wId}-${idx}`}>
                        <span className="font-bold text-slate-200">{pos}</span>
                        {idx < traj.length - 1 && (
                          <span className="text-slate-600 mx-1">→</span>
                        )}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
