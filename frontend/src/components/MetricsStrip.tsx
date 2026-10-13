import React from 'react';
import { MetricsDto } from '../lib/api';
import { CheckCircle2, AlertTriangle, XCircle, Footprints, RotateCcw, MapPin, MessageSquare, DollarSign } from 'lucide-react';

interface MetricsStripProps {
  metrics: MetricsDto | null;
  totalNodes: number;
  isAesw?: boolean;
  algorithm?: string;
}

export const MetricsStrip: React.FC<MetricsStripProps> = ({
  metrics,
  totalNodes,
  isAesw,
  algorithm,
}) => {
  if (!metrics) {
    return (
      <div className="bg-graphite-900 border-b border-slate-800 px-5 py-2.5 text-xs text-slate-500 font-mono">
        Waiting for simulation initialization to stream metrics...
      </div>
    );
  }

  const coveragePct = totalNodes > 0 ? ((metrics.nodes_visited / totalNodes) * 100).toFixed(1) : '0.0';

  return (
    <div className="bg-graphite-900/95 border-b border-slate-800 px-5 py-2.5 overflow-x-auto">
      <div className="flex items-center gap-4 min-w-max text-xs font-mono">
        {/* Status */}
        <div className="flex items-center gap-1.5 pr-3 border-r border-slate-800">
          <span className="text-slate-400 uppercase text-[11px]">Outcome:</span>
          {metrics.success ? (
            <span className="flex items-center gap-1 text-emerald-400 font-semibold bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/60">
              <CheckCircle2 className="w-3.5 h-3.5" /> ACQUIRED (SUCCESS)
            </span>
          ) : metrics.status === 'BUDGET_EXHAUSTED' ? (
            <span className="flex items-center gap-1 text-amber-400 font-semibold bg-amber-950/40 px-2 py-0.5 rounded border border-amber-800/60">
              <AlertTriangle className="w-3.5 h-3.5" /> BUDGET EXHAUSTED
            </span>
          ) : metrics.status === 'DISCONNECTED' ? (
            <span className="flex items-center gap-1 text-rose-400 font-semibold bg-rose-950/40 px-2 py-0.5 rounded border border-rose-800/60">
              <XCircle className="w-3.5 h-3.5" /> DISCONNECTED
            </span>
          ) : (
            <span className="text-cyan-400 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/60">
              SEARCHING...
            </span>
          )}
        </div>

        {/* Moves M */}
        <div className="flex items-center gap-1.5 pr-3 border-r border-slate-800">
          <Footprints className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">Moves (M):</span>
          <span className="text-slate-100 font-bold">{metrics.total_moves}</span>
        </div>

        {/* Revisits */}
        <div className="flex items-center gap-1.5 pr-3 border-r border-slate-800">
          <RotateCcw className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-slate-400">Revisits:</span>
          <span className="text-slate-100 font-bold">{metrics.node_revisits}</span>
        </div>

        {/* Unique Visited */}
        <div className="flex items-center gap-1.5 pr-3 border-r border-slate-800">
          <MapPin className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-400">Coverage:</span>
          <span className="text-slate-100 font-bold">
            {metrics.nodes_visited} / {totalNodes}
          </span>
          <span className="text-slate-500 text-[11px]">({coveragePct}%)</span>
        </div>

        {/* Messages Q */}
        {(() => {
          const effectiveAlgorithm = algorithm || metrics.algorithm;
          const isCommunicationFree = isAesw === false || (effectiveAlgorithm && effectiveAlgorithm.toLowerCase() !== 'aesw');
          const commTooltip = "Communication-free baseline — this algorithm does not exchange messages (Q = 0 by design).";

          return (
            <div
              className="flex items-center gap-1.5 pr-3 border-r border-slate-800"
              title={isCommunicationFree ? commTooltip : undefined}
            >
              <MessageSquare className="w-3.5 h-3.5 text-teal-400" />
              <span className="text-slate-400">Messages (Q):</span>
              <span className="text-slate-100 font-bold">{metrics.total_messages}</span>
              {isCommunicationFree && (
                <span
                  className="text-slate-400 text-[10px] bg-slate-800/80 px-1.5 py-0.2 rounded border border-slate-700/60 cursor-help"
                  title={commTooltip}
                >
                  No comm
                </span>
              )}
            </div>
          );
        })()}

        {/* Total Cost C = M + c_m Q */}
        <div className="flex items-center gap-1.5">
          <DollarSign className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">Total Cost (C):</span>
          <span className="text-cyan-300 font-bold text-sm bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/60">
            {metrics.total_cost.toFixed(2)}
          </span>
          <span className="text-slate-500 text-[10px]">
            (M + {metrics.c_m} × Q)
          </span>
        </div>
      </div>
    </div>
  );
};
