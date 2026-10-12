import React from 'react';
import { GraphNodeDto, EvidenceItemDto } from '../lib/api';
import { X, MapPin, Footprints, Shield, Radio } from 'lucide-react';

interface NodeDetailModalProps {
  node: GraphNodeDto | null;
  evidence: EvidenceItemDto[];
  targetVisible: boolean;
  onClose: () => void;
}

export const NodeDetailModal: React.FC<NodeDetailModalProps> = ({
  node,
  evidence,
  targetVisible,
  onClose,
}) => {
  if (!node) return null;

  // Find evidence associated with this node across all walkers
  const nodeEvidence = evidence.filter((e) => String(e.node_id) === String(node.id));

  return (
    <div className="absolute top-12 right-4 w-72 bg-graphite-900/95 border border-slate-700/80 rounded-lg shadow-xl backdrop-blur p-3.5 text-xs font-mono z-30 animate-in fade-in slide-in-from-top-2 duration-150">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-3">
        <div className="flex items-center space-x-2">
          <MapPin className="w-4 h-4 text-cyan-400" />
          <h3 className="font-bold text-slate-100 text-sm">
            Node {node.label}
          </h3>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200"
          title="Close details"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="space-y-2 text-slate-300">
        <div className="flex justify-between items-center py-1 border-b border-slate-800/60">
          <span className="text-slate-400">Static Degree:</span>
          <span className="font-bold text-slate-100">{node.degree}</span>
        </div>

        <div className="flex justify-between items-center py-1 border-b border-slate-800/60">
          <span className="text-slate-400">Search Status:</span>
          <span className={node.is_visited ? 'text-cyan-400 font-semibold' : 'text-slate-500'}>
            {node.is_visited ? 'VISITED' : 'UNVISITED'}
          </span>
        </div>

        {/* Walkers Present */}
        <div className="flex justify-between items-center py-1 border-b border-slate-800/60">
          <span className="text-slate-400 flex items-center gap-1">
            <Footprints className="w-3.5 h-3.5 text-cyan-400" />
            Walkers:
          </span>
          <span>
            {node.walker_ids.length > 0 ? (
              <span className="text-emerald-400 font-bold">
                {node.walker_ids.map((w: number) => `#${w}`).join(', ')}
              </span>
            ) : (
              <span className="text-slate-500">None</span>
            )}
          </span>
        </div>

        {/* Target Ground Truth */}
        {targetVisible && (
          <div className="flex justify-between items-center py-1 border-b border-slate-800/60">
            <span className="text-slate-400 flex items-center gap-1">
              <Radio className="w-3.5 h-3.5 text-amber-400" />
              Target Location:
            </span>
            <span>
              {node.is_target ? (
                <span className="text-amber-400 font-bold bg-amber-950/60 px-1.5 py-0.5 rounded border border-amber-700/60">
                  TARGET PRESENT
                </span>
              ) : (
                <span className="text-slate-500">Absent</span>
              )}
            </span>
          </div>
        )}

        {/* Local Evidence Records */}
        <div className="pt-2">
          <div className="flex items-center gap-1 text-slate-400 font-medium mb-1.5">
            <Shield className="w-3.5 h-3.5 text-teal-400" />
            <span>Stored Evidence ({nodeEvidence.length}):</span>
          </div>
          {nodeEvidence.length === 0 ? (
            <p className="text-slate-600 italic text-[11px]">No local evidence stored for this vertex.</p>
          ) : (
            <div className="space-y-1.5 max-h-32 overflow-y-auto pr-1">
              {nodeEvidence.map((ev, idx) => (
                <div
                  key={`ev-${idx}`}
                  className="bg-graphite-950 p-2 rounded border border-slate-800 text-[10px] space-y-0.5"
                >
                  <div className="flex justify-between text-slate-300">
                    <span>Walker #{ev.walker_id}:</span>
                    <span
                      className={`font-bold ${
                        ev.polarity === 'POSITIVE' ? 'text-emerald-400' : 'text-slate-400'
                      }`}
                    >
                      {ev.polarity}
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Decayed Weight w(t):</span>
                    <span className="text-cyan-300 font-mono">{ev.effective_weight.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between text-slate-500 text-[9px]">
                    <span>Age: {ev.age} steps</span>
                    <span>Src: {ev.source}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
