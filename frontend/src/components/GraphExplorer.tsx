import React, { useState, useRef } from 'react';
import { GraphResponse, GraphNodeDto } from '../lib/api';
import { ZoomIn, ZoomOut, Maximize2, Eye, EyeOff, Info } from 'lucide-react';

interface GraphExplorerProps {
  graph: GraphResponse | null;
  selectedNodeId: number | string | null;
  onSelectNode: (node: GraphNodeDto | null) => void;
  targetVisible: boolean;
  onToggleTargetVisible: (visible: boolean) => void;
}

const WALKER_COLORS = [
  '#06b6d4', // Cyan
  '#10b981', // Emerald
  '#a855f7', // Purple
  '#f43f5e', // Rose
  '#f59e0b', // Amber
  '#3b82f6', // Blue
];

export const GraphExplorer: React.FC<GraphExplorerProps> = ({
  graph,
  selectedNodeId,
  onSelectNode,
  targetVisible,
  onToggleTargetVisible,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleZoom = (factor: number) => {
    setZoom((prev) => Math.max(0.4, Math.min(3.5, prev * factor)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Node coordinate mapping helper
  const getNodeCoords = (node: GraphNodeDto, width: number, height: number) => {
    const scale = Math.min(width, height) * 0.44;
    const cx = width / 2 + node.x * scale;
    const cy = height / 2 + node.y * scale;
    return { cx, cy };
  };

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="bg-graphite-900 border border-slate-800 rounded-lg p-8 flex flex-col items-center justify-center h-full min-h-[420px] text-slate-500 font-mono text-xs">
        <Info className="w-8 h-8 text-slate-600 mb-2" />
        <p>No active graph topology loaded.</p>
        <p className="text-[11px] text-slate-600 mt-1">
          Configure parameters and click "Generate Simulation" to initialize the environment.
        </p>
      </div>
    );
  }

  const width = 800;
  const height = 560;

  // Build node coordinate lookup
  const nodeMap = new Map<number | string, GraphNodeDto>();
  const coordMap = new Map<number | string, { cx: number; cy: number }>();
  for (const n of graph.nodes) {
    nodeMap.set(n.id, n);
    coordMap.set(n.id, getNodeCoords(n, width, height));
  }

  return (
    <div className="bg-graphite-900 border border-slate-800 rounded-lg flex flex-col h-full overflow-hidden relative">
      {/* Explorer Header & Controls */}
      <div className="bg-graphite-950/80 border-b border-slate-800 px-3 py-2 flex items-center justify-between text-xs z-10">
        <div className="flex items-center space-x-3">
          <span className="font-semibold text-slate-200 uppercase tracking-wider text-[11px]">
            Topology Explorer
          </span>
          <span className="text-slate-500 font-mono text-[11px]">
            |V| = {graph.nodes.length} • |E_active| = {graph.active_edge_count} / {graph.total_edge_count}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          {/* Ground truth target toggle */}
          <button
            type="button"
            onClick={() => onToggleTargetVisible(!targetVisible)}
            className={`flex items-center space-x-1 px-2 py-1 rounded text-[11px] font-mono border ${
              targetVisible
                ? 'bg-amber-950/40 border-amber-700/60 text-amber-300'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
            title="Toggle ground-truth target visibility (for researcher debugging only, strictly invisible to agents)"
          >
            {targetVisible ? <Eye className="w-3.5 h-3.5 text-amber-400" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>Target Beacon {targetVisible ? 'ON' : 'OFF'}</span>
          </button>

          {/* Zoom & Pan Controls */}
          <div className="flex items-center bg-graphite-900 border border-slate-800 rounded p-0.5">
            <button
              type="button"
              onClick={() => handleZoom(1.2)}
              className="p-1 hover:bg-slate-800 rounded text-slate-300"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => handleZoom(0.8)}
              className="p-1 hover:bg-slate-800 rounded text-slate-300"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={resetView}
              className="p-1 hover:bg-slate-800 rounded text-slate-300"
              title="Reset View"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Interactive SVG Canvas */}
      <div
        className="flex-1 w-full h-full cursor-grab active:cursor-grabbing overflow-hidden bg-[#070a0f] relative select-none"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full"
          style={{ minHeight: '440px' }}
        >
          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
            {/* Edges */}
            <g className="edges">
              {graph.edges.map((edge, idx) => {
                const c1 = coordMap.get(edge.source);
                const c2 = coordMap.get(edge.target);
                if (!c1 || !c2) return null;

                const isActive = edge.is_active;
                const isTransition = edge.is_recent_transition;

                return (
                  <line
                    key={`edge-${edge.source}-${edge.target}-${idx}`}
                    x1={c1.cx}
                    y1={c1.cy}
                    x2={c2.cx}
                    y2={c2.cy}
                    stroke={
                      isTransition
                        ? '#f59e0b'
                        : isActive
                        ? '#0891b2'
                        : '#334155'
                    }
                    strokeWidth={isTransition ? 2.5 : isActive ? 1.5 : 1.0}
                    strokeDasharray={isActive ? undefined : '3,3'}
                    opacity={isActive ? 0.75 : 0.25}
                  />
                );
              })}
            </g>

            {/* Nodes */}
            <g className="nodes">
              {graph.nodes.map((node) => {
                const coords = coordMap.get(node.id);
                if (!coords) return null;

                const isSelected = selectedNodeId === node.id;
                const isTarget = node.is_target && targetVisible;
                const isVisited = node.is_visited;
                const hasWalkers = node.walker_ids.length > 0;
                const hasEvidence = node.has_evidence;

                return (
                  <g
                    key={`node-${node.id}`}
                    transform={`translate(${coords.cx}, ${coords.cy})`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectNode(isSelected ? null : node);
                    }}
                    className="cursor-pointer group"
                  >
                    {/* Target Pulsing Beacon (Ground Truth) */}
                    {isTarget && (
                      <>
                        <circle
                          r={24}
                          className="fill-amber-500/10 stroke-amber-400 stroke-1 animate-ping"
                        />
                        <circle
                          r={18}
                          className="fill-amber-500/20 stroke-amber-500 stroke-1"
                        />
                      </>
                    )}

                    {/* Selected Focus Ring */}
                    {isSelected && (
                      <circle
                        r={19}
                        className="fill-none stroke-cyan-400 stroke-2 stroke-dasharray-4"
                      />
                    )}

                    {/* Node Base Circle */}
                    <circle
                      r={13}
                      fill={
                        isVisited
                          ? isSelected
                            ? '#0891b2'
                            : '#164e63'
                          : '#0f172a'
                      }
                      stroke={
                        isSelected
                          ? '#22d3ee'
                          : isVisited
                          ? '#06b6d4'
                          : hasEvidence
                          ? '#38bdf8'
                          : '#334155'
                      }
                      strokeWidth={isSelected ? 2 : isVisited ? 1.5 : 1}
                      className="transition-colors group-hover:stroke-cyan-300"
                    />

                    {/* Target Central Pin */}
                    {isTarget && (
                      <circle r={4.5} fill="#f59e0b" className="animate-pulse" />
                    )}

                    {/* Node ID Label */}
                    <text
                      textAnchor="middle"
                      dy="0.35em"
                      className="text-[10px] font-mono fill-slate-200 pointer-events-none select-none font-bold"
                    >
                      {node.label}
                    </text>

                    {/* Walker Markers */}
                    {hasWalkers && (
                      <g transform="translate(0, -18)">
                        {node.walker_ids.map((wId, wIdx) => {
                          const offsetX = (wIdx - (node.walker_ids.length - 1) / 2) * 14;
                          const color = WALKER_COLORS[wId % WALKER_COLORS.length];
                          return (
                            <g key={`w-${wId}`} transform={`translate(${offsetX}, 0)`}>
                              <rect
                                x={-6}
                                y={-6}
                                width={12}
                                height={12}
                                rx={2.5}
                                fill={color}
                                stroke="#070a0f"
                                strokeWidth={1}
                              />
                              <text
                                textAnchor="middle"
                                dy="0.35em"
                                className="text-[8px] font-mono fill-slate-950 font-black pointer-events-none"
                              >
                                {wId}
                              </text>
                            </g>
                          );
                        })}
                      </g>
                    )}

                    {/* Evidence Marker Indicator */}
                    {hasEvidence && !hasWalkers && (
                      <circle
                        cx={8}
                        cy={-8}
                        r={3}
                        fill={node.evidence_polarity === 'POSITIVE' ? '#10b981' : '#64748b'}
                        stroke="#070a0f"
                        strokeWidth={1}
                      />
                    )}
                  </g>
                );
              })}
            </g>
          </g>
        </svg>
      </div>

      {/* Legend Bar */}
      <div className="bg-graphite-950/90 border-t border-slate-800 px-4 py-2 flex flex-wrap items-center justify-between gap-3 text-[11px] font-mono text-slate-400">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-4 h-0.5 bg-cyan-600 inline-block" />
            <span>Active Link (ON)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-4 h-0.5 border-t border-dashed border-slate-600 inline-block" />
            <span>Inactive Link (OFF)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-cyan-900 border border-cyan-500 inline-block" />
            <span>Visited Node</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-cyan-500 text-slate-950 font-bold flex items-center justify-center text-[8px]">
              0
            </span>
            <span>Walker Presence</span>
          </div>
          {targetVisible && (
            <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
              <span className="w-3 h-3 rounded-full bg-amber-500/30 border border-amber-400 inline-block" />
              <span>Target (Ground Truth)</span>
            </div>
          )}
        </div>

        <span className="text-[10px] text-slate-500">
          Click any vertex to inspect local evidence & degrees
        </span>
      </div>
    </div>
  );
};
