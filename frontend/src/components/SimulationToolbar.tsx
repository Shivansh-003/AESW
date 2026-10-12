import { Play, Pause, StepForward, RotateCcw, Gauge } from 'lucide-react';
import { SimulationStatusResponse } from '../lib/api';

interface SimulationToolbarProps {
  simulation: SimulationStatusResponse | null;
  autoStepping: boolean;
  stepSpeed: number;
  onStart: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  disabled: boolean;
  requestInFlight: boolean;
}

export const SimulationToolbar: React.FC<SimulationToolbarProps> = ({
  simulation,
  autoStepping,
  stepSpeed,
  onStart,
  onPause,
  onStep,
  onReset,
  onSpeedChange,
  disabled,
  requestInFlight,
}) => {
  const isTerminated = simulation?.termination_status !== null && simulation?.termination_status !== undefined;

  return (
    <div className="bg-graphite-900 border border-slate-800 rounded-lg p-2.5 flex flex-wrap items-center justify-between gap-3 text-xs">
      {/* Control Buttons */}
      <div className="flex items-center space-x-2">
        {/* Start / Auto-Step */}
        {!autoStepping ? (
          <button
            type="button"
            onClick={onStart}
            disabled={disabled || !simulation || isTerminated || requestInFlight}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-slate-950 font-bold transition-colors"
            title="Start automatic stepping"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>START</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={onPause}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold transition-colors"
            title="Pause automatic stepping"
          >
            <Pause className="w-4 h-4 fill-current" />
            <span>PAUSE</span>
          </button>
        )}

        {/* Step Forward (Manual Single Advance) */}
        <button
          type="button"
          onClick={onStep}
          disabled={disabled || !simulation || isTerminated || autoStepping || requestInFlight}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-cyan-950 hover:bg-cyan-900 border border-cyan-800 text-cyan-300 disabled:border-slate-800 disabled:bg-slate-900 disabled:text-slate-600 font-semibold transition-colors"
          title="Advance single discrete step"
        >
          <StepForward className="w-4 h-4" />
          <span>STEP (1)</span>
        </button>

        {/* Reset */}
        <button
          type="button"
          onClick={onReset}
          disabled={disabled || !simulation || requestInFlight}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:bg-slate-900 disabled:text-slate-600 font-medium transition-colors"
          title="Reset simulation to deterministic t=0"
        >
          <RotateCcw className="w-4 h-4" />
          <span>RESET</span>
        </button>
      </div>

      {/* Stepping Speed Selector */}
      <div className="flex items-center space-x-3 text-slate-400 font-mono text-[11px]">
        <div className="flex items-center space-x-1.5">
          <Gauge className="w-3.5 h-3.5 text-cyan-400" />
          <span>SPEED:</span>
        </div>
        <div className="flex items-center bg-graphite-950 p-0.5 rounded border border-slate-800">
          <button
            type="button"
            onClick={() => onSpeedChange(800)}
            className={`px-2 py-0.5 rounded text-[10px] ${
              stepSpeed === 800 ? 'bg-cyan-600 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            SLOW
          </button>
          <button
            type="button"
            onClick={() => onSpeedChange(350)}
            className={`px-2 py-0.5 rounded text-[10px] ${
              stepSpeed === 350 ? 'bg-cyan-600 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            NORMAL
          </button>
          <button
            type="button"
            onClick={() => onSpeedChange(120)}
            className={`px-2 py-0.5 rounded text-[10px] ${
              stepSpeed === 120 ? 'bg-cyan-600 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            FAST
          </button>
        </div>
      </div>
    </div>
  );
};
