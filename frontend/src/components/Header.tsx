import { Activity, Wifi, WifiOff, Clock, Database, Compass } from 'lucide-react';
import { SimulationStatusResponse } from '../lib/api';

interface HeaderProps {
  backendOnline: boolean;
  simulation: SimulationStatusResponse | null;
  autoStepping: boolean;
}

export const Header: React.FC<HeaderProps> = ({ backendOnline, simulation, autoStepping }) => {
  return (
    <header className="border-b border-slate-800 bg-graphite-900/90 backdrop-blur px-5 py-3 sticky top-0 z-40">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        {/* Brand identity */}
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400 font-bold shadow-sm shadow-cyan-950">
            <Compass className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base font-bold tracking-tight text-slate-100 uppercase">
                AESW <span className="text-cyan-400 font-semibold text-sm">Workstation</span>
              </h1>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                v0.1.0
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono flex items-center gap-1.5">
              <span>Adaptive Evidence-Sharing Walkers</span>
              <span className="text-slate-600">•</span>
              <span className="text-cyan-400/80">Dynamic Search Under Uncertainty</span>
            </p>
          </div>
        </div>

        {/* Status Indicators & Metadata */}
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          {/* Backend Connection */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded border ${
              backendOnline
                ? 'bg-emerald-950/30 border-emerald-800/50 text-emerald-300'
                : 'bg-rose-950/30 border-rose-800/50 text-rose-300'
            }`}
          >
            {backendOnline ? (
              <>
                <Wifi className="w-3.5 h-3.5 text-emerald-400" />
                <span>API ONLINE</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-rose-400" />
                <span>API OFFLINE</span>
              </>
            )}
          </div>

          {/* Current Simulation Session */}
          {simulation ? (
            <>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800/80 border border-slate-700 text-slate-300">
                <Database className="w-3.5 h-3.5 text-cyan-400" />
                <span>ID: <strong className="text-cyan-300">{simulation.simulation_id}</strong></span>
              </div>

              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800/80 border border-slate-700 text-slate-300">
                <Clock className="w-3.5 h-3.5 text-amber-400" />
                <span>
                  STEP: <strong className="text-slate-100">{simulation.time}</strong> / {simulation.time_horizon}
                </span>
              </div>

              {/* Status Badge */}
              <div
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded border font-semibold ${
                  simulation.termination_status === 'SUCCESS'
                    ? 'bg-emerald-950/50 border-emerald-600 text-emerald-300'
                    : simulation.termination_status === 'BUDGET_EXHAUSTED'
                    ? 'bg-amber-950/50 border-amber-600 text-amber-300'
                    : simulation.termination_status === 'DISCONNECTED'
                    ? 'bg-rose-950/50 border-rose-600 text-rose-300'
                    : autoStepping
                    ? 'bg-cyan-950/50 border-cyan-600 text-cyan-300 animate-pulse'
                    : 'bg-slate-800 border-slate-700 text-slate-300'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span>
                  {simulation.termination_status
                    ? simulation.termination_status
                    : autoStepping
                    ? 'RUNNING'
                    : simulation.time > 0
                    ? 'PAUSED'
                    : 'READY'}
                </span>
              </div>
            </>
          ) : (
            <div className="px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-500">
              NO ACTIVE SESSION
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
