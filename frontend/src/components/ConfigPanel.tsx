import React, { useState } from 'react';
import { SimulationConfig } from '../lib/api';
import { Sliders, RefreshCw, Layers, ShieldAlert, Radio, Cpu, Network } from 'lucide-react';

interface ConfigPanelProps {
  onGenerate: (config: SimulationConfig) => void;
  loading: boolean;
  disabled: boolean;
}

interface NumericInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'> {
  value: number | undefined;
  onChange: (val: number) => void;
  isInteger?: boolean;
}

const NumericInput: React.FC<NumericInputProps> = ({
  value,
  onChange,
  isInteger = false,
  min,
  max,
  step,
  className,
  disabled,
  ...props
}) => {
  const [text, setText] = React.useState<string>(value !== undefined ? String(value) : '');

  // Synchronize internal text whenever external value changes
  React.useEffect(() => {
    if (value !== undefined) {
      setText((current) => {
        const parsed = isInteger ? parseInt(current, 10) : parseFloat(current);
        // Only update if current string doesn't match the incoming numeric value
        return isNaN(parsed) || parsed !== value ? String(value) : current;
      });
    }
  }, [value, isInteger]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setText(raw);

    // If empty or intermediate like "-" or ".", do not force parse
    if (raw === '' || raw === '-' || raw === '.' || raw === '-.') {
      return;
    }

    const parsed = isInteger ? parseInt(raw, 10) : parseFloat(raw);
    if (!isNaN(parsed)) {
      onChange(parsed);
    }
  };

  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    props.onBlur?.(e);
    const parsed = isInteger ? parseInt(text, 10) : parseFloat(text);
    if (isNaN(parsed)) {
      setText(value !== undefined ? String(value) : '');
    } else {
      let clamped = parsed;
      if (min !== undefined && clamped < Number(min)) clamped = Number(min);
      if (max !== undefined && clamped > Number(max)) clamped = Number(max);
      setText(String(clamped));
      onChange(clamped);
    }
  };

  return (
    <input
      type="number"
      value={text}
      onChange={handleChange}
      onBlur={handleBlur}
      min={min}
      max={max}
      step={step ?? (isInteger ? '1' : 'any')}
      disabled={disabled}
      className={className}
      {...props}
    />
  );
};

export const ConfigPanel: React.FC<ConfigPanelProps> = ({ onGenerate, loading, disabled }) => {
  const [config, setConfig] = useState<SimulationConfig>({
    algorithm: 'aesw',
    graph_type: 'er',
    num_nodes: 30,
    regime: 'MEDIUM',
    p_move: 0.25,
    target_mode: 'MOVING',
    signal_radius: 1,
    p_d: 0.9,
    p_fa: 0.05,
    neighbor_budget: 8,
    walker_count: 3,
    seed: 42,
    time_horizon: 100,
    communication_mode: 'PUSH_PULL',
    c_m: 0.1,
    graph_parameters: { p: 0.18 },
  });

  const [validationError, setValidationError] = useState<string | null>(null);

  const handleGraphTypeChange = (gt: string) => {
    let params: Record<string, any> = {};
    if (gt === 'er') params = { p: 0.18 };
    else if (gt === 'ba') params = { m: 2 };
    else if (gt === 'ws') params = { k: 4, p: 0.1 };
    else if (gt === 'grid') params = { rows: 6, cols: 6, obstacle_ratio: 0.1 };
    else if (gt === 'rgg') params = { radius: 0.28 };

    setConfig((prev) => ({
      ...prev,
      graph_type: gt,
      graph_parameters: params,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    if (config.num_nodes < 4 || config.num_nodes > 300) {
      setValidationError('Node count must be between 4 and 300.');
      return;
    }
    if (config.walker_count < 1 || config.walker_count > 15) {
      setValidationError('Walker count must be between 1 and 15.');
      return;
    }
    if (config.time_horizon < 5 || config.time_horizon > 1000) {
      setValidationError('Time horizon must be between 5 and 1000 steps.');
      return;
    }

    onGenerate(config);
  };

  const isAesw = config.algorithm === 'aesw';

  return (
    <div className="bg-graphite-900 border border-slate-800 rounded-lg p-4 flex flex-col h-full text-xs">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
        <div className="flex items-center space-x-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <h2 className="font-semibold text-slate-100 uppercase tracking-wider text-xs">
            Experiment Configuration
          </h2>
        </div>
        <span className="text-[10px] font-mono text-slate-500 uppercase">Parameter Space</span>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 flex-1 overflow-y-auto pr-1">
        {/* Algorithm Selection */}
        <div>
          <label className="block text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            Search Algorithm
          </label>
          <select
            value={config.algorithm}
            onChange={(e) => setConfig({ ...config, algorithm: e.target.value })}
            disabled={disabled}
            className="w-full bg-graphite-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
          >
            <option value="aesw">Adaptive Evidence-Sharing Walkers (AESW)</option>
            <option value="random_walk">Random Walk (Baseline)</option>
            <option value="non_backtracking">Non-Backtracking Walk (Baseline)</option>
            <option value="k_walk">k Independent Walkers (Baseline)</option>
            <option value="degree">Degree-Based Walk (Baseline)</option>
            <option value="flooding">Flooding / Frontier (Baseline)</option>
            <option value="ant_colony">Ant Colony Walk (Baseline)</option>
          </select>
        </div>

        {/* Graph Topology */}
        <div>
          <label className="block text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
            <Network className="w-3.5 h-3.5 text-cyan-400" />
            Graph Topology
          </label>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">Family</label>
              <select
                value={config.graph_type}
                onChange={(e) => handleGraphTypeChange(e.target.value)}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
              >
                <option value="er">Erdős–Rényi (ER)</option>
                <option value="ba">Barabási–Albert (BA)</option>
                <option value="ws">Watts–Strogatz (WS)</option>
                <option value="grid">2D Grid + Obstacles</option>
                <option value="rgg">Random Geometric (RGG)</option>
              </select>
            </div>
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">Node Count (|V|)</label>
              <NumericInput
                isInteger
                min={4}
                max={250}
                value={config.num_nodes}
                onChange={(val) => setConfig({ ...config, num_nodes: val })}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
              />
            </div>
          </div>

          {/* Topology Specific Param */}
          <div className="bg-graphite-950 p-2 rounded border border-slate-800/80 text-[11px] font-mono text-slate-400">
            {config.graph_type === 'er' && (
              <div className="flex items-center justify-between">
                <span>Edge Attachment Prob (p):</span>
                <NumericInput
                  step="0.01"
                  min="0.01"
                  max="1.0"
                  value={config.graph_parameters?.p ?? 0.18}
                  onChange={(val) =>
                    setConfig({
                      ...config,
                      graph_parameters: { ...config.graph_parameters, p: val },
                    })
                  }
                  disabled={disabled}
                  className="w-16 bg-slate-900 border border-slate-700 px-1 py-0.5 rounded text-right text-slate-200"
                />
              </div>
            )}
            {config.graph_type === 'ba' && (
              <div className="flex items-center justify-between">
                <span>Edges per Step (m):</span>
                <NumericInput
                  isInteger
                  min="1"
                  max="10"
                  value={config.graph_parameters?.m ?? 2}
                  onChange={(val) =>
                    setConfig({
                      ...config,
                      graph_parameters: { ...config.graph_parameters, m: val },
                    })
                  }
                  disabled={disabled}
                  className="w-16 bg-slate-900 border border-slate-700 px-1 py-0.5 rounded text-right text-slate-200"
                />
              </div>
            )}
            {config.graph_type === 'ws' && (
              <div className="flex items-center justify-between">
                <span>Rewiring Prob (p):</span>
                <NumericInput
                  step="0.05"
                  min="0.0"
                  max="1.0"
                  value={config.graph_parameters?.p ?? 0.1}
                  onChange={(val) =>
                    setConfig({
                      ...config,
                      graph_parameters: { ...config.graph_parameters, p: val },
                    })
                  }
                  disabled={disabled}
                  className="w-16 bg-slate-900 border border-slate-700 px-1 py-0.5 rounded text-right text-slate-200"
                />
              </div>
            )}
            {config.graph_type === 'grid' && (
              <div className="flex items-center justify-between">
                <span>Obstacle Ratio:</span>
                <NumericInput
                  step="0.05"
                  min="0.0"
                  max="0.4"
                  value={config.graph_parameters?.obstacle_ratio ?? 0.1}
                  onChange={(val) =>
                    setConfig({
                      ...config,
                      graph_parameters: { ...config.graph_parameters, obstacle_ratio: val },
                    })
                  }
                  disabled={disabled}
                  className="w-16 bg-slate-900 border border-slate-700 px-1 py-0.5 rounded text-right text-slate-200"
                />
              </div>
            )}
            {config.graph_type === 'rgg' && (
              <div className="flex items-center justify-between">
                <span>Geometric Radius (r):</span>
                <NumericInput
                  step="0.02"
                  min="0.05"
                  max="1.0"
                  value={config.graph_parameters?.radius ?? 0.28}
                  onChange={(val) =>
                    setConfig({
                      ...config,
                      graph_parameters: { ...config.graph_parameters, radius: val },
                    })
                  }
                  disabled={disabled}
                  className="w-16 bg-slate-900 border border-slate-700 px-1 py-0.5 rounded text-right text-slate-200"
                />
              </div>
            )}
          </div>
        </div>

        {/* Dynamic Churn Regime */}
        <div>
          <label className="block text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-amber-400" />
            Dynamic Edge Churn Regime
          </label>
          <select
            value={config.regime}
            onChange={(e) => setConfig({ ...config, regime: e.target.value })}
            disabled={disabled}
            className="w-full bg-graphite-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
          >
            <option value="STATIC">STATIC (p_on=0, p_off=0)</option>
            <option value="SLOW">SLOW (p_on=0.01, p_off=0.02)</option>
            <option value="MEDIUM">MEDIUM (p_on=0.05, p_off=0.10)</option>
            <option value="FAST">FAST (p_on=0.15, p_off=0.25)</option>
            <option value="VERY_FAST">VERY_FAST (p_on=0.30, p_off=0.40)</option>
          </select>
        </div>

        {/* Target Locomotion & Sensing */}
        <div className="border-t border-slate-800/80 pt-3">
          <label className="block text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
            <Radio className="w-3.5 h-3.5 text-cyan-400" />
            Target & Sensor Uncertainty
          </label>
          <div className="grid grid-cols-2 gap-2 mb-2 font-mono">
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">Target Mode</label>
              <select
                value={config.target_mode}
                onChange={(e) => setConfig({ ...config, target_mode: e.target.value })}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
              >
                <option value="MOVING">MOVING</option>
                <option value="STATIC">STATIC</option>
              </select>
            </div>
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">p_move</label>
              <NumericInput
                step="0.05"
                min="0.0"
                max="1.0"
                value={config.p_move}
                onChange={(val) => setConfig({ ...config, p_move: val })}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
              />
            </div>
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">Detection (p_d)</label>
              <NumericInput
                step="0.01"
                min="0.0"
                max="1.0"
                value={config.p_d}
                onChange={(val) => setConfig({ ...config, p_d: val })}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
              />
            </div>
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">False Alarm (p_fa)</label>
              <NumericInput
                step="0.01"
                min="0.0"
                max="0.5"
                value={config.p_fa}
                onChange={(val) => setConfig({ ...config, p_fa: val })}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
              />
            </div>
          </div>
        </div>

        {/* Walkers and Search Budget */}
        <div className="border-t border-slate-800/80 pt-3">
          <label className="block text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            Search Team & Limits
          </label>
          <div className="grid grid-cols-3 gap-2 font-mono">
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">Walkers (k)</label>
              <NumericInput
                isInteger
                min="1"
                max="10"
                value={config.walker_count}
                onChange={(val) => setConfig({ ...config, walker_count: val })}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
              />
            </div>
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">Budget (B)</label>
              <NumericInput
                isInteger
                min="1"
                max="25"
                value={config.neighbor_budget}
                onChange={(val) => setConfig({ ...config, neighbor_budget: val })}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
              />
            </div>
            <div>
              <label className="text-[11px] text-slate-400 block mb-0.5">Max Steps</label>
              <NumericInput
                isInteger
                min="10"
                max="500"
                value={config.time_horizon}
                onChange={(val) => setConfig({ ...config, time_horizon: val })}
                disabled={disabled}
                className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
              />
            </div>
          </div>
        </div>

        {/* AESW Novelty Parameters (when AESW is selected) */}
        {isAesw && (
          <div className="border-t border-slate-800/80 pt-3 bg-cyan-950/20 p-2.5 rounded border border-cyan-900/40">
            <label className="block text-cyan-300 font-medium mb-1.5 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              AESW Specific Settings
            </label>
            <div className="grid grid-cols-2 gap-2 font-mono">
              <div>
                <label className="text-[11px] text-slate-400 block mb-0.5">Communication</label>
                <select
                  value={config.communication_mode}
                  onChange={(e) => setConfig({ ...config, communication_mode: e.target.value })}
                  disabled={disabled}
                  className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
                >
                  <option value="PUSH_PULL">PUSH_PULL</option>
                  <option value="PUSH">PUSH ONLY</option>
                  <option value="PULL">PULL ONLY</option>
                  <option value="NO_SHARING">NO SHARING</option>
                </select>
              </div>
              <div>
                <label className="text-[11px] text-slate-400 block mb-0.5">Msg Cost (c_m)</label>
                <NumericInput
                  step="0.05"
                  min="0.0"
                  max="1.0"
                  value={config.c_m}
                  onChange={(val) => setConfig({ ...config, c_m: val })}
                  disabled={disabled}
                  className="w-full bg-graphite-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
                />
              </div>
            </div>
          </div>
        )}

        {/* Seed Input */}
        <div className="flex items-center justify-between text-slate-400 font-mono text-[11px] pt-1">
          <span>PRNG Master Seed:</span>
          <NumericInput
            isInteger
            value={config.seed}
            onChange={(val) => setConfig({ ...config, seed: val })}
            disabled={disabled}
            className="w-20 bg-graphite-950 border border-slate-800 px-2 py-1 rounded text-right text-slate-200"
          />
        </div>

        {/* Validation Errors */}
        {validationError && (
          <div className="p-2 rounded bg-rose-950/60 border border-rose-800 text-rose-300 text-[11px] flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>{validationError}</span>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={disabled || loading}
          className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 font-bold py-2 px-4 rounded transition-colors flex items-center justify-center space-x-2 text-xs uppercase tracking-wider"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-slate-950" />
              <span>Generating Environment...</span>
            </>
          ) : (
            <>
              <RefreshCw className="w-4 h-4 text-slate-950" />
              <span>Generate Simulation</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
};
