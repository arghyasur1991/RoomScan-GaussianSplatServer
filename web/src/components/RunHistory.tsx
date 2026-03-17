import { useCallback, useEffect, useState } from 'react';
import type { TrainingStatus, Run } from '../api';
import { fetchRuns } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

const ITER_PRESETS = [7000, 15000, 30000];

export default function RunHistory({ status }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [switching, setSwitching] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [retraining, setRetraining] = useState<string | null>(null);
  const [clearingAll, setClearingAll] = useState(false);
  const [retrainTarget, setRetrainTarget] = useState<string | null>(null);
  const [retrainIters, setRetrainIters] = useState(7000);

  const isTraining = status.state === 'training';
  const busy = switching !== null || deleting !== null || retraining !== null || clearingAll;

  const loadRuns = useCallback(() => {
    fetchRuns()
      .then((data) => setRuns(data.runs ?? []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns, status.state]);

  const activate = useCallback(async (name: string) => {
    setSwitching(name);
    try {
      await fetch(`/api/runs/${name}/activate`, { method: 'POST' });
      loadRuns();
    } catch { /* ignore */ } finally {
      setSwitching(null);
    }
  }, [loadRuns]);

  const deleteRun = useCallback(async (name: string) => {
    if (!confirm(`Delete run "${name}"? This cannot be undone.`)) return;
    setDeleting(name);
    try {
      await fetch(`/api/runs/${name}`, { method: 'DELETE' });
      loadRuns();
    } catch { /* ignore */ } finally {
      setDeleting(null);
    }
  }, [loadRuns]);

  const startRetrain = useCallback(async (name: string) => {
    setRetraining(name);
    setRetrainTarget(null);
    try {
      await fetch(`/api/runs/${name}/retrain?iterations=${retrainIters}`, { method: 'POST' });
      loadRuns();
    } catch { /* ignore */ } finally {
      setRetraining(null);
    }
  }, [loadRuns, retrainIters]);

  const deleteAll = useCallback(async () => {
    if (!confirm(`Delete ALL ${runs.length} runs? This cannot be undone.`)) return;
    setClearingAll(true);
    try {
      await fetch('/api/runs', { method: 'DELETE' });
      loadRuns();
    } catch { /* ignore */ } finally {
      setClearingAll(false);
    }
  }, [loadRuns, runs.length]);

  if (runs.length === 0) return null;

  return (
    <Panel title={`RUN HISTORY (${runs.length})`}>
      <div className="space-y-1 max-h-[300px] overflow-y-auto">
        {runs.map((run) => (
          <div key={run.name}>
            <div
              className={`flex items-center justify-between text-xs px-3 py-2 rounded-lg border ${
                run.is_current
                  ? 'border-sentience-cyan/50 bg-sentience-cyan/5'
                  : 'border-sentience-border bg-sentience-bg/30'
              }`}
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className={`font-mono truncate ${run.is_current ? 'text-sentience-cyan' : 'text-sentience-text'}`}>
                  {run.name}
                </span>
                {run.has_ply && (
                  <span className="text-sentience-muted flex-shrink-0">
                    {run.ply_size_mb} MB
                  </span>
                )}
                {run.eval_psnr != null && (
                  <span className="text-sentience-muted flex-shrink-0">
                    {run.eval_psnr.toFixed(1)} dB
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                {run.is_current ? (
                  <span className="text-sentience-cyan font-semibold">Active</span>
                ) : run.has_ply ? (
                  <button
                    onClick={() => activate(run.name)}
                    disabled={busy || isTraining}
                    className="btn btn-sm"
                  >
                    {switching === run.name ? 'Loading...' : 'Load'}
                  </button>
                ) : (
                  <span className="text-sentience-muted">No PLY</span>
                )}
                <button
                  onClick={() => {
                    if (retrainTarget === run.name) {
                      setRetrainTarget(null);
                    } else {
                      setRetrainTarget(run.name);
                      setRetrainIters(run.iterations ?? 7000);
                    }
                  }}
                  disabled={busy || isTraining}
                  className="btn btn-sm !bg-amber-900/40 !border-amber-700/50 hover:!bg-amber-800/60 !text-amber-300"
                  title={`Retrain ${run.name}`}
                >
                  {retraining === run.name ? '...' : '⟳'}
                </button>
                <button
                  onClick={() => deleteRun(run.name)}
                  disabled={busy || isTraining}
                  className="btn btn-sm !bg-red-900/40 !border-red-700/50 hover:!bg-red-800/60 !text-red-300"
                  title={`Delete ${run.name}`}
                >
                  {deleting === run.name ? '...' : '✕'}
                </button>
              </div>
            </div>

            {retrainTarget === run.name && (
              <div className="flex items-center gap-2 mt-1 ml-3 text-xs">
                <span className="text-sentience-muted">Iters:</span>
                {ITER_PRESETS.map((p) => (
                  <button
                    key={p}
                    onClick={() => setRetrainIters(p)}
                    className={`px-2 py-0.5 rounded border text-[10px] ${
                      retrainIters === p
                        ? 'border-sentience-cyan bg-sentience-cyan/10 text-sentience-cyan'
                        : 'border-sentience-border text-sentience-muted hover:border-sentience-cyan/50'
                    }`}
                  >
                    {(p / 1000).toFixed(0)}K
                  </button>
                ))}
                <input
                  type="number"
                  min={100}
                  max={100000}
                  step={1000}
                  value={retrainIters}
                  onChange={(e) => setRetrainIters(parseInt(e.target.value) || 7000)}
                  className="w-16 px-1.5 py-0.5 rounded border border-sentience-border bg-sentience-bg text-sentience-text text-[10px] text-center"
                />
                <button
                  onClick={() => startRetrain(run.name)}
                  disabled={busy || isTraining}
                  className="btn btn-sm !bg-amber-700/40 !border-amber-600/50 hover:!bg-amber-600/60 !text-amber-200"
                >
                  GO
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
      {runs.length > 1 && (
        <button
          onClick={deleteAll}
          disabled={busy || isTraining}
          className="btn btn-sm w-full mt-2 !bg-red-900/30 !border-red-700/40 hover:!bg-red-800/50 !text-red-300"
        >
          {clearingAll ? 'Clearing...' : `Clear All Runs (${runs.length})`}
        </button>
      )}
    </Panel>
  );
}
