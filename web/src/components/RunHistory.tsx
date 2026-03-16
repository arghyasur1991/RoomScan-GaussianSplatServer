import { useCallback, useEffect, useState } from 'react';
import type { TrainingStatus } from '../api';
import { Panel } from './TrainingStatus';

interface Run {
  name: string;
  has_ply: boolean;
  ply_size_mb: number;
  is_current: boolean;
}

interface Props {
  status: TrainingStatus;
}

export default function RunHistory({ status }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [switching, setSwitching] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [retraining, setRetraining] = useState<string | null>(null);
  const [clearingAll, setClearingAll] = useState(false);

  const isTraining = status.state === 'training';
  const busy = switching !== null || deleting !== null || retraining !== null || clearingAll;

  const fetchRuns = useCallback(() => {
    fetch('/api/runs')
      .then((r) => r.json())
      .then((data) => setRuns(data.runs ?? []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns, status.state]);

  const activate = useCallback(async (name: string) => {
    setSwitching(name);
    try {
      await fetch(`/api/runs/${name}/activate`, { method: 'POST' });
      fetchRuns();
    } catch { /* ignore */ } finally {
      setSwitching(null);
    }
  }, [fetchRuns]);

  const deleteRun = useCallback(async (name: string) => {
    if (!confirm(`Delete run "${name}"? This cannot be undone.`)) return;
    setDeleting(name);
    try {
      await fetch(`/api/runs/${name}`, { method: 'DELETE' });
      fetchRuns();
    } catch { /* ignore */ } finally {
      setDeleting(null);
    }
  }, [fetchRuns]);

  const retrain = useCallback(async (name: string) => {
    if (!confirm(`Retrain run "${name}"? This will replace its current output.`)) return;
    setRetraining(name);
    try {
      await fetch(`/api/runs/${name}/retrain`, { method: 'POST' });
      fetchRuns();
    } catch { /* ignore */ } finally {
      setRetraining(null);
    }
  }, [fetchRuns]);

  const deleteAll = useCallback(async () => {
    if (!confirm(`Delete ALL ${runs.length} runs? This cannot be undone.`)) return;
    setClearingAll(true);
    try {
      await fetch('/api/runs', { method: 'DELETE' });
      fetchRuns();
    } catch { /* ignore */ } finally {
      setClearingAll(false);
    }
  }, [fetchRuns, runs.length]);

  if (runs.length === 0) return null;

  return (
    <Panel title={`RUN HISTORY (${runs.length})`}>
      <div className="space-y-1 max-h-[200px] overflow-y-auto">
        {runs.map((run) => (
          <div
            key={run.name}
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
                onClick={() => retrain(run.name)}
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
