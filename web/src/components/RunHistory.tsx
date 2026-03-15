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
                  disabled={switching !== null || status.state === 'training'}
                  className="btn btn-sm"
                >
                  {switching === run.name ? 'Loading...' : 'Load'}
                </button>
              ) : (
                <span className="text-sentience-muted">No PLY</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
