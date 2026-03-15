import { useEffect, useState } from 'react';
import type { TrainingStatus } from '../api';
import { fetchRenders } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

export default function RenderGallery({ status }: Props) {
  const [renders, setRenders] = useState<string[]>([]);

  useEffect(() => {
    if (status.state !== 'training' && status.state !== 'done') return;
    const interval = setInterval(() => {
      fetchRenders().then((r) => setRenders(r.renders)).catch(() => {});
    }, 5000);
    fetchRenders().then((r) => setRenders(r.renders)).catch(() => {});
    return () => clearInterval(interval);
  }, [status.state]);

  const latest = renders.length > 0 ? renders[renders.length - 1] : null;

  return (
    <Panel title={`INTERMEDIATE RENDERS${renders.length > 0 ? ` (${renders.length})` : ''}`} className="h-full">
      {latest ? (
        <div className="space-y-3">
          <img
            src={`/api/renders/${latest}`}
            alt="Latest render"
            className="w-full rounded-lg border border-sentience-border"
          />
          {renders.length > 1 && (
            <div className="flex gap-1 overflow-x-auto pb-1">
              {renders.slice(-8).map((r) => (
                <img
                  key={r}
                  src={`/api/renders/${r}`}
                  alt={r}
                  className="h-12 rounded border border-sentience-border flex-shrink-0"
                />
              ))}
            </div>
          )}
        </div>
      ) : (
        <p className="text-sentience-muted text-sm">
          No intermediate renders yet. These appear during training if the backend produces evaluation images.
        </p>
      )}
    </Panel>
  );
}
