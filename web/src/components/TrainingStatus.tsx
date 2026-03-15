import type { TrainingStatus } from '../api';

interface Props {
  status: TrainingStatus;
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export default function TrainingStatusPanel({ status }: Props) {
  const pct = Math.round(status.progress * 100);

  return (
    <Panel title="TRAINING STATUS">
      {/* Progress bar */}
      <div className="w-full h-2 bg-sentience-bg rounded-full overflow-hidden mb-4">
        <div
          className="h-full bg-sentience-cyan transition-all duration-500 ease-out rounded-full"
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <StatusItem label="State" value={status.state} />
        <StatusItem
          label="Iteration"
          value={status.state === 'training' || status.state === 'done'
            ? `${status.current_iteration} / ${status.total_iterations}`
            : '--'}
        />
        <StatusItem
          label="Elapsed"
          value={status.elapsed_seconds > 0 ? formatElapsed(status.elapsed_seconds) : '--'}
        />
        <StatusItem label="Backend" value={status.backend ?? '--'} />
      </div>

      <p className="mt-3 text-xs text-sentience-muted truncate">
        {status.message}
      </p>
    </Panel>
  );
}

function StatusItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-sentience-muted text-xs uppercase tracking-wider mb-1">{label}</div>
      <div className="text-sentience-text font-semibold">{value}</div>
    </div>
  );
}

export function Panel({ title, children, className = '' }: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`bg-sentience-panel border border-sentience-border rounded-xl p-5 ${className}`}>
      <h2 className="text-xs font-bold text-sentience-cyan-dim tracking-[0.2em] uppercase mb-4">
        {title}
      </h2>
      {children}
    </div>
  );
}
