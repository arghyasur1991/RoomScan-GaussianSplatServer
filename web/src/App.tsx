import { useTrainingStatus } from './hooks/useTrainingStatus';
import Dashboard from './components/Dashboard';

export default function App() {
  const status = useTrainingStatus();

  return (
    <div className="min-h-screen bg-sentience-bg">
      <header className="border-b border-sentience-border px-6 py-4 flex items-end gap-3">
        <h1 className="text-2xl font-bold text-sentience-cyan tracking-[0.25em]">
          SENTIENCE
        </h1>
        <span className="text-sm text-sentience-cyan-dim tracking-[0.2em] mb-0.5">
          GS TRAINING SERVER
        </span>
        <div className="ml-auto">
          <StateBadge state={status.state} />
        </div>
      </header>
      <Dashboard status={status} />
    </div>
  );
}

function StateBadge({ state }: { state: string }) {
  const colors: Record<string, string> = {
    idle: 'bg-sentience-muted/20 text-sentience-muted border-sentience-muted/40',
    training: 'bg-sentience-cyan/10 text-sentience-cyan border-sentience-cyan/40 animate-pulse',
    done: 'bg-sentience-success/10 text-sentience-success border-sentience-success/40',
    error: 'bg-sentience-error/10 text-sentience-error border-sentience-error/40',
  };

  return (
    <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${colors[state] ?? colors.idle}`}>
      {state}
    </span>
  );
}
