import { useEffect, useRef, useState } from 'react';
import { Panel } from './TrainingStatus';

interface LogLine {
  line: string;
}

export default function LogPanel() {
  const [lines, setLines] = useState<string[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const evtSource = new EventSource('/api/logs');
    evtSource.onmessage = (e) => {
      try {
        const data: LogLine = JSON.parse(e.data);
        setLines((prev) => {
          const next = [...prev, data.line];
          return next.length > 1000 ? next.slice(-500) : next;
        });
      } catch { /* ignore */ }
    };
    return () => evtSource.close();
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (autoScroll && el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [lines, autoScroll]);

  return (
    <Panel title="LOGS" className="h-full flex flex-col">
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto font-mono text-xs leading-relaxed min-h-[200px] max-h-[360px] bg-sentience-bg/50 rounded-lg p-3"
      >
        {lines.length === 0 && (
          <span className="text-sentience-muted">Waiting for logs...</span>
        )}
        {lines.map((line, i) => (
          <div key={i} className={lineColor(line)}>
            {line}
          </div>
        ))}
      </div>
      <label className="flex items-center gap-2 mt-2 text-xs text-sentience-muted cursor-pointer">
        <input
          type="checkbox"
          checked={autoScroll}
          onChange={(e) => setAutoScroll(e.target.checked)}
          className="accent-sentience-cyan"
        />
        Auto-scroll
      </label>
    </Panel>
  );
}

function lineColor(line: string): string {
  const lower = line.toLowerCase();
  if (lower.includes('error')) return 'text-sentience-error';
  if (lower.includes('warn')) return 'text-sentience-warning';
  if (lower.includes('done') || lower.includes('complete')) return 'text-sentience-success';
  return 'text-sentience-muted';
}
