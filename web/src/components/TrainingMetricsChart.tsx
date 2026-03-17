import { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import type { TrainingStatus, StepMetric } from '../api';
import { fetchMetrics } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

export default function TrainingMetricsChart({ status }: Props) {
  const [metrics, setMetrics] = useState<StepMetric[]>([]);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const data = await fetchMetrics();
        if (!cancelled) setMetrics(data.steps);
      } catch { /* ignore */ }
    };
    poll();
    if (status.state === 'training') {
      const id = setInterval(poll, 2000);
      return () => { cancelled = true; clearInterval(id); };
    }
    return () => { cancelled = true; };
  }, [status.state]);

  if (metrics.length === 0) return null;

  return (
    <Panel title="TRAINING METRICS">
      <div className="space-y-4">
        <div>
          <h3 className="text-xs text-sentience-muted mb-2">Gaussian Count</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis
                dataKey="step"
                tick={{ fill: '#64748b', fontSize: 10 }}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 10 }}
                axisLine={{ stroke: '#334155' }}
                tickFormatter={(v: number) => v >= 1000 ? `${(v/1000).toFixed(0)}k` : String(v)}
              />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#94a3b8' }}
                itemStyle={{ color: '#fbbf24' }}
                formatter={(v: number) => [v.toLocaleString(), 'Splats']}
                labelFormatter={(l: number) => `Step ${l}`}
              />
              <Line type="monotone" dataKey="splats" stroke="#fbbf24" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h3 className="text-xs text-sentience-muted mb-2">Performance (ms/iter)</h3>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis
                dataKey="step"
                tick={{ fill: '#64748b', fontSize: 10 }}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 10 }}
                axisLine={{ stroke: '#334155' }}
                unit="ms"
              />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#94a3b8' }}
                itemStyle={{ color: '#64748b' }}
                formatter={(v: number) => [`${v.toFixed(1)} ms`, 'Time/iter']}
                labelFormatter={(l: number) => `Step ${l}`}
              />
              <Line type="monotone" dataKey="ms_per_iter" stroke="#64748b" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Panel>
  );
}
