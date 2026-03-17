import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import type { TrainingStatus, Run } from '../api';
import { fetchRuns } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

function formatTime(s: number | null): string {
  if (s == null) return '--';
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export default function BenchmarkComparison({ status }: Props) {
  const [runs, setRuns] = useState<Run[]>([]);

  useEffect(() => {
    fetchRuns()
      .then((data) => setRuns(data.runs))
      .catch(() => {});
  }, [status.state, status.run_name]);

  const evaluated = runs.filter((r) => r.eval_psnr != null);
  if (evaluated.length < 2) return null;

  const baseline = evaluated[evaluated.length - 1];

  const chartData = evaluated.map((r) => ({
    name: r.name.slice(0, 8),
    psnr: r.eval_psnr ?? 0,
    ssim: r.eval_ssim ?? 0,
  }));

  return (
    <Panel title="BENCHMARK COMPARISON">
      <div className="overflow-x-auto mb-4">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-sentience-border text-sentience-muted">
              <th className="text-left py-2 pr-3">Run</th>
              <th className="text-right px-2">PSNR</th>
              <th className="text-right px-2">Delta</th>
              <th className="text-right px-2">SSIM</th>
              <th className="text-right px-2">Splats</th>
              <th className="text-right px-2">Time</th>
            </tr>
          </thead>
          <tbody>
            {evaluated.map((r) => {
              const delta = baseline.eval_psnr != null && r.eval_psnr != null
                ? r.eval_psnr - baseline.eval_psnr
                : null;
              return (
                <tr key={r.name} className={`border-b border-sentience-border/30 ${r.is_current ? 'text-sentience-cyan' : 'text-sentience-text'}`}>
                  <td className="py-1.5 pr-3 font-mono">{r.name}</td>
                  <td className="text-right px-2">{r.eval_psnr?.toFixed(2) ?? '--'}</td>
                  <td className={`text-right px-2 ${delta != null && delta > 0 ? 'text-green-400' : delta != null && delta < 0 ? 'text-red-400' : ''}`}>
                    {delta != null ? (delta > 0 ? '+' : '') + delta.toFixed(2) : '--'}
                  </td>
                  <td className="text-right px-2">{r.eval_ssim?.toFixed(3) ?? '--'}</td>
                  <td className="text-right px-2">
                    {r.gaussian_count != null ? `${(r.gaussian_count / 1000).toFixed(0)}k` : '--'}
                  </td>
                  <td className="text-right px-2">{formatTime(r.elapsed_seconds)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3 className="text-xs text-sentience-muted mb-2">PSNR by Run</h3>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
          <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#334155' }} />
          <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#334155' }} domain={['dataMin - 0.5', 'dataMax + 0.5']} unit=" dB" />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          <Bar dataKey="psnr" fill="#22d3ee" fillOpacity={0.7} radius={[4, 4, 0, 0]} name="PSNR (dB)" />
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  );
}
