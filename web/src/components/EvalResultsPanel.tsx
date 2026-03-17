import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import type { TrainingStatus, EvalView } from '../api';
import { fetchMetrics } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

function psnrColor(v: number): string {
  if (v >= 30) return '#22c55e';
  if (v >= 25) return '#eab308';
  return '#ef4444';
}

export default function EvalResultsPanel({ status }: Props) {
  const [perView, setPerView] = useState<EvalView[]>([]);
  const isDone = status.state === 'done';

  useEffect(() => {
    if (!isDone) return;
    fetchMetrics()
      .then((data) => setPerView(data.eval?.per_view ?? []))
      .catch(() => {});
  }, [isDone, status.run_name]);

  if (!isDone || status.eval_psnr == null) return null;

  return (
    <Panel title="EVALUATION RESULTS">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <MetricCard label="PSNR" value={status.eval_psnr?.toFixed(2) ?? '--'} unit="dB" />
        <MetricCard label="SSIM" value={status.eval_ssim?.toFixed(3) ?? '--'} />
        <MetricCard label="L1" value={status.eval_l1?.toFixed(4) ?? '--'} />
        <MetricCard
          label="Gaussians"
          value={status.gaussian_count != null ? (status.gaussian_count / 1000).toFixed(0) + 'k' : '--'}
        />
      </div>

      {perView.length > 0 && (
        <div>
          <h3 className="text-xs text-sentience-muted mb-2">Per-View PSNR</h3>
          <ResponsiveContainer width="100%" height={Math.max(140, perView.length * 22)}>
            <BarChart data={perView} layout="vertical" margin={{ left: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis
                type="number"
                tick={{ fill: '#64748b', fontSize: 10 }}
                axisLine={{ stroke: '#334155' }}
                domain={['dataMin - 1', 'dataMax + 1']}
                unit=" dB"
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: '#94a3b8', fontSize: 9 }}
                axisLine={{ stroke: '#334155' }}
                width={55}
              />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(v: number) => [`${v.toFixed(2)} dB`, 'PSNR']}
              />
              <Bar dataKey="psnr" radius={[0, 4, 4, 0]}>
                {perView.map((entry, i) => (
                  <Cell key={i} fill={psnrColor(entry.psnr)} fillOpacity={0.7} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  );
}

function MetricCard({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="bg-sentience-bg/50 border border-sentience-border rounded-lg p-3 text-center">
      <div className="text-sentience-muted text-[10px] uppercase tracking-wider mb-1">{label}</div>
      <div className="text-sentience-text text-lg font-bold">
        {value}
        {unit && <span className="text-xs text-sentience-muted ml-1">{unit}</span>}
      </div>
    </div>
  );
}
