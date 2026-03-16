import { useCallback, useEffect, useState } from 'react';
import type { Keyframe, TrainingStatus } from '../api';
import { fetchKeyframes } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

export default function KeyframeBrowser({ status }: Props) {
  const [keyframes, setKeyframes] = useState<Keyframe[]>([]);
  const [selected, setSelected] = useState<Keyframe | null>(null);

  useEffect(() => {
    if (status.state === 'idle') return;
    fetchKeyframes().then((res) => setKeyframes(res.keyframes)).catch(() => {});
  }, [status.state, status.run_name]);

  const close = useCallback(() => setSelected(null), []);

  return (
    <Panel title={`KEYFRAMES${keyframes.length > 0 ? ` (${keyframes.length})` : ''}`}>
      {keyframes.length === 0 ? (
        <p className="text-sentience-muted text-sm">No keyframes available. Upload training data to see captured frames.</p>
      ) : (
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2 max-h-[300px] overflow-y-auto">
          {keyframes.map((kf) => (
            <button
              key={kf.id}
              onClick={() => setSelected(kf)}
              className="aspect-[4/3] rounded-lg overflow-hidden border border-sentience-border hover:border-sentience-cyan transition-colors cursor-pointer bg-sentience-bg"
            >
              <img
                src={`${kf.image_url}?run=${status.run_name ?? ''}`}
                alt={`Frame ${kf.id}`}
                loading="lazy"
                className="w-full h-full object-cover"
              />
            </button>
          ))}
        </div>
      )}

      {selected && <KeyframeDetail keyframe={selected} onClose={close} />}
    </Panel>
  );
}

function KeyframeDetail({ keyframe: kf, onClose }: { keyframe: Keyframe; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-sentience-panel border border-sentience-border rounded-2xl overflow-hidden max-w-3xl w-full mx-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={kf.image_url}
          alt={`Frame ${kf.id}`}
          className="w-full max-h-[60vh] object-contain bg-black"
        />
        <div className="p-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <MetaItem label="ID" value={String(kf.id)} />
          <MetaItem label="Timestamp" value={`${kf.ts.toFixed(2)}s`} />
          <MetaItem label="Position" value={`(${kf.px.toFixed(2)}, ${kf.py.toFixed(2)}, ${kf.pz.toFixed(2)})`} />
          <MetaItem label="Focal" value={`${kf.fx.toFixed(1)}, ${kf.fy.toFixed(1)}`} />
          <MetaItem label="Principal" value={`${kf.cx.toFixed(1)}, ${kf.cy.toFixed(1)}`} />
          <MetaItem label="Image Size" value={`${kf.w}x${kf.h}`} />
          <MetaItem label="Sensor" value={`${kf.sw}x${kf.sh}`} />
          <MetaItem
            label="Rotation"
            value={`(${kf.qx.toFixed(3)}, ${kf.qy.toFixed(3)}, ${kf.qz.toFixed(3)}, ${kf.qw.toFixed(3)})`}
          />
        </div>
        <div className="px-4 pb-4">
          <button onClick={onClose} className="btn w-full">Close</button>
        </div>
      </div>
    </div>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-sentience-muted uppercase tracking-wider">{label}</span>
      <div className="text-sentience-text font-mono mt-0.5">{value}</div>
    </div>
  );
}
