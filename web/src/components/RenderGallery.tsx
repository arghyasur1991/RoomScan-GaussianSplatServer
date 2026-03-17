import { useEffect, useState } from 'react';
import type { TrainingStatus } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

export default function RenderGallery({ status }: Props) {
  const [renders, setRenders] = useState<string[]>([]);
  const [selected, setSelected] = useState(0);
  const [sliderPos, setSliderPos] = useState(50);
  const [dragging, setDragging] = useState(false);
  const isDone = status.state === 'done';

  useEffect(() => {
    if (!isDone) { setRenders([]); return; }
    fetch('/api/renders')
      .then((r) => r.json())
      .then((data) => {
        const imgs = (data.renders ?? []) as string[];
        setRenders(imgs);
        setSelected(0);
      })
      .catch(() => {});
  }, [isDone, status.run_name]);

  if (!isDone || renders.length === 0) return null;

  const renderUrl = `/api/renders/${encodeURIComponent(renders[selected])}`;

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragging) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setSliderPos(Math.max(0, Math.min(100, pct)));
  };

  return (
    <Panel title="RENDER GALLERY">
      <div className="flex items-center gap-2 mb-3">
        <button
          onClick={() => setSelected((s) => Math.max(0, s - 1))}
          disabled={selected === 0}
          className="btn btn-sm"
        >
          Prev
        </button>
        <span className="text-xs text-sentience-muted flex-1 text-center">
          {renders[selected]} ({selected + 1}/{renders.length})
        </span>
        <button
          onClick={() => setSelected((s) => Math.min(renders.length - 1, s + 1))}
          disabled={selected === renders.length - 1}
          className="btn btn-sm"
        >
          Next
        </button>
      </div>

      <div
        className="relative overflow-hidden rounded-lg border border-sentience-border cursor-col-resize select-none"
        style={{ height: 300 }}
        onMouseDown={() => setDragging(true)}
        onMouseUp={() => setDragging(false)}
        onMouseLeave={() => setDragging(false)}
        onMouseMove={handleMouseMove}
      >
        <img
          src={renderUrl}
          alt="Rendered"
          className="absolute inset-0 w-full h-full object-contain"
        />
        <div
          className="absolute inset-0"
          style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
        >
          <div className="absolute inset-0 bg-sentience-bg flex items-center justify-center text-sentience-muted text-sm">
            Ground truth not available
          </div>
        </div>
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-sentience-cyan z-10"
          style={{ left: `${sliderPos}%` }}
        />
      </div>
    </Panel>
  );
}
