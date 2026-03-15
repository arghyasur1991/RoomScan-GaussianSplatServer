import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { TrainingStatus } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

type ViewerState = 'idle' | 'downloading' | 'processing' | 'ready' | 'error';

export default function SplatViewer({ status }: Props) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [viewerState, setViewerState] = useState<ViewerState>('idle');
  const [progress, setProgress] = useState('');
  const [error, setError] = useState('');
  const viewerRef = useRef<any>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<any>(null);
  const frameRef = useRef<number>(0);
  const cancelledRef = useRef(false);

  const isDone = status.state === 'done';

  const cleanup = useCallback(() => {
    cancelledRef.current = true;
    cancelAnimationFrame(frameRef.current);
    viewerRef.current?.dispose?.();
    viewerRef.current = null;
    rendererRef.current?.dispose();
    rendererRef.current = null;
    controlsRef.current?.dispose?.();
    controlsRef.current = null;
    if (canvasRef.current) canvasRef.current.innerHTML = '';
  }, []);

  const loadSplat = useCallback(async () => {
    cleanup();
    cancelledRef.current = false;

    if (!canvasRef.current) return;

    setViewerState('downloading');
    setProgress('Checking availability...');
    setError('');

    try {
      const probe = await fetch('/api/splat', { method: 'HEAD' });
      if (!probe.ok) throw new Error(`Splat not available (HTTP ${probe.status})`);
      if (cancelledRef.current) return;

      const contentLength = probe.headers.get('content-length');
      const sizeMB = contentLength ? (parseInt(contentLength) / 1048576).toFixed(1) : '?';
      setProgress(`Downloading ${sizeMB} MB...`);

      const { Viewer } = await import('@mkkellogg/gaussian-splats-3d');
      if (cancelledRef.current || !canvasRef.current) return;

      const container = canvasRef.current;
      const width = container.clientWidth || 600;
      const height = container.clientHeight || 400;

      // Use selfDrivenMode: true — the library manages its own render loop
      // and worker-based sorting, which is critical for 335k+ gaussians
      const viewer = new Viewer({
        cameraUp: [0, -1, 0],
        initialCameraPosition: [0, -2, -6],
        initialCameraLookAt: [0, 0, 0],
        selfDrivenMode: true,
        useBuiltInControls: true,
        dynamicScene: false,
        sharedMemoryForWorkers: false,
        rootElement: container,
      });
      viewerRef.current = viewer;

      setViewerState('downloading');
      console.time('addSplatScene');

      await viewer.addSplatScene('/api/splat', {
        showLoadingUI: true,
        format: 0, /* SceneFormat.Splat */
        splatAlphaRemovalThreshold: 10,
        onProgress: (_percent: number, label: string, loaderStatus: number) => {
          console.timeLog('addSplatScene', `${label} status=${loaderStatus}`);
          if (loaderStatus === 0) {
            setProgress(`Downloading... ${label}`);
          } else {
            setViewerState('processing');
            setProgress('');
          }
        },
      });
      console.timeEnd('addSplatScene');

      if (cancelledRef.current) return;
      setViewerState('ready');
      console.log('Splat viewer ready');

    } catch (e: any) {
      console.error('SplatViewer error:', e);
      if (!cancelledRef.current) {
        setError(e.message || 'Failed to load splat');
        setViewerState('error');
      }
    }
  }, [cleanup]);

  useEffect(() => {
    if (isDone && viewerState === 'idle') {
      loadSplat();
    }
    if (!isDone && viewerState !== 'idle') {
      cleanup();
      setViewerState('idle');
    }
  }, [isDone, viewerState, loadSplat, cleanup]);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return (
    <Panel title="GAUSSIAN SPLAT" className="h-full flex flex-col">
      <div className="relative flex-1 rounded-lg overflow-hidden bg-sentience-bg border border-sentience-border min-h-[400px]">
        {/* Library creates its own canvas + controls inside this div */}
        <div ref={canvasRef} className="absolute inset-0" />

        {viewerState !== 'ready' && viewerState !== 'processing' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 z-10">
            {viewerState === 'idle' && !isDone && (
              <span className="text-sentience-muted text-sm">
                Splat viewer will appear after training completes.
              </span>
            )}

            {viewerState === 'downloading' && (
              <>
                <div className="w-8 h-8 border-2 border-sentience-cyan border-t-transparent rounded-full animate-spin" />
                <span className="text-sentience-cyan text-sm">{progress}</span>
              </>
            )}

            {viewerState === 'error' && (
              <>
                <span className="text-sentience-error text-sm">{error}</span>
                {isDone && (
                  <button
                    onClick={loadSplat}
                    className="btn btn-primary text-xs mt-2"
                  >
                    Retry
                  </button>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}
