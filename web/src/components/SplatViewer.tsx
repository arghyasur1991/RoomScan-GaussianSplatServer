import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { TrainingStatus } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

type ViewerState = 'idle' | 'loading' | 'ready' | 'error';

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

    setViewerState('loading');
    setProgress('Checking...');
    setError('');

    try {
      const probe = await fetch('/api/splat', { method: 'HEAD' });
      if (!probe.ok) throw new Error(`Splat not available (HTTP ${probe.status})`);
      if (cancelledRef.current) return;

      setProgress('Loading viewer...');
      const { Viewer } = await import('@mkkellogg/gaussian-splats-3d');
      if (cancelledRef.current || !canvasRef.current) return;

      const container = canvasRef.current;
      const width = container.clientWidth || 600;
      const height = container.clientHeight || 400;
      console.log(`SplatViewer container: ${width}x${height}`);

      const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setClearColor(0x111111);
      container.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      const camera = new THREE.PerspectiveCamera(65, width / height, 0.1, 100);
      camera.position.set(0, 0, 3);
      camera.lookAt(0, 0, 0);

      const scene = new THREE.Scene();

      const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;
      controls.target.set(0, 0, 0);
      controlsRef.current = controls;

      // Start render loop immediately
      const animate = () => {
        if (cancelledRef.current) return;
        frameRef.current = requestAnimationFrame(animate);
        controls.update();
        viewer.update();
        viewer.render();
      };

      const viewer = new Viewer({
        scene,
        renderer,
        camera,
        selfDrivenMode: false,
        sharedMemoryForWorkers: false,
      });
      viewerRef.current = viewer;
      animate();

      setProgress('Downloading splat...');
      console.time('addSplatScene');

      await viewer.addSplatScene('/api/splat', {
        showLoadingUI: false,
        format: 2, /* SceneFormat.Ply — full PLY with all gaussians */
        splatAlphaRemovalThreshold: 5,
        onProgress: (_p: number, label: string, status: number) => {
          if (status === 0) setProgress(`Downloading... ${label}`);
          else setProgress('Processing...');
        },
      });
      console.timeEnd('addSplatScene');

      if (cancelledRef.current) return;
      setViewerState('ready');
      console.log('Splat viewer ready');

      const handleResize = () => {
        if (cancelledRef.current || !container) return;
        const w = container.clientWidth || 600;
        const h = container.clientHeight || 400;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener('resize', handleResize);

    } catch (e: any) {
      console.error('SplatViewer error:', e);
      if (!cancelledRef.current) {
        setError(e.message || 'Failed to load splat');
        setViewerState('error');
      }
    }
  }, [cleanup]);

  const runName = status.run_name;

  useEffect(() => {
    if (isDone) {
      cleanup();
      setViewerState('idle');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runName]);

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
        <div ref={canvasRef} className="absolute inset-0" />

        {viewerState !== 'ready' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 pointer-events-none z-10">
            {viewerState === 'idle' && !isDone && (
              <span className="text-sentience-muted text-sm">
                Splat viewer will appear after training completes.
              </span>
            )}

            {viewerState === 'loading' && (
              <>
                <div className="w-8 h-8 border-2 border-sentience-cyan border-t-transparent rounded-full animate-spin" />
                <span className="text-sentience-cyan text-sm">{progress}</span>
              </>
            )}

            {viewerState === 'error' && (
              <div className="pointer-events-auto flex flex-col items-center gap-2">
                <span className="text-sentience-error text-sm">{error}</span>
                {isDone && (
                  <button onClick={loadSplat} className="btn btn-primary text-xs">
                    Retry
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}
