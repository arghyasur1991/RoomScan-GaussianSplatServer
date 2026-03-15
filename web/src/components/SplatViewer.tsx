import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { TrainingStatus } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

export default function SplatViewer({ status }: Props) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const viewerRef = useRef<any>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<any>(null);
  const frameRef = useRef<number>(0);

  const isDone = status.state === 'done';

  useEffect(() => {
    if (!isDone || !canvasRef.current) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const splatUrl = `/api/splat`;

        const probe = await fetch(splatUrl, { method: 'HEAD' });
        if (!probe.ok) {
          throw new Error(`Splat not available (${probe.status})`);
        }
        if (cancelled) return;

        const { Viewer } = await import('@mkkellogg/gaussian-splats-3d');
        if (cancelled || !canvasRef.current) return;

        const container = canvasRef.current;
        const width = container.clientWidth;
        const height = container.clientHeight || 400;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 500);
        camera.position.set(2, 2, 2);

        const scene = new THREE.Scene();

        const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controlsRef.current = controls;

        const viewer = new Viewer({
          scene,
          renderer,
          camera,
          selfDrivenMode: false,
        });
        viewerRef.current = viewer;

        await viewer.addSplatScene(splatUrl, { showLoadingUI: true, format: 2 /* SceneFormat.Ply */ });

        if (cancelled) return;

        const animate = () => {
          if (cancelled) return;
          frameRef.current = requestAnimationFrame(animate);
          controls.update();
          viewer.update();
          viewer.render();
        };
        animate();

        setLoaded(true);
        setLoading(false);

        const handleResize = () => {
          if (!container || cancelled) return;
          const w = container.clientWidth;
          const h = container.clientHeight || 400;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        };
        window.addEventListener('resize', handleResize);

        return () => {
          window.removeEventListener('resize', handleResize);
        };
      } catch (e: any) {
        if (!cancelled) {
          setError(e.message || 'Failed to load splat');
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(frameRef.current);
      viewerRef.current?.dispose?.();
      rendererRef.current?.dispose();
      controlsRef.current?.dispose?.();
      if (canvasRef.current) canvasRef.current.innerHTML = '';
      setLoaded(false);
    };
  }, [isDone]);

  const showOverlay = !loaded;

  return (
    <Panel title="GAUSSIAN SPLAT" className="h-full flex flex-col">
      <div className="relative flex-1 rounded-lg overflow-hidden bg-sentience-bg border border-sentience-border min-h-[300px]">
        {/* Three.js canvas lives here -- React never touches children of this div */}
        <div ref={canvasRef} className="absolute inset-0" />

        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            {!isDone && !loading && !error && (
              <span className="text-sentience-muted text-sm">
                Splat viewer will appear after training completes.
              </span>
            )}
            {error && (
              <span className="text-sentience-error text-sm">
                {error}
              </span>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}
