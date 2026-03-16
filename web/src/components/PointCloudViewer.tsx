import { useEffect, useMemo, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import type { TrainingStatus } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

export default function PointCloudViewer({ status }: Props) {
  const [pointSize, setPointSize] = useState(2.0);
  const hasData = status.state !== 'idle';
  const runName = status.run_name;

  return (
    <Panel title="POINT CLOUD" className="h-full flex flex-col">
      <div className="flex items-center gap-3 mb-3">
        <label className="text-xs text-sentience-muted">Point Size</label>
        <input
          type="range"
          min="0.5"
          max="8"
          step="0.5"
          value={pointSize}
          onChange={(e) => setPointSize(parseFloat(e.target.value))}
          className="flex-1 accent-sentience-cyan"
        />
        <span className="text-xs text-sentience-text w-6 text-right">{pointSize}</span>
      </div>
      <div className="flex-1 rounded-lg overflow-hidden bg-sentience-bg border border-sentience-border min-h-[300px]">
        {hasData ? (
          <Canvas key={runName ?? 'default'} camera={{ position: [2, 2, 2], fov: 60 }}>
            <ambientLight intensity={0.5} />
            <PointCloudScene pointSize={pointSize} />
            <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
          </Canvas>
        ) : (
          <div className="flex items-center justify-center h-full text-sentience-muted text-sm">
            Upload training data to view the point cloud.
          </div>
        )}
      </div>
    </Panel>
  );
}

function PointCloudScene({ pointSize }: { pointSize: number }) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const { camera } = useThree();

  useEffect(() => {
    const loader = new PLYLoader();
    loader.load('/api/pointcloud', (geo) => {
      geo.computeBoundingBox();
      const box = geo.boundingBox!;
      const center = new THREE.Vector3();
      box.getCenter(center);
      geo.translate(-center.x, -center.y, -center.z);

      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      if (maxDim > 0) {
        const scale = 4.0 / maxDim;
        geo.scale(scale, scale, scale);
      }

      setGeometry(geo);
      (camera as THREE.PerspectiveCamera).position.set(3, 3, 3);
      camera.lookAt(0, 0, 0);
    });
  }, [camera]);

  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      size: pointSize * 0.01,
      vertexColors: true,
      sizeAttenuation: true,
    });
  }, [pointSize]);

  if (!geometry) return null;

  return <points geometry={geometry} material={material} />;
}
