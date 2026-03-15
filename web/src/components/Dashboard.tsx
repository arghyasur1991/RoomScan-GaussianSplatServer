import type { TrainingStatus } from '../api';
import TrainingStatusPanel from './TrainingStatus';
import TrainingControls from './TrainingControls';
import KeyframeBrowser from './KeyframeBrowser';
import PointCloudViewer from './PointCloudViewer';
import SplatViewer from './SplatViewer';
import LogPanel from './LogPanel';
import RunHistory from './RunHistory';

interface Props {
  status: TrainingStatus;
}

export default function Dashboard({ status }: Props) {
  return (
    <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-[1920px] mx-auto">
      {/* Row 1: Status + Controls | Logs */}
      <div className="lg:col-span-2 flex flex-col gap-4">
        <TrainingStatusPanel status={status} />
        <TrainingControls status={status} />
      </div>
      <div className="lg:col-span-1">
        <LogPanel />
      </div>

      {/* Row 2: Run History | Keyframes */}
      <div className="lg:col-span-1">
        <RunHistory status={status} />
      </div>
      <div className="lg:col-span-2">
        <KeyframeBrowser status={status} />
      </div>

      {/* Row 3: Point Cloud | Splat */}
      <div className="lg:col-span-1 min-h-[400px]">
        <PointCloudViewer status={status} />
      </div>
      <div className="lg:col-span-2 min-h-[400px]">
        <SplatViewer status={status} />
      </div>
    </div>
  );
}
