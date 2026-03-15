import { useCallback, useRef, useState } from 'react';
import type { TrainingStatus } from '../api';
import { cancelTraining, downloadPlyUrl, uploadZip } from '../api';
import { Panel } from './TrainingStatus';

interface Props {
  status: TrainingStatus;
}

export default function TrainingControls({ status }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');

  const handleUpload = useCallback(async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg('Uploading...');
    try {
      const res = await uploadZip(file);
      setUploadMsg(res.status === 'started' ? 'Training started!' : res.error ?? 'Unknown error');
    } catch (e) {
      setUploadMsg(`Upload failed: ${e}`);
    } finally {
      setUploading(false);
    }
  }, []);

  const handleCancel = useCallback(async () => {
    await cancelTraining();
  }, []);

  const isTraining = status.state === 'training';
  const isDone = status.state === 'done';

  return (
    <Panel title="CONTROLS">
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            ref={fileRef}
            type="file"
            accept=".zip"
            className="hidden"
            onChange={handleUpload}
            disabled={uploading || isTraining}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading || isTraining}
            className="btn"
          >
            {uploading ? 'Uploading...' : 'Upload ZIP'}
          </button>
        </label>

        <button
          onClick={handleCancel}
          disabled={!isTraining}
          className="btn btn-danger"
        >
          Cancel Training
        </button>

        <a
          href={downloadPlyUrl()}
          download="splat.ply"
          className={`btn btn-primary ${!isDone ? 'pointer-events-none opacity-40' : ''}`}
        >
          Download PLY
        </a>

        {uploadMsg && (
          <span className="text-xs text-sentience-muted ml-2">{uploadMsg}</span>
        )}
      </div>
    </Panel>
  );
}
