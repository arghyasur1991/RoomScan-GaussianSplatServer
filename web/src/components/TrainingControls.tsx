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
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);

  const doUpload = useCallback(async (file: File) => {
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

  const handleFileChange = useCallback(() => {
    const file = fileRef.current?.files?.[0];
    if (file) doUpload(file);
  }, [doUpload]);

  const handleCancel = useCallback(async () => {
    await cancelTraining();
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) setDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    dragCounter.current = 0;
    const file = e.dataTransfer.files?.[0];
    if (file && file.name.endsWith('.zip')) {
      doUpload(file);
    } else if (file) {
      setUploadMsg('Please drop a .zip file');
    }
  }, [doUpload]);

  const isTraining = status.state === 'training';
  const isDone = status.state === 'done';
  const disabled = uploading || isTraining;

  return (
    <Panel title="CONTROLS">
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => !disabled && fileRef.current?.click()}
        className={`
          relative rounded-xl border-2 border-dashed p-6 mb-4 transition-all cursor-pointer
          flex flex-col items-center justify-center gap-2 text-center
          ${disabled ? 'opacity-40 pointer-events-none' : ''}
          ${dragging
            ? 'border-sentience-cyan bg-sentience-cyan/10 scale-[1.01]'
            : 'border-sentience-border hover:border-sentience-cyan/50 hover:bg-sentience-panel/50'}
        `}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".zip"
          className="hidden"
          onChange={handleFileChange}
          disabled={disabled}
        />
        <svg className="w-8 h-8 text-sentience-cyan-dim" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
        </svg>
        <span className="text-sm text-sentience-text font-medium">
          {uploading ? 'Uploading...' : dragging ? 'Drop ZIP here' : 'Drag & drop ZIP or click to browse'}
        </span>
        <span className="text-xs text-sentience-muted">
          GSExport ZIP (images, frames.jsonl, points3d.ply)
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-3">
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
