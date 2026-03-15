export interface TrainingStatus {
  state: 'idle' | 'training' | 'done' | 'error';
  progress: number;
  message: string;
  backend: string | null;
  current_iteration: number;
  total_iterations: number;
  elapsed_seconds: number;
}

export interface Keyframe {
  id: number;
  ts: number;
  px: number; py: number; pz: number;
  qx: number; qy: number; qz: number; qw: number;
  fx: number; fy: number;
  cx: number; cy: number;
  sw: number; sh: number;
  w: number; h: number;
  image_url: string;
}

export interface KeyframesResponse {
  keyframes: Keyframe[];
  count: number;
}

export interface RendersResponse {
  renders: string[];
  count: number;
}

export async function fetchStatus(): Promise<TrainingStatus> {
  const res = await fetch('/api/status');
  return res.json();
}

export async function fetchKeyframes(): Promise<KeyframesResponse> {
  const res = await fetch('/api/keyframes');
  return res.json();
}

export async function fetchRenders(): Promise<RendersResponse> {
  const res = await fetch('/api/renders');
  return res.json();
}

export async function cancelTraining(): Promise<void> {
  await fetch('/cancel', { method: 'POST' });
}

export async function uploadZip(file: File): Promise<{ status?: string; error?: string }> {
  const res = await fetch('/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/zip' },
    body: file,
  });
  return res.json();
}

export function downloadPlyUrl(): string {
  return '/download';
}
