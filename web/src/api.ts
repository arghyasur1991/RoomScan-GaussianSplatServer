export interface TrainingStatus {
  state: 'idle' | 'training' | 'done' | 'error';
  progress: number;
  message: string;
  backend: string | null;
  current_iteration: number;
  total_iterations: number;
  elapsed_seconds: number;
  run_name: string | null;
  eval_psnr: number | null;
  eval_ssim: number | null;
  eval_l1: number | null;
  gaussian_count: number | null;
}

export interface StepMetric {
  step: number;
  splats: number;
  ms_per_iter: number;
  loss?: number;
}

export interface EvalView {
  name: string;
  psnr: number;
  ssim: number;
  l1: number;
}

export interface MetricsResponse {
  steps: StepMetric[];
  eval?: {
    psnr: number;
    ssim: number;
    l1: number;
    gaussian_count: number;
    per_view: EvalView[];
  };
}

export interface Run {
  name: string;
  has_ply: boolean;
  ply_size_mb: number;
  is_current: boolean;
  eval_psnr: number | null;
  eval_ssim: number | null;
  eval_l1: number | null;
  gaussian_count: number | null;
  elapsed_seconds: number | null;
  iterations: number | null;
  backend: string | null;
}

export interface RunsResponse {
  runs: Run[];
}

export interface Checkpoint {
  step: number;
  filename: string;
  size_mb: number;
}

export interface CheckpointsResponse {
  checkpoints: Checkpoint[];
  count: number;
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

export async function fetchStatus(): Promise<TrainingStatus> {
  const res = await fetch('/api/status');
  return res.json();
}

export async function fetchKeyframes(): Promise<KeyframesResponse> {
  const res = await fetch('/api/keyframes');
  return res.json();
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch('/api/metrics');
  return res.json();
}

export async function fetchRuns(): Promise<RunsResponse> {
  const res = await fetch('/api/runs');
  return res.json();
}

export async function fetchCheckpoints(): Promise<CheckpointsResponse> {
  const res = await fetch('/api/checkpoints');
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
