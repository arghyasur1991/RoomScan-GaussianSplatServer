import { useEffect, useRef, useState, useCallback } from 'react';
import type { TrainingStatus } from '../api';

const INITIAL_STATUS: TrainingStatus = {
  state: 'idle',
  progress: 0,
  message: 'Connecting...',
  backend: null,
  current_iteration: 0,
  total_iterations: 0,
  elapsed_seconds: 0,
  run_name: null,
};

const MIN_RETRY_MS = 1000;
const MAX_RETRY_MS = 10000;

export function useTrainingStatus(): TrainingStatus {
  const [status, setStatus] = useState<TrainingStatus>(INITIAL_STATUS);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const closedIntentionally = useRef(false);
  const retryDelay = useRef(MIN_RETRY_MS);

  const connect = useCallback(() => {
    if (closedIntentionally.current) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/status`);
    wsRef.current = ws;

    ws.onopen = () => {
      retryDelay.current = MIN_RETRY_MS;
    };

    ws.onmessage = (e) => {
      try {
        setStatus(JSON.parse(e.data));
      } catch { /* ignore parse errors */ }
    };

    ws.onclose = () => {
      if (closedIntentionally.current) return;
      setStatus((prev) => ({
        ...prev,
        message: prev.state === 'idle' ? 'Reconnecting...' : prev.message,
      }));
      retryRef.current = setTimeout(connect, retryDelay.current);
      retryDelay.current = Math.min(retryDelay.current * 1.5, MAX_RETRY_MS);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    closedIntentionally.current = false;
    connect();
    return () => {
      closedIntentionally.current = true;
      clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return status;
}
