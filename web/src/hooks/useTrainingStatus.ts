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
};

export function useTrainingStatus(): TrainingStatus {
  const [status, setStatus] = useState<TrainingStatus>(INITIAL_STATUS);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/status`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      try {
        setStatus(JSON.parse(e.data));
      } catch { /* ignore parse errors */ }
    };

    ws.onclose = () => {
      retryRef.current = setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return status;
}
