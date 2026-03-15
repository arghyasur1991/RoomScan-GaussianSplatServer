"""
Training manager — orchestrates GS training and exposes rich status.

Extracted from gs_server.py with additions:
- Log ring buffer (subprocess stdout/stderr captured)
- Iteration progress parsing from training backend output
- Intermediate render image tracking
- Elapsed time tracking
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import threading
import time
import traceback
import zipfile
from collections import deque
from pathlib import Path

import gs_pipeline


class TrainingState:
    IDLE = "idle"
    TRAINING = "training"
    DONE = "done"
    ERROR = "error"


_ITER_PATTERNS = [
    re.compile(r"Step\s+(\d+)/(\d+)", re.IGNORECASE),
    re.compile(r"Iter(?:ation)?\s+(\d+)/(\d+)", re.IGNORECASE),
    re.compile(r"\[(\d+)/(\d+)\]"),
    re.compile(r"step\s*=\s*(\d+).*?max_steps\s*=\s*(\d+)", re.IGNORECASE),
]


class TrainingManager:
    LOG_RING_SIZE = 5000

    def __init__(self, work_dir: Path, iterations: int):
        self.work_dir = work_dir
        self.iterations = iterations
        self.state = TrainingState.IDLE
        self.progress = 0.0
        self.message = "Ready"
        self.capture_dir: Path | None = None
        self.output_ply: Path | None = None
        self.backend_name: str | None = None
        self.current_iteration = 0
        self.total_iterations = 0
        self.start_time: float | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._logs: deque[str] = deque(maxlen=self.LOG_RING_SIZE)
        self._log_counter = 0

    def start_training(self, zip_data: bytes) -> bool:
        with self._lock:
            if self.state == TrainingState.TRAINING:
                return False
            self._cancel.clear()
            self.state = TrainingState.TRAINING
            self.progress = 0.0
            self.message = "Extracting upload..."
            self.output_ply = None
            self.backend_name = None
            self.current_iteration = 0
            self.total_iterations = self.iterations
            self.start_time = time.time()
            self._logs.clear()
            self._log_counter = 0

        self._thread = threading.Thread(target=self._run, args=(zip_data,), daemon=True)
        self._thread.start()
        return True

    def cancel(self):
        self._cancel.set()
        with self._lock:
            if self.state == TrainingState.TRAINING:
                self.state = TrainingState.IDLE
                self.message = "Cancelled"
                self.progress = 0.0
                self.start_time = None

    def get_status(self) -> dict:
        with self._lock:
            elapsed = 0.0
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
            return {
                "state": self.state,
                "progress": round(self.progress, 3),
                "message": self.message,
                "backend": self.backend_name,
                "current_iteration": self.current_iteration,
                "total_iterations": self.total_iterations,
                "elapsed_seconds": round(elapsed, 1),
            }

    def get_logs(self, since_line: int = 0) -> tuple[list[str], int]:
        """Return (lines, next_line_number) for lines after since_line."""
        with self._lock:
            all_logs = list(self._logs)
            start = max(0, since_line - (self._log_counter - len(all_logs)))
            if start < 0:
                start = 0
            lines = all_logs[start:]
            return lines, self._log_counter

    def get_render_images(self) -> list[str]:
        """Return filenames of intermediate render images in the output dir."""
        if self.capture_dir is None:
            return []
        output_dir = self.capture_dir / "output"
        if not output_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg"}
        files = []
        for f in sorted(output_dir.rglob("*")):
            if f.suffix.lower() in exts and f.is_file():
                files.append(str(f.relative_to(output_dir)))
        return files

    def get_render_image_path(self, filename: str) -> Path | None:
        if self.capture_dir is None:
            return None
        path = (self.capture_dir / "output" / filename).resolve()
        output_dir = (self.capture_dir / "output").resolve()
        if not str(path).startswith(str(output_dir)):
            return None
        if path.exists() and path.is_file():
            return path
        return None

    def _log(self, line: str):
        with self._lock:
            self._logs.append(line)
            self._log_counter += 1
        self._parse_iteration(line)

    def _parse_iteration(self, line: str):
        for pattern in _ITER_PATTERNS:
            m = pattern.search(line)
            if m:
                current = int(m.group(1))
                total = int(m.group(2))
                with self._lock:
                    self.current_iteration = current
                    self.total_iterations = total
                    self.progress = 0.2 + 0.8 * (current / max(total, 1))
                    self.message = f"Training... {current}/{total}"
                return

    def _run(self, zip_data: bytes):
        try:
            capture_dir = self.work_dir / "current_run"
            if capture_dir.exists():
                shutil.rmtree(capture_dir)
            capture_dir.mkdir(parents=True)

            with self._lock:
                self.message = "Extracting ZIP..."
                self.progress = 0.05
            self._log("Extracting uploaded ZIP...")

            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                zf.extractall(capture_dir)

            if self._cancel.is_set():
                return

            if not (capture_dir / "frames.jsonl").exists():
                raise FileNotFoundError("frames.jsonl not found in upload")

            with self._lock:
                self.message = "Converting to COLMAP format..."
                self.progress = 0.1
                self.capture_dir = capture_dir
            self._log("Converting to COLMAP format...")

            frames = gs_pipeline.parse_frames(capture_dir)
            gs_pipeline.convert_to_colmap(capture_dir, frames)
            self._log(f"COLMAP conversion done: {len(frames)} frames")

            if self._cancel.is_set():
                return

            backend = self._detect_backend()
            with self._lock:
                self.backend_name = backend
                self.message = f"Training ({self.iterations} iters, {backend})..."
                self.progress = 0.2
            self._log(f"Starting training: backend={backend}, iterations={self.iterations}")

            args = argparse.Namespace(
                iterations=self.iterations,
                gs_repo=None,
            )
            output_dir = gs_pipeline.train(capture_dir, args)

            if self._cancel.is_set():
                return

            trained_ply = output_dir / "splat.ply"
            if not trained_ply.exists():
                candidates = list(output_dir.rglob("*.ply"))
                trained_ply = candidates[0] if candidates else None

            if trained_ply and trained_ply.exists():
                with self._lock:
                    self.state = TrainingState.DONE
                    self.progress = 1.0
                    self.message = f"Training complete: {trained_ply.name}"
                    self.output_ply = trained_ply
                self._log(f"Training complete! Output: {trained_ply.name}")
            else:
                with self._lock:
                    self.state = TrainingState.ERROR
                    self.message = "Training finished but no PLY output found"
                self._log("ERROR: No PLY output found after training")

        except Exception as e:
            traceback.print_exc()
            with self._lock:
                self.state = TrainingState.ERROR
                self.message = str(e)
            self._log(f"ERROR: {e}")

    @staticmethod
    def _detect_backend() -> str:
        try:
            import msplat  # noqa: F401
            return "msplat"
        except ImportError:
            pass
        try:
            import gsplat  # noqa: F401
            return "gsplat"
        except ImportError:
            pass
        return "3dgs"
