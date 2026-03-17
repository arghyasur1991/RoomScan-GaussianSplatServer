"""
Training manager — orchestrates GS training and exposes rich status.

Extracted from gs_server.py with additions:
- Log ring buffer (subprocess stdout/stderr captured)
- Iteration progress parsing from training backend output
- Intermediate render image tracking
- Elapsed time tracking
- Run history: timestamped folders with symlink to current
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import threading
import time
import traceback
import zipfile
from collections import deque
from datetime import datetime
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

# msplat outputs "step=  100  splats= 35,071  ms=4.9" — no total in the line
_MSPLAT_STEP_PATTERN = re.compile(r"step\s*=\s*(\d+)")

# msplat step metrics: "step=  100  splats= 35,071  ms=4.9"
_MSPLAT_METRICS_PATTERN = re.compile(
    r"step\s*=\s*(\d+)\s+splats\s*=\s*([\d,]+)\s+ms\s*=\s*([\d.]+)"
)

# msplat per-view eval: "  [1/24] IMG_001.jpg  PSNR=28.5  SSIM=0.89  L1=0.03"
_EVAL_VIEW_PATTERN = re.compile(
    r"\[(\d+)/(\d+)\]\s+(\S+)\s+PSNR=([\d.]+)\s+SSIM=([\d.]+)\s+L1=([\d.]+)"
)

# msplat final eval summary: "  PSNR:  27.14  SSIM:  0.853  L1:  0.021  Gaussians: 3510000"
_EVAL_SUMMARY_PATTERN = re.compile(
    r"PSNR:\s+([\d.]+)\s+SSIM:\s+([\d.]+)\s+L1:\s+([\d.]+)\s+Gaussians:\s+([\d,]+)"
)


class TrainingManager:
    LOG_RING_SIZE = 5000
    MAX_RUNS = 10

    def __init__(self, work_dir: Path, iterations: int):
        self.work_dir = work_dir
        self.runs_dir = work_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.current_link = work_dir / "current_run"
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
        self._elapsed_final: float | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._logs: deque[str] = deque(maxlen=self.LOG_RING_SIZE)
        self._log_counter = 0
        self._run_name: str | None = None
        # --- Metrics tracking ---
        self._step_metrics: list[dict] = []   # [{step, splats, ms_per_iter, loss}]
        self.eval_psnr: float | None = None
        self.eval_ssim: float | None = None
        self.eval_l1: float | None = None
        self.gaussian_count: int | None = None
        self._eval_per_view: list[dict] = []  # [{name, psnr, ssim, l1}]

        self._migrate_legacy_run()
        self._restore_previous_run()

    def _migrate_legacy_run(self):
        """Migrate old flat current_run/ directory into runs/ with a symlink."""
        if self.current_link.is_symlink():
            return
        if not self.current_link.is_dir():
            return
        # Old-style directory exists — move it into runs/
        name = "migrated_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.runs_dir / name
        print(f"Migrating legacy current_run/ → runs/{name}/")
        shutil.move(str(self.current_link), str(dest))
        self.current_link.symlink_to(dest)

    def _restore_previous_run(self):
        """If a previous run's output exists on disk, restore 'done' state."""
        if not self.current_link.exists():
            return
        run_dir = self.current_link.resolve()
        ply = run_dir / "output" / "splat.ply"
        if ply.exists():
            self.state = TrainingState.DONE
            self.progress = 1.0
            self.output_ply = ply
            self.capture_dir = run_dir
            self._run_name = run_dir.name
            self.message = f"Previous run restored: {run_dir.name}"
            self._log(f"Restored previous training output: {ply}")

    def _create_run_dir(self) -> Path:
        """Create a new timestamped run directory and update the current_run symlink."""
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)

        if self.current_link.is_symlink() or self.current_link.exists():
            self.current_link.unlink()
        self.current_link.symlink_to(run_dir)

        self._run_name = name
        self._cleanup_old_runs()
        return run_dir

    def _cleanup_old_runs(self):
        """Remove oldest runs beyond MAX_RUNS."""
        runs = sorted(
            [d for d in self.runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        while len(runs) > self.MAX_RUNS:
            oldest = runs.pop(0)
            try:
                shutil.rmtree(oldest)
                self._log(f"Cleaned up old run: {oldest.name}")
            except Exception as e:
                self._log(f"Failed to clean up {oldest.name}: {e}")

    def start_training(self, zip_data: bytes, iterations_override: int | None = None) -> bool:
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
            self._run_iterations = iterations_override or self.iterations
            self.total_iterations = self._run_iterations
            self.start_time = time.time()
            self._elapsed_final = None
            self._logs.clear()
            self._log_counter = 0
            self._step_metrics = []
            self.eval_psnr = None
            self.eval_ssim = None
            self.eval_l1 = None
            self.gaussian_count = None
            self._eval_per_view = []

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
                self._elapsed_final = None
                self.start_time = None

    def get_status(self) -> dict:
        with self._lock:
            if self._elapsed_final is not None:
                elapsed = self._elapsed_final
            elif self.start_time is not None:
                elapsed = time.time() - self.start_time
            else:
                elapsed = 0.0
            return {
                "state": self.state,
                "progress": round(self.progress, 3),
                "message": self.message,
                "backend": self.backend_name,
                "current_iteration": self.current_iteration,
                "total_iterations": self.total_iterations,
                "elapsed_seconds": round(elapsed, 1),
                "run_name": self._run_name,
                "eval_psnr": self.eval_psnr,
                "eval_ssim": self.eval_ssim,
                "eval_l1": self.eval_l1,
                "gaussian_count": self.gaussian_count,
            }

    def get_runs(self) -> list[dict]:
        """Return metadata for all stored runs, newest first."""
        runs = []
        for d in sorted(self.runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if not d.is_dir():
                continue
            ply = d / "output" / "splat.ply"
            has_ply = ply.exists()
            size_mb = round(ply.stat().st_size / (1024 * 1024), 1) if has_ply else 0
            is_current = (
                self.current_link.is_symlink()
                and self.current_link.resolve() == d.resolve()
            )
            metrics_file = d / "output" / "metrics.json"
            run_metrics = {}
            if metrics_file.exists():
                try:
                    run_metrics = json.loads(metrics_file.read_text())
                except Exception:
                    pass
            runs.append({
                "name": d.name,
                "has_ply": has_ply,
                "ply_size_mb": size_mb,
                "is_current": is_current,
                "eval_psnr": run_metrics.get("eval_psnr"),
                "eval_ssim": run_metrics.get("eval_ssim"),
                "eval_l1": run_metrics.get("eval_l1"),
                "gaussian_count": run_metrics.get("gaussian_count"),
                "elapsed_seconds": run_metrics.get("elapsed_seconds"),
                "iterations": run_metrics.get("iterations"),
                "backend": run_metrics.get("backend"),
            })
        return runs

    def switch_run(self, run_name: str) -> bool:
        """Switch current_run symlink to a different historical run."""
        target = self.runs_dir / run_name
        if not target.is_dir():
            return False
        ply = target / "output" / "splat.ply"
        if not ply.exists():
            return False

        if self.current_link.is_symlink() or self.current_link.exists():
            self.current_link.unlink()
        self.current_link.symlink_to(target)

        with self._lock:
            self.state = TrainingState.DONE
            self.progress = 1.0
            self.output_ply = ply
            self.capture_dir = target
            self._run_name = run_name
            self.message = f"Switched to run: {run_name}"
            self._elapsed_final = None
            self.start_time = None
        return True

    def delete_run(self, run_name: str) -> bool:
        """Delete a single run directory. Cannot delete while training or if it's the active run."""
        target = self.runs_dir / run_name
        if not target.is_dir():
            return False

        is_current = (
            self.current_link.is_symlink()
            and self.current_link.resolve() == target.resolve()
        )
        if is_current:
            if self.current_link.is_symlink():
                self.current_link.unlink()
            with self._lock:
                self.state = TrainingState.IDLE
                self.progress = 0.0
                self.message = "Ready"
                self.output_ply = None
                self.capture_dir = None
                self._run_name = None
                self._elapsed_final = None
                self.start_time = None

        try:
            shutil.rmtree(target)
            self._log(f"Deleted run: {run_name}")
            return True
        except Exception as e:
            self._log(f"Failed to delete {run_name}: {e}")
            return False

    def delete_all_runs(self) -> int:
        """Delete all run directories. Returns the number deleted."""
        if self.current_link.is_symlink() or self.current_link.exists():
            self.current_link.unlink()

        with self._lock:
            self.state = TrainingState.IDLE
            self.progress = 0.0
            self.message = "Ready"
            self.output_ply = None
            self.capture_dir = None
            self._run_name = None
            self._elapsed_final = None
            self.start_time = None

        count = 0
        for d in list(self.runs_dir.iterdir()):
            if d.is_dir():
                try:
                    shutil.rmtree(d)
                    count += 1
                except Exception as e:
                    self._log(f"Failed to delete {d.name}: {e}")
        self._log(f"Cleared all runs ({count} deleted)")
        return count

    def retrain_run(self, run_name: str, iterations_override: int | None = None) -> bool:
        """Re-run training on an existing run's capture data (skips upload/extract)."""
        with self._lock:
            if self.state == TrainingState.TRAINING:
                return False

        target = self.runs_dir / run_name
        if not target.is_dir():
            return False
        if not (target / "frames.jsonl").exists():
            return False

        with self._lock:
            self._cancel.clear()
            self.state = TrainingState.TRAINING
            self.progress = 0.0
            self.message = "Retraining..."
            self.output_ply = None
            self.backend_name = None
            self.current_iteration = 0
            self._run_iterations = iterations_override or self.iterations
            self.total_iterations = self._run_iterations
            self.start_time = time.time()
            self._elapsed_final = None
            self._logs.clear()
            self._log_counter = 0
            self._step_metrics = []
            self.eval_psnr = None
            self.eval_ssim = None
            self.eval_l1 = None
            self.gaussian_count = None
            self._eval_per_view = []
            self.capture_dir = target
            self._run_name = run_name

        if self.current_link.is_symlink() or self.current_link.exists():
            self.current_link.unlink()
        self.current_link.symlink_to(target)

        self._thread = threading.Thread(
            target=self._run_retrain, args=(target,), daemon=True
        )
        self._thread.start()
        return True

    def _run_retrain(self, capture_dir: Path):
        """Training thread for retrain — skips ZIP extraction, re-uses capture data."""
        try:
            # Clean previous output so we get a fresh train
            output_dir = capture_dir / "output"
            if output_dir.exists():
                shutil.rmtree(output_dir)
                self._log("Cleared previous output directory")

            with self._lock:
                self.message = "Converting to COLMAP format..."
                self.progress = 0.1
            self._log("Converting to COLMAP format...")

            frames = gs_pipeline.parse_frames(capture_dir)
            gs_pipeline.convert_to_colmap(capture_dir, frames)
            self._log(f"COLMAP conversion done: {len(frames)} frames")

            if self._cancel.is_set():
                return

            backend = self._detect_backend()
            iters = self._run_iterations
            with self._lock:
                self.backend_name = backend
                self.message = f"Retraining ({iters} iters, {backend})..."
                self.progress = 0.2
            self._log(f"Starting retrain: backend={backend}, iterations={iters}")

            args = argparse.Namespace(iterations=iters, gs_repo=None)
            output_dir = gs_pipeline.train(capture_dir, args, log_fn=self._log)

            if self._cancel.is_set():
                return

            trained_ply = output_dir / "splat.ply"
            if not trained_ply.exists():
                candidates = list(output_dir.rglob("*.ply"))
                trained_ply = candidates[0] if candidates else None

            if trained_ply and trained_ply.exists():
                with self._lock:
                    self._freeze_elapsed()
                    self.state = TrainingState.DONE
                    self.progress = 1.0
                    self.message = f"Retrain complete: {trained_ply.name}"
                    self.output_ply = trained_ply
                self._log(f"Retrain complete! Output: {trained_ply.name}")
                self._save_run_metrics()
            else:
                with self._lock:
                    self._freeze_elapsed()
                    self.state = TrainingState.ERROR
                    self.message = "Retrain finished but no PLY output found"
                self._log("ERROR: No PLY output found after retrain")

        except Exception as e:
            traceback.print_exc()
            with self._lock:
                self._freeze_elapsed()
                self.state = TrainingState.ERROR
                self.message = str(e)
            self._log(f"ERROR: {e}")

    def _freeze_elapsed(self):
        """Snapshot elapsed time so it stops ticking. Must be called under _lock."""
        if self.start_time is not None:
            self._elapsed_final = time.time() - self.start_time

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

    def get_metrics(self) -> dict:
        """Return training metrics time-series and eval results."""
        with self._lock:
            result = {"steps": list(self._step_metrics)}
            if self.eval_psnr is not None:
                result["eval"] = {
                    "psnr": self.eval_psnr,
                    "ssim": self.eval_ssim,
                    "l1": self.eval_l1,
                    "gaussian_count": self.gaussian_count,
                    "per_view": list(self._eval_per_view),
                }
            return result

    def get_checkpoints(self) -> list[dict]:
        """Return list of intermediate checkpoint PLY files for the current run."""
        if self.capture_dir is None:
            return []
        output_dir = self.capture_dir / "output"
        if not output_dir.exists():
            return []
        checkpoints = []
        for f in sorted(output_dir.glob("splat*.ply")):
            name = f.stem  # e.g. "splat_1400" or "splat"
            step_match = re.search(r"_(\d+)$", name)
            step = int(step_match.group(1)) if step_match else self.total_iterations
            size_mb = round(f.stat().st_size / (1024 * 1024), 1)
            checkpoints.append({
                "step": step,
                "filename": f.name,
                "size_mb": size_mb,
            })
        checkpoints.sort(key=lambda c: c["step"])
        return checkpoints

    def get_checkpoint_path(self, filename: str) -> Path | None:
        """Return path to a specific checkpoint PLY, or None if not found."""
        if self.capture_dir is None:
            return None
        path = (self.capture_dir / "output" / filename).resolve()
        output_dir = (self.capture_dir / "output").resolve()
        if not str(path).startswith(str(output_dir)):
            return None
        if path.exists() and path.is_file() and path.suffix.lower() == ".ply":
            return path
        return None

    def _log(self, line: str):
        with self._lock:
            self._logs.append(line)
            self._log_counter += 1
        self._parse_iteration(line)

    def _parse_iteration(self, line: str):
        # Check eval patterns FIRST — they contain [N/M] which would
        # otherwise be caught by the generic _ITER_PATTERNS
        m = _EVAL_VIEW_PATTERN.search(line)
        if m:
            with self._lock:
                self._eval_per_view.append({
                    "name": m.group(3),
                    "psnr": float(m.group(4)),
                    "ssim": float(m.group(5)),
                    "l1": float(m.group(6)),
                })
                self.message = f"Evaluating... {m.group(1)}/{m.group(2)}"
            return

        m = _EVAL_SUMMARY_PATTERN.search(line)
        if m:
            with self._lock:
                self.eval_psnr = float(m.group(1))
                self.eval_ssim = float(m.group(2))
                self.eval_l1 = float(m.group(3))
                self.gaussian_count = int(m.group(4).replace(",", ""))
            return

        m = _MSPLAT_METRICS_PATTERN.search(line)
        if m:
            step = int(m.group(1))
            splats = int(m.group(2).replace(",", ""))
            ms = float(m.group(3))
            with self._lock:
                self.current_iteration = step
                total = self.total_iterations or self.iterations
                self.progress = 0.2 + 0.8 * (step / max(total, 1))
                self.message = f"Training... {step}/{total}"
                self._step_metrics.append({
                    "step": step, "splats": splats, "ms_per_iter": ms,
                })
            return

        m = _MSPLAT_STEP_PATTERN.search(line)
        if m:
            current = int(m.group(1))
            with self._lock:
                self.current_iteration = current
                total = self.total_iterations or self.iterations
                self.progress = 0.2 + 0.8 * (current / max(total, 1))
                self.message = f"Training... {current}/{total}"
            return

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
            capture_dir = self._create_run_dir()

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
            iters = self._run_iterations
            with self._lock:
                self.backend_name = backend
                self.message = f"Training ({iters} iters, {backend})..."
                self.progress = 0.2
            self._log(f"Starting training: backend={backend}, iterations={iters}")

            args = argparse.Namespace(
                iterations=iters,
                gs_repo=None,
            )
            output_dir = gs_pipeline.train(capture_dir, args, log_fn=self._log)

            if self._cancel.is_set():
                return

            trained_ply = output_dir / "splat.ply"
            if not trained_ply.exists():
                candidates = list(output_dir.rglob("*.ply"))
                trained_ply = candidates[0] if candidates else None

            if trained_ply and trained_ply.exists():
                with self._lock:
                    self._freeze_elapsed()
                    self.state = TrainingState.DONE
                    self.progress = 1.0
                    self.message = f"Training complete: {trained_ply.name}"
                    self.output_ply = trained_ply
                self._log(f"Training complete! Output: {trained_ply.name}")
                self._save_run_metrics()
            else:
                with self._lock:
                    self._freeze_elapsed()
                    self.state = TrainingState.ERROR
                    self.message = "Training finished but no PLY output found"
                self._log("ERROR: No PLY output found after training")

        except Exception as e:
            traceback.print_exc()
            with self._lock:
                self._freeze_elapsed()
                self.state = TrainingState.ERROR
                self.message = str(e)
            self._log(f"ERROR: {e}")

    def _save_run_metrics(self):
        """Persist eval metrics to metrics.json in the current run's output dir."""
        if self.capture_dir is None:
            return
        output_dir = self.capture_dir / "output"
        if not output_dir.exists():
            return
        metrics = {}
        with self._lock:
            if self.eval_psnr is not None:
                metrics["eval_psnr"] = self.eval_psnr
                metrics["eval_ssim"] = self.eval_ssim
                metrics["eval_l1"] = self.eval_l1
                metrics["gaussian_count"] = self.gaussian_count
            if self._elapsed_final is not None:
                metrics["elapsed_seconds"] = round(self._elapsed_final, 1)
            metrics["iterations"] = self._run_iterations
            metrics["backend"] = self.backend_name
        if metrics:
            try:
                (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            except Exception:
                pass

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
