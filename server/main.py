"""
FastAPI training server for QuestRoomScan Gaussian Splatting.

Replaces gs_server.py with a modern async server that serves:
- Quest API (backward-compatible): /upload, /status, /download, /cancel
- Dashboard API: /api/* endpoints for the web dashboard
- WebSocket: /ws/status for real-time training status push
- Static files: built React app served from /
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles

from training_manager import TrainingManager, TrainingState

WORK_DIR = Path("gs_server_work").resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ITERATIONS = 7000

manager = TrainingManager(WORK_DIR, DEFAULT_ITERATIONS)

app = FastAPI(title="QuestRoomScan GS Training Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoCacheAPIMiddleware(BaseHTTPMiddleware):
    """Prevent browser caching on /api/ data endpoints so run-switching works."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


app.add_middleware(NoCacheAPIMiddleware)


# ═══════════════════════════════════════════════════════════════════════
#  Mesh Enhancement API
# ═══════════════════════════════════════════════════════════════════════

MESH_ENHANCE_DIR = WORK_DIR / "mesh_enhance_runs"
MESH_ENHANCE_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/enhance-mesh")
async def enhance_mesh_endpoint(
    request: Request,
    smooth_iterations: int = 5,
    smooth_method: str = "bilateral",
    enable_plane_snap: bool = True,
    plane_threshold: float = 0.02,
    snap_threshold: float = 0.03,
    max_planes: int = 10,
):
    """Upload refined_mesh.bin, return enhanced version with smoothing + plane snapping."""
    body = await request.body()
    if len(body) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty body"})

    logger = logging.getLogger("mesh_enhance")

    try:
        from mesh_enhance import enhance_mesh

        name = datetime.now().strftime("meshenhance_%Y%m%d_%H%M%S")
        run_dir = MESH_ENHANCE_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)

        input_path = run_dir / "refined_mesh.bin"
        input_path.write_bytes(body)
        logger.info(f"Received mesh ({len(body)} bytes)")

        output_path = enhance_mesh(
            input_path,
            output_path=run_dir / "enhanced_mesh.bin",
            smooth_iterations=smooth_iterations,
            smooth_method=smooth_method,
            enable_plane_snap=enable_plane_snap,
            plane_threshold=plane_threshold,
            plane_snap_threshold=snap_threshold,
            max_planes=max_planes,
        )

        return FileResponse(
            path=output_path,
            media_type="application/octet-stream",
            filename="enhanced_mesh.bin",
        )

    except Exception as e:
        logger.exception("Mesh enhancement failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════
#  Atlas Inpainting API
# ═══════════════════════════════════════════════════════════════════════

INPAINT_DIR = WORK_DIR / "inpaint_runs"
INPAINT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/inpaint-atlas")
async def inpaint_atlas_endpoint(request: Request, method: str = "auto"):
    """Upload an atlas PNG with gaps, return inpainted version."""
    body = await request.body()
    if len(body) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty body"})

    logger = logging.getLogger("atlas_inpaint")

    try:
        from atlas_inpaint import inpaint_atlas

        name = datetime.now().strftime("inpaint_%Y%m%d_%H%M%S")
        run_dir = INPAINT_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)

        input_path = run_dir / "atlas_input.png"
        input_path.write_bytes(body)

        output_path = inpaint_atlas(input_path, method=method)

        return FileResponse(
            path=output_path,
            media_type="image/png",
            filename="atlas_inpainted.png",
        )

    except Exception as e:
        logger.exception("Atlas inpainting failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════
#  Texture Refinement API
# ═══════════════════════════════════════════════════════════════════════

REFINE_DIR = WORK_DIR / "refine_runs"
REFINE_DIR.mkdir(parents=True, exist_ok=True)
MAX_REFINE_RUNS = 10

_refine_state = {
    "state": "idle",     # idle | processing | done | error
    "progress": 0.0,
    "message": "Ready",
    "run_dir": None,
    "output_path": None,
}
_refine_lock = threading.Lock()
_refine_thread: threading.Thread | None = None


def _run_refine_thread(run_dir: Path, num_steps: int):
    from texture_refine import refine_texture as do_refine
    logger = logging.getLogger("texture_refine")

    def on_progress(step, total, loss):
        with _refine_lock:
            _refine_state["progress"] = step / max(total, 1)
            _refine_state["message"] = f"Optimizing: step {step}/{total}, loss={loss:.6f}"

    try:
        with _refine_lock:
            _refine_state["message"] = "Loading data..."
            _refine_state["progress"] = 0.05

        out_path = do_refine(run_dir, num_steps, progress_fn=on_progress)

        with _refine_lock:
            _refine_state["state"] = "done"
            _refine_state["progress"] = 1.0
            _refine_state["message"] = "Refinement complete"
            _refine_state["output_path"] = out_path
        logger.info(f"Texture refinement done: {out_path}")

    except Exception as e:
        logger.exception("Texture refinement failed")
        with _refine_lock:
            _refine_state["state"] = "error"
            _refine_state["message"] = str(e)


@app.post("/refine-texture")
async def refine_texture_start(request: Request, steps: int | None = None):
    """Start async HQ texture refinement. Returns immediately."""
    global _refine_thread

    with _refine_lock:
        if _refine_state["state"] == "processing":
            return JSONResponse(status_code=409, content={"error": "Refinement already in progress"})

    body = await request.body()
    if len(body) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty body"})

    import zipfile, io, shutil
    logger = logging.getLogger("texture_refine")

    try:
        name = datetime.now().strftime("texrefine_%Y%m%d_%H%M%S")
        run_dir = REFINE_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting refinement data to {run_dir}")

        zf = zipfile.ZipFile(io.BytesIO(body))
        zf.extractall(run_dir)

        runs = sorted([d for d in REFINE_DIR.iterdir() if d.is_dir()], key=lambda d: d.name)
        while len(runs) > MAX_REFINE_RUNS:
            oldest = runs.pop(0)
            if oldest != run_dir:
                shutil.rmtree(oldest, ignore_errors=True)
                logger.info(f"Cleaned up old refine run: {oldest.name}")

        num_steps = steps if steps and steps > 0 else 300

        with _refine_lock:
            _refine_state["state"] = "processing"
            _refine_state["progress"] = 0.0
            _refine_state["message"] = "Extracting data..."
            _refine_state["run_dir"] = run_dir
            _refine_state["output_path"] = None

        _refine_thread = threading.Thread(
            target=_run_refine_thread, args=(run_dir, num_steps), daemon=True
        )
        _refine_thread.start()

        return {"status": "started", "run": name}

    except Exception as e:
        logger.exception("Failed to start texture refinement")
        with _refine_lock:
            _refine_state["state"] = "error"
            _refine_state["message"] = str(e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/refine-texture/status")
async def refine_texture_status():
    """Poll refinement progress."""
    with _refine_lock:
        return {
            "state": _refine_state["state"],
            "progress": _refine_state["progress"],
            "message": _refine_state["message"],
        }


@app.get("/refine-texture/result")
async def refine_texture_result():
    """Download the HQ atlas PNG once refinement is done."""
    with _refine_lock:
        if _refine_state["state"] != "done":
            return JSONResponse(status_code=404, content={
                "error": f"No result available (state={_refine_state['state']})"
            })
        out_path = _refine_state["output_path"]

    if out_path is None or not out_path.exists():
        return JSONResponse(status_code=404, content={"error": "Output file not found"})

    return FileResponse(path=out_path, media_type="image/png", filename="hq_atlas.png")


# ═══════════════════════════════════════════════════════════════════════
#  Atlas Enhancement API (super-resolution)
# ═══════════════════════════════════════════════════════════════════════

ENHANCE_DIR = WORK_DIR / "enhance_runs"
ENHANCE_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/enhance-atlas")
async def enhance_atlas_endpoint(request: Request, scale: int = 2, inpaint: bool = True):
    """Upload a PNG atlas, return the SR-upscaled (and optionally inpainted) version."""
    body = await request.body()
    if len(body) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty body"})

    logger = logging.getLogger("atlas_enhance")

    try:
        from atlas_enhance import enhance_atlas as do_enhance

        name = datetime.now().strftime("enhance_%Y%m%d_%H%M%S")
        run_dir = ENHANCE_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)

        input_path = run_dir / "atlas_input.png"
        input_path.write_bytes(body)
        logger.info(f"Received atlas ({len(body)} bytes), scale={scale}, inpaint={inpaint}")

        output_path = run_dir / "atlas_enhanced.png"
        do_enhance(input_path, output_path, scale=scale, inpaint=inpaint)

        return FileResponse(
            path=output_path,
            media_type="image/png",
            filename="atlas_enhanced.png",
        )

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return JSONResponse(status_code=501, content={"error": str(e)})
    except Exception as e:
        logger.exception("Atlas enhancement failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════
#  Quest API (backward-compatible with GSplatServerClient.cs)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload(request: Request, iterations: int | None = None):
    body = await request.body()
    if len(body) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty body"})
    if len(body) > 2 * 1024 * 1024 * 1024:
        return JSONResponse(status_code=413, content={"error": "Upload too large"})

    iters = iterations if iterations and iterations > 0 else None
    if not manager.start_training(body, iterations_override=iters):
        return JSONResponse(status_code=409, content={"error": "Training already in progress"})

    return {"status": "started", "iterations": iters or manager.iterations}


@app.get("/status")
async def status():
    s = manager.get_status()
    return {
        "state": s["state"],
        "progress": s["progress"],
        "message": s["message"],
    }


@app.get("/download")
async def download():
    s = manager.get_status()
    if s["state"] != TrainingState.DONE or manager.output_ply is None:
        return JSONResponse(status_code=404, content={"error": "No trained model available"})

    ply_path = manager.output_ply
    if not ply_path.exists():
        return JSONResponse(status_code=404, content={"error": "PLY file not found on disk"})

    return FileResponse(
        path=ply_path,
        media_type="application/octet-stream",
        filename=ply_path.name,
    )


@app.post("/cancel")
async def cancel():
    manager.cancel()
    return {"status": "cancelled"}


# ═══════════════════════════════════════════════════════════════════════
#  Dashboard API (/api/*)
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status():
    return manager.get_status()


@app.get("/api/runs")
async def api_runs():
    return {"runs": manager.get_runs()}


@app.post("/api/runs/{run_name}/activate")
async def api_activate_run(run_name: str):
    if manager.state == TrainingState.TRAINING:
        return JSONResponse(status_code=409, content={"error": "Cannot switch while training"})
    if manager.switch_run(run_name):
        return {"status": "ok", "run_name": run_name}
    return JSONResponse(status_code=404, content={"error": f"Run '{run_name}' not found or has no PLY"})


@app.post("/api/runs/{run_name}/retrain")
async def api_retrain_run(run_name: str, iterations: int | None = None):
    if manager.state == TrainingState.TRAINING:
        return JSONResponse(status_code=409, content={"error": "Training already in progress"})
    iters = iterations if iterations and iterations > 0 else None
    if manager.retrain_run(run_name, iterations_override=iters):
        return {"status": "started", "run_name": run_name, "iterations": iters or manager.iterations}
    return JSONResponse(status_code=404, content={"error": f"Run '{run_name}' not found or missing capture data"})


@app.delete("/api/runs/{run_name}")
async def api_delete_run(run_name: str):
    if manager.state == TrainingState.TRAINING:
        return JSONResponse(status_code=409, content={"error": "Cannot delete while training"})
    if manager.delete_run(run_name):
        return {"status": "ok", "deleted": run_name}
    return JSONResponse(status_code=404, content={"error": f"Run '{run_name}' not found"})


@app.delete("/api/runs")
async def api_delete_all_runs():
    if manager.state == TrainingState.TRAINING:
        return JSONResponse(status_code=409, content={"error": "Cannot delete while training"})
    count = manager.delete_all_runs()
    return {"status": "ok", "deleted_count": count}


@app.get("/api/keyframes")
async def api_keyframes():
    if manager.capture_dir is None:
        return {"keyframes": [], "count": 0}

    manifest = manager.capture_dir / "frames.jsonl"
    if not manifest.exists():
        return {"keyframes": [], "count": 0}

    frames = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            frame["image_url"] = f"/api/keyframes/{frame['id']}/image"
            frames.append(frame)

    by_id = {}
    for fr in frames:
        by_id[fr["id"]] = fr
    unique = [by_id[k] for k in sorted(by_id.keys())]

    return {"keyframes": unique, "count": len(unique)}


@app.get("/api/keyframes/{frame_id}/image")
async def api_keyframe_image(frame_id: int):
    if manager.capture_dir is None:
        return JSONResponse(status_code=404, content={"error": "No capture data"})

    img_path = manager.capture_dir / "images" / f"{frame_id:06d}.jpg"
    if not img_path.exists():
        return JSONResponse(status_code=404, content={"error": "Image not found"})

    return FileResponse(path=img_path, media_type="image/jpeg")


@app.get("/api/pointcloud")
async def api_pointcloud():
    if manager.capture_dir is None:
        return JSONResponse(status_code=404, content={"error": "No capture data"})

    ply_path = manager.capture_dir / "points3d.ply"
    if not ply_path.exists():
        return JSONResponse(status_code=404, content={"error": "Point cloud not found"})

    return FileResponse(
        path=ply_path,
        media_type="application/octet-stream",
        filename="points3d.ply",
    )


@app.get("/api/renders")
async def api_renders():
    images = manager.get_render_images()
    return {"renders": images, "count": len(images)}


@app.get("/api/renders/{filename:path}")
async def api_render_image(filename: str):
    path = manager.get_render_image_path(filename)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "Render not found"})
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path=path, media_type=media)


@app.api_route("/api/splat", methods=["GET", "HEAD"])
async def api_splat():
    s = manager.get_status()
    if s["state"] != TrainingState.DONE or manager.output_ply is None:
        return JSONResponse(status_code=404, content={"error": "No trained splat available"})

    ply_path = manager.output_ply
    if not ply_path.exists():
        return JSONResponse(status_code=404, content={"error": "Splat PLY not found"})

    return FileResponse(
        path=ply_path,
        media_type="application/octet-stream",
        filename="splat.ply",
    )


@app.api_route("/api/splat/full", methods=["GET", "HEAD"])
async def api_splat_full():
    """Serve the full PLY with all SH coefficients."""
    s = manager.get_status()
    if s["state"] != TrainingState.DONE or manager.output_ply is None:
        return JSONResponse(status_code=404, content={"error": "No trained splat available"})

    ply_path = manager.output_ply
    if not ply_path.exists():
        return JSONResponse(status_code=404, content={"error": "Splat PLY not found"})

    return FileResponse(
        path=ply_path,
        media_type="application/octet-stream",
        filename="splat.ply",
    )


@app.get("/api/logs")
async def api_logs(request: Request):
    """SSE endpoint streaming training log lines."""
    async def event_stream():
        last_line = 0
        while True:
            if await request.is_disconnected():
                break
            lines, counter = manager.get_logs(since_line=last_line)
            for line in lines:
                yield f"data: {json.dumps({'line': line})}\n\n"
            last_line = counter
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════
#  WebSocket: real-time training status
# ═══════════════════════════════════════════════════════════════════════

@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            status_data = manager.get_status()
            await websocket.send_json(status_data)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
#  Static file serving (production: built React app)
# ═══════════════════════════════════════════════════════════════════════

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = static_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(static_dir / "index.html")


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="GS Training Server")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--work-dir", default="gs_server_work")
    args = parser.parse_args()

    global manager
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    manager = TrainingManager(work, args.iterations)

    print(f"[gs_server] Listening on {args.host}:{args.port}")
    print(f"[gs_server] Work directory: {work}")
    print(f"[gs_server] Training iterations: {args.iterations}")
    print(f"[gs_server] Dashboard: http://localhost:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
