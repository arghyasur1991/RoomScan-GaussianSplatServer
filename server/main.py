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
import struct
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from training_manager import TrainingManager, TrainingState

WORK_DIR = Path("gs_server_work").resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ITERATIONS = 7000

manager = TrainingManager(WORK_DIR, DEFAULT_ITERATIONS)

app = FastAPI(title="Sentience GS Training Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
#  Quest API (backward-compatible with GSplatServerClient.cs)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload(request: Request):
    body = await request.body()
    if len(body) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty body"})
    if len(body) > 2 * 1024 * 1024 * 1024:
        return JSONResponse(status_code=413, content={"error": "Upload too large"})

    if not manager.start_training(body):
        return JSONResponse(status_code=409, content={"error": "Training already in progress"})

    return {"status": "started"}


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


def _strip_ply_sh(src: Path) -> Path:
    """Create a lightweight PLY with only SH degree-0 (strips f_rest_*).
    Reduces ~79MB to ~14MB for 335k gaussians."""
    dst = src.with_name("splat_web.ply")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst

    with open(src, "rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        props: list[tuple[str, str]] = []
        vertex_count = 0
        for line in header_lines:
            text = line.decode().strip()
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property"):
                parts = text.split()
                props.append((parts[1], parts[2]))

        keep_indices: list[int] = []
        keep_props: list[tuple[str, str]] = []
        for i, (dtype, name) in enumerate(props):
            if not name.startswith("f_rest_"):
                keep_indices.append(i)
                keep_props.append((dtype, name))

        src_stride = len(props) * 4
        dst_stride = len(keep_indices) * 4

        new_header = "ply\nformat binary_little_endian 1.0\n"
        new_header += f"element vertex {vertex_count}\n"
        for dtype, name in keep_props:
            new_header += f"property {dtype} {name}\n"
        new_header += "end_header\n"

        fmt_src = "<" + "f" * len(props)
        fmt_dst = "<" + "f" * len(keep_indices)

        data_start = f.tell()
        with open(dst, "wb") as out:
            out.write(new_header.encode())
            for _ in range(vertex_count):
                vertex = struct.unpack(fmt_src, f.read(src_stride))
                out.write(struct.pack(fmt_dst, *(vertex[i] for i in keep_indices)))

    return dst


@app.api_route("/api/splat", methods=["GET", "HEAD"])
async def api_splat():
    s = manager.get_status()
    if s["state"] != TrainingState.DONE or manager.output_ply is None:
        return JSONResponse(status_code=404, content={"error": "No trained splat available"})

    ply_path = manager.output_ply
    if not ply_path.exists():
        return JSONResponse(status_code=404, content={"error": "Splat PLY not found"})

    web_ply = _strip_ply_sh(ply_path)

    return FileResponse(
        path=web_ply,
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
    except WebSocketDisconnect:
        pass
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
