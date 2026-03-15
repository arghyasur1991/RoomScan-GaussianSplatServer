# RoomScan-GaussianSplatServer

PC training server + web dashboard for [QuestRoomScan](https://github.com/arghyasur1991/QuestRoomScan) Gaussian Splatting pipeline.

## Overview

Receives captured keyframes and point cloud from a Meta Quest headset, runs COLMAP conversion + Gaussian Splat training, and serves the trained model back. Includes a real-time web dashboard for monitoring training, browsing keyframes, and interactively viewing point clouds and trained splats.

## Quick Start

```bash
# Install Python dependencies
cd server
pip install -r requirements.txt

# Install frontend dependencies
cd ../web
npm install

# Development (two terminals)
cd server && uvicorn main:app --host 0.0.0.0 --port 8420
cd web && npm run dev

# Production (single process)
./scripts/build.sh
cd server && uvicorn main:app --host 0.0.0.0 --port 8420
```

## Architecture

- **Backend**: FastAPI (Python) — serves Quest API + dashboard API + WebSocket
- **Frontend**: React + Vite + TypeScript + Tailwind CSS
- **3D Rendering**: Three.js (point cloud) + gaussian-splats-3d (trained splat)
- **Training**: msplat (Metal/Apple Silicon), gsplat (CUDA), or original 3DGS

## API

### Quest Endpoints (backward-compatible)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload ZIP of keyframes + point cloud, starts training |
| GET | `/status` | Training status JSON |
| GET | `/download` | Download trained PLY |
| POST | `/cancel` | Cancel in-progress training |

### Dashboard Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Extended status with iteration count, elapsed time |
| GET | `/api/keyframes` | List keyframe metadata |
| GET | `/api/keyframes/{id}/image` | Serve keyframe JPEG |
| GET | `/api/pointcloud` | Serve points3d.ply |
| GET | `/api/renders` | List intermediate render images |
| GET | `/api/splat` | Serve trained splat PLY |
| GET | `/api/logs` | SSE stream of training logs |
| WS | `/ws/status` | Real-time training status push |
