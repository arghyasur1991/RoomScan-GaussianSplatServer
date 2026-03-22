"""
Server-side HQ texture refinement via pure PyTorch tensor optimization.
Works on MPS (Apple Silicon), CUDA (NVIDIA), and CPU — no custom kernels.

Pipeline:
  1. Load mesh + UVs + keyframes from extracted ZIP
  2. Pre-compute UV-to-pixel correspondences (NumPy)
  3. Initialize atlas from per-texel best-view projection
  4. Differentiable optimization: gather + L2 loss + backprop
  5. Export refined atlas as PNG
"""

from __future__ import annotations

import json
import struct
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("texture_refine")
logger.setLevel(logging.INFO)


def get_device():
    """Auto-select best available PyTorch device."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except ImportError:
        pass
    import torch
    return torch.device("cpu")


def load_mesh(mesh_path: Path) -> dict:
    """Load refined_mesh.bin (binary format from Unity persistence)."""
    with open(mesh_path, "rb") as f:
        vert_count = struct.unpack("<i", f.read(4))[0]
        idx_count = struct.unpack("<i", f.read(4))[0]
        atlas_w = struct.unpack("<i", f.read(4))[0]
        atlas_h = struct.unpack("<i", f.read(4))[0]

        # Each vertex: float3 pos + float3 norm + float2 uv = 32 bytes
        verts_raw = np.frombuffer(f.read(vert_count * 32), dtype=np.float32).reshape(vert_count, 8)
        positions = verts_raw[:, 0:3]
        normals = verts_raw[:, 3:6]
        uvs = verts_raw[:, 6:8]

        indices = np.frombuffer(f.read(idx_count * 4), dtype=np.int32).reshape(-1, 3)

    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "indices": indices,
        "atlas_w": atlas_w,
        "atlas_h": atlas_h,
    }


def load_keyframes(run_dir: Path) -> list[dict]:
    """Load keyframe metadata and images from frames.jsonl + images/."""
    manifest = run_dir / "frames.jsonl"
    if not manifest.exists():
        return []

    from PIL import Image

    keyframes = []
    for line in manifest.read_text().strip().split("\n"):
        if not line.strip():
            continue
        meta = json.loads(line)
        img_path = run_dir / "images" / f"{meta['id']:06d}.jpg"
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        pixels = np.array(img, dtype=np.float32) / 255.0  # H x W x 3

        sw = meta.get("sw", img.width)
        sh = meta.get("sh", img.height)
        crop_x = (sw - img.width) * 0.5
        crop_y = (sh - img.height) * 0.5

        keyframes.append({
            "pixels": pixels,
            "width": img.width,
            "height": img.height,
            "position": np.array([meta["px"], meta["py"], meta["pz"]], dtype=np.float32),
            "rotation": np.array([meta["qx"], meta["qy"], meta["qz"], meta["qw"]], dtype=np.float32),
            "fx": meta["fx"], "fy": meta["fy"],
            "cx": meta["cx"] - crop_x, "cy": meta["cy"] - crop_y,
        })

    return keyframes


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def build_view_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build 4x4 world-to-camera view matrix from position + quaternion."""
    R = quat_to_rotation_matrix(rotation)
    T = position
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R.T
    view[:3, 3] = -R.T @ T
    return view


def project_to_screen(world_pts: np.ndarray, view_mat: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points to Nx2 screen coords. Returns (screen_xy, cam_z)."""
    ones = np.ones((world_pts.shape[0], 1), dtype=np.float32)
    pts4 = np.hstack([world_pts, ones])
    cam = (view_mat @ pts4.T).T  # Nx4
    cam_z = cam[:, 2]
    valid = cam_z > 0.001
    screen_x = np.where(valid, fx * cam[:, 0] / np.maximum(cam_z, 0.001) + cx, -1)
    screen_y = np.where(valid, fy * cam[:, 1] / np.maximum(cam_z, 0.001) + cy, -1)
    return np.stack([screen_x, screen_y], axis=1), cam_z


def compute_correspondences(mesh: dict, keyframes: list[dict]) -> dict:
    """
    Pre-compute which atlas texels correspond to which keyframe pixels.
    Returns per-texel best-view info for initialization + multi-view index lists
    for optimization.
    """
    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]
    uvs = mesh["uvs"]  # Nx2 in [0,1]
    positions = mesh["positions"]
    normals = mesh["normals"]
    indices = mesh["indices"]

    texel_count = atlas_w * atlas_h
    best_score = np.full(texel_count, -1.0, dtype=np.float32)
    best_color = np.zeros((texel_count, 3), dtype=np.float32)

    # Multi-view correspondences for optimization
    texel_indices_list = []  # list of (texel_idx, kf_idx, px, py, score)

    for kf_idx, kf in enumerate(keyframes):
        view_mat = build_view_matrix(kf["position"], kf["rotation"])
        cam_pos = kf["position"]

        tri_count = indices.shape[0]
        for t in range(tri_count):
            i0, i1, i2 = indices[t]
            p0, p1, p2 = positions[i0], positions[i1], positions[i2]
            n0 = normals[i0]

            face_normal = np.cross(p1 - p0, p2 - p0)
            norm_len = np.linalg.norm(face_normal)
            if norm_len < 1e-6:
                face_normal = n0
            else:
                face_normal = face_normal / norm_len

            centroid = (p0 + p1 + p2) / 3.0
            view_dir = cam_pos - centroid
            dist = np.linalg.norm(view_dir)
            if dist < 0.01:
                continue
            view_dir /= dist

            dot = float(np.dot(face_normal, view_dir))
            if dot <= 0.05:
                continue

            # Project vertices to screen
            verts = np.stack([p0, p1, p2])
            screen_xy, cam_z = project_to_screen(verts, view_mat,
                                                  kf["fx"], kf["fy"], kf["cx"], kf["cy"])

            if np.all(cam_z <= 0):
                continue

            # Check frustum
            in_frustum = np.any(
                (screen_xy[:, 0] >= 0) & (screen_xy[:, 0] < kf["width"]) &
                (screen_xy[:, 1] >= 0) & (screen_xy[:, 1] < kf["height"])
            )
            if not in_frustum:
                continue

            score = dot / max(dist, 0.1)

            # UV coordinates in atlas pixel space
            uv0 = uvs[i0] * [atlas_w, atlas_h]
            uv1 = uvs[i1] * [atlas_w, atlas_h]
            uv2 = uvs[i2] * [atlas_w, atlas_h]

            # Rasterize triangle in UV space (simplified — just sample vertices and centroid)
            for bw in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1/3, 1/3, 1/3)]:
                uv_px = bw[0] * uv0 + bw[1] * uv1 + bw[2] * uv2
                scr = bw[0] * screen_xy[0] + bw[1] * screen_xy[1] + bw[2] * screen_xy[2]

                tx, ty = int(uv_px[0]), int(uv_px[1])
                if tx < 0 or tx >= atlas_w or ty < 0 or ty >= atlas_h:
                    continue

                sx, sy = int(round(scr[0])), int(round(scr[1]))
                if sx < 0 or sx >= kf["width"] or sy < 0 or sy >= kf["height"]:
                    continue

                texel_idx = ty * atlas_w + tx

                # PIL row 0 = top of image, but projection Y increases upward
                py = kf["height"] - 1 - sy

                # Best-view initialization
                if score > best_score[texel_idx]:
                    best_score[texel_idx] = score
                    best_color[texel_idx] = kf["pixels"][py, sx]

                texel_indices_list.append((texel_idx, kf_idx, sx, py, score))

        if kf_idx % 20 == 0:
            logger.info(f"Correspondences: {kf_idx + 1}/{len(keyframes)} keyframes")

    return {
        "best_color": best_color,
        "best_score": best_score,
        "correspondences": texel_indices_list,
        "atlas_w": atlas_w,
        "atlas_h": atlas_h,
    }


def optimize_atlas(correspondences: dict, keyframes: list[dict],
                   num_steps: int = 300, lr: float = 0.01) -> np.ndarray:
    """
    Differentiable texture optimization using pure PyTorch ops.
    Works on MPS, CUDA, or CPU.
    """
    import torch
    import torch.nn.functional as F

    device = get_device()
    logger.info(f"Optimizing atlas on device: {device}")

    atlas_w = correspondences["atlas_w"]
    atlas_h = correspondences["atlas_h"]

    # Initialize atlas from best-view projection
    initial = correspondences["best_color"].reshape(atlas_h, atlas_w, 3)
    atlas = torch.nn.Parameter(
        torch.from_numpy(initial.copy()).to(device)
    )

    # Build correspondence tensors
    corr = correspondences["correspondences"]
    if len(corr) == 0:
        logger.warning("No correspondences found, returning initial atlas")
        return initial

    corr_arr = np.array(corr, dtype=np.float32)
    texel_ids = torch.from_numpy(corr_arr[:, 0].astype(np.int64)).to(device)
    kf_ids = corr_arr[:, 1].astype(np.int32)
    pixel_xs = corr_arr[:, 2].astype(np.int32)
    pixel_ys = corr_arr[:, 3].astype(np.int32)

    # Build target colors from keyframes
    target_colors = np.zeros((len(corr), 3), dtype=np.float32)
    for i in range(len(corr)):
        ki = int(kf_ids[i])
        px, py = int(pixel_xs[i]), int(pixel_ys[i])
        target_colors[i] = keyframes[ki]["pixels"][py, px]

    target = torch.from_numpy(target_colors).to(device)

    # Convert texel indices to (row, col) for atlas gathering
    texel_rows = texel_ids // atlas_w
    texel_cols = texel_ids % atlas_w

    optimizer = torch.optim.Adam([atlas], lr=lr)

    batch_size = min(len(corr), 50000)

    for step in range(num_steps):
        # Random batch for large correspondence sets
        if len(corr) > batch_size:
            perm = torch.randperm(len(corr), device=device)[:batch_size]
            b_rows = texel_rows[perm]
            b_cols = texel_cols[perm]
            b_target = target[perm]
        else:
            b_rows = texel_rows
            b_cols = texel_cols
            b_target = target

        predicted = atlas[b_rows, b_cols]  # Bx3
        loss = F.mse_loss(predicted, b_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            atlas.data.clamp_(0, 1)

        if step % 50 == 0 or step == num_steps - 1:
            logger.info(f"Step {step}/{num_steps}, loss={loss.item():.6f}")

    result = atlas.detach().cpu().numpy()
    return result


def refine_texture(run_dir: Path, num_steps: int = 300) -> Path:
    """
    Full server-side texture refinement pipeline.
    Returns path to the output PNG atlas.
    """
    logger.info(f"Starting texture refinement from {run_dir}")

    # Load mesh if available
    mesh_path = run_dir / "refined_mesh.bin"
    if not mesh_path.exists():
        raise FileNotFoundError(f"No refined_mesh.bin found in {run_dir}")

    mesh = load_mesh(mesh_path)
    logger.info(f"Loaded mesh: {mesh['positions'].shape[0]} verts, "
                f"atlas {mesh['atlas_w']}x{mesh['atlas_h']}")

    # Load keyframes
    keyframes = load_keyframes(run_dir)
    if not keyframes:
        raise ValueError("No keyframes found")
    logger.info(f"Loaded {len(keyframes)} keyframes")

    # Pre-compute correspondences
    logger.info("Computing UV-to-pixel correspondences...")
    correspondences = compute_correspondences(mesh, keyframes)
    n_corr = len(correspondences['correspondences'])
    filled = int(np.sum(correspondences['best_score'] > 0))
    total_texels = correspondences['atlas_w'] * correspondences['atlas_h']
    logger.info(f"Found {n_corr} correspondences, {filled}/{total_texels} texels covered "
                f"({100*filled/max(total_texels,1):.1f}%)")

    # Optimize
    logger.info(f"Running differentiable optimization ({num_steps} steps)...")
    atlas_float = optimize_atlas(correspondences, keyframes, num_steps=num_steps)

    # Convert to uint8 and save as PNG
    atlas_uint8 = (np.clip(atlas_float, 0, 1) * 255).astype(np.uint8)

    from PIL import Image
    img = Image.fromarray(atlas_uint8, "RGB")
    out_path = run_dir / "hq_atlas.png"
    img.save(out_path, "PNG")
    logger.info(f"Saved HQ atlas to {out_path}")

    return out_path
