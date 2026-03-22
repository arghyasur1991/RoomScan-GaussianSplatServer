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
    Vectorized: all triangles processed per keyframe in batched NumPy ops.
    """
    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]
    uvs = mesh["uvs"]
    positions = mesh["positions"]
    normals = mesh["normals"]
    indices = mesh["indices"]  # Tx3

    T = indices.shape[0]
    texel_count = atlas_w * atlas_h
    best_score = np.full(texel_count, -1.0, dtype=np.float32)
    best_color = np.zeros((texel_count, 3), dtype=np.float32)
    texel_indices_list = []

    # Pre-compute per-triangle constants (invariant across keyframes)
    i0s, i1s, i2s = indices[:, 0], indices[:, 1], indices[:, 2]
    p0s = positions[i0s]  # Tx3
    p1s = positions[i1s]
    p2s = positions[i2s]
    n0s = normals[i0s]

    face_normals = np.cross(p1s - p0s, p2s - p0s)  # Tx3
    norm_lens = np.linalg.norm(face_normals, axis=1, keepdims=True)  # Tx1
    degenerate = (norm_lens.ravel() < 1e-6)
    safe_lens = np.where(norm_lens < 1e-6, 1.0, norm_lens)
    face_normals = np.where(degenerate[:, None], n0s, face_normals / safe_lens)

    centroids = (p0s + p1s + p2s) / 3.0  # Tx3

    # UV coords in atlas pixel space
    uv_scale = np.array([atlas_w, atlas_h], dtype=np.float32)
    uv0s = uvs[i0s] * uv_scale  # Tx2
    uv1s = uvs[i1s] * uv_scale
    uv2s = uvs[i2s] * uv_scale

    bary_weights = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1/3, 1/3, 1/3],
    ], dtype=np.float32)  # 4x3

    for kf_idx, kf in enumerate(keyframes):
        view_mat = build_view_matrix(kf["position"], kf["rotation"])
        cam_pos = kf["position"]
        fx, fy, cx, cy = kf["fx"], kf["fy"], kf["cx"], kf["cy"]
        w, h = kf["width"], kf["height"]

        # Batch view direction + dot product
        view_dirs = cam_pos - centroids  # Tx3
        dists = np.linalg.norm(view_dirs, axis=1)  # T
        safe_dists = np.maximum(dists, 0.01)
        view_dirs_n = view_dirs / safe_dists[:, None]
        dots = np.sum(face_normals * view_dirs_n, axis=1)  # T

        # Filter: dot > 0.05 and dist >= 0.01
        mask = (dots > 0.05) & (dists >= 0.01)
        if not np.any(mask):
            if kf_idx % 20 == 0:
                logger.info(f"Correspondences: {kf_idx + 1}/{len(keyframes)} keyframes")
            continue

        # Batch project all 3 vertex sets for surviving triangles
        m_idx = np.where(mask)[0]
        m_p0 = p0s[m_idx]  # Mx3
        m_p1 = p1s[m_idx]
        m_p2 = p2s[m_idx]

        all_verts = np.concatenate([m_p0, m_p1, m_p2], axis=0)  # 3Mx3
        screen_all, camz_all = project_to_screen(all_verts, view_mat, fx, fy, cx, cy)
        M = len(m_idx)
        s0 = screen_all[:M]      # Mx2
        s1 = screen_all[M:2*M]
        s2 = screen_all[2*M:]
        z0 = camz_all[:M]
        z1 = camz_all[M:2*M]
        z2 = camz_all[2*M:]

        # Frustum: at least one vertex in-bounds
        in_bounds = lambda s: (s[:, 0] >= 0) & (s[:, 0] < w) & (s[:, 1] >= 0) & (s[:, 1] < h)
        frustum_ok = in_bounds(s0) | in_bounds(s1) | in_bounds(s2)
        any_z_ok = (z0 > 0) | (z1 > 0) | (z2 > 0)
        keep = frustum_ok & any_z_ok

        if not np.any(keep):
            if kf_idx % 20 == 0:
                logger.info(f"Correspondences: {kf_idx + 1}/{len(keyframes)} keyframes")
            continue

        k_idx = np.where(keep)[0]
        k_global = m_idx[k_idx]
        k_s0 = s0[k_idx]
        k_s1 = s1[k_idx]
        k_s2 = s2[k_idx]
        k_dots = dots[k_global]
        k_dists = np.maximum(dists[k_global], 0.1)
        scores = k_dots / k_dists  # K

        k_uv0 = uv0s[k_global]
        k_uv1 = uv1s[k_global]
        k_uv2 = uv2s[k_global]

        pixels = kf["pixels"]

        # For each barycentric weight, compute atlas + screen coords in batch
        for bw0, bw1, bw2 in bary_weights:
            uv_px = bw0 * k_uv0 + bw1 * k_uv1 + bw2 * k_uv2  # Kx2
            scr = bw0 * k_s0 + bw1 * k_s1 + bw2 * k_s2  # Kx2

            tx = uv_px[:, 0].astype(np.int32)
            ty = uv_px[:, 1].astype(np.int32)
            sx = np.round(scr[:, 0]).astype(np.int32)
            sy = np.round(scr[:, 1]).astype(np.int32)

            valid = (
                (tx >= 0) & (tx < atlas_w) &
                (ty >= 0) & (ty < atlas_h) &
                (sx >= 0) & (sx < w) &
                (sy >= 0) & (sy < h)
            )

            v_idx = np.where(valid)[0]
            if len(v_idx) == 0:
                continue

            v_tx = tx[v_idx]
            v_ty = ty[v_idx]
            v_sx = sx[v_idx]
            v_sy = sy[v_idx]
            v_scores = scores[v_idx]

            texel_ids = v_ty * atlas_w + v_tx
            py_arr = h - 1 - v_sy

            # Update best-view initialization
            better = v_scores > best_score[texel_ids]
            b_idx = np.where(better)[0]
            if len(b_idx) > 0:
                b_texels = texel_ids[b_idx]
                b_py = py_arr[b_idx]
                b_sx = v_sx[b_idx]
                best_score[b_texels] = v_scores[b_idx]
                best_color[b_texels] = pixels[b_py, b_sx]

            # Append correspondences
            for j in v_idx:
                texel_indices_list.append((
                    int(texel_ids[j]), kf_idx, int(v_sx[j]), int(py_arr[j]), float(scores[j])
                ))

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
                   num_steps: int = 300, lr: float = 0.01,
                   progress_fn: callable = None) -> np.ndarray:
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
            if progress_fn:
                progress_fn(step, num_steps, loss.item())

    result = atlas.detach().cpu().numpy()
    return result


def refine_texture(run_dir: Path, num_steps: int = 300, progress_fn: callable = None) -> Path:
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
    atlas_float = optimize_atlas(correspondences, keyframes, num_steps=num_steps,
                                  progress_fn=progress_fn)

    # Convert to uint8 and save as PNG
    atlas_uint8 = (np.clip(atlas_float, 0, 1) * 255).astype(np.uint8)

    from PIL import Image
    img = Image.fromarray(atlas_uint8, "RGB")
    out_path = run_dir / "hq_atlas.png"
    img.save(out_path, "PNG")
    logger.info(f"Saved HQ atlas to {out_path}")

    return out_path
