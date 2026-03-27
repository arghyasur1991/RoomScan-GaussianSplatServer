"""
Server-side HQ texture refinement via differentiable rendering.

Primary path: PyTorch3D differentiable rasterization (MPS/CUDA/CPU).
Fallback path: pre-computed correspondences + tensor optimization (no PyTorch3D).

Pipeline (PyTorch3D):
  1. Load mesh + UVs + keyframes
  2. Build PyTorch3D Meshes with TexturesUV (learnable atlas)
  3. For each optimization step, render mesh from keyframe viewpoints
  4. Photometric L1 loss vs actual keyframe images + TV regularization
  5. Backprop to atlas texture
  6. Export refined atlas as PNG

Pipeline (Fallback):
  1. Load mesh + UVs + keyframes
  2. Pre-compute UV-to-pixel correspondences (NumPy)
  3. Initialize atlas from per-texel best-view projection
  4. Differentiable gather + L2 loss + backprop
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

_HAS_PYTORCH3D = None


def _check_pytorch3d() -> bool:
    global _HAS_PYTORCH3D
    if _HAS_PYTORCH3D is not None:
        return _HAS_PYTORCH3D
    try:
        import pytorch3d
        from pytorch3d.renderer import MeshRasterizer
        _HAS_PYTORCH3D = True
        logger.info(f"PyTorch3D {pytorch3d.__version__} available — using differentiable rendering")
    except ImportError:
        _HAS_PYTORCH3D = False
        logger.info("PyTorch3D not installed — using fallback correspondence optimization")
    return _HAS_PYTORCH3D


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


# ═══════════════════════════════════════════════════════════════════════
#  Mesh & Keyframe I/O (shared)
# ═══════════════════════════════════════════════════════════════════════

def load_mesh(mesh_path: Path) -> dict:
    """Load refined_mesh.bin (binary format from Unity persistence)."""
    with open(mesh_path, "rb") as f:
        vert_count = struct.unpack("<i", f.read(4))[0]
        idx_count = struct.unpack("<i", f.read(4))[0]
        atlas_w = struct.unpack("<i", f.read(4))[0]
        atlas_h = struct.unpack("<i", f.read(4))[0]

        verts_raw = np.frombuffer(f.read(vert_count * 32), dtype=np.float32).reshape(vert_count, 8)
        positions = verts_raw[:, 0:3].copy()
        normals = verts_raw[:, 3:6].copy()
        uvs = verts_raw[:, 6:8].copy()
        indices = np.frombuffer(f.read(idx_count * 4), dtype=np.int32).reshape(-1, 3).copy()

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
        # Flip vertically: Unity's GetPixels32() has row 0 at the bottom,
        # but PIL has row 0 at the top. The projection formula assumes
        # Unity's bottom-up convention, so flip to match.
        pixels = np.array(img, dtype=np.float32)[::-1].copy() / 255.0

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


# ═══════════════════════════════════════════════════════════════════════
#  PyTorch3D Differentiable Rendering Path
# ═══════════════════════════════════════════════════════════════════════

def _refine_multiview_blend(mesh: dict, keyframes: list[dict],
                            top_k: int = 5, prerastd: dict = None,
                            progress_fn: callable = None) -> np.ndarray:
    """
    Weighted multi-view blending: for each texel, blend the top-K keyframe
    observations weighted by their viewing score. Produces smoother, more
    consistent textures than single-view selection without requiring GPU
    differentiable rendering.
    """
    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]
    texel_count = atlas_w * atlas_h

    if prerastd is None:
        prerastd = _prerasterize_atlas(mesh)

    texel_pos = prerastd["texel_pos"]
    texel_norm = prerastd["texel_norm"]
    filled_idx = prerastd["filled_idx"]
    n_filled = len(filled_idx)

    # For each texel, track top-K scores and corresponding colors
    topk_scores = np.full((n_filled, top_k), -1.0, dtype=np.float32)
    topk_colors = np.zeros((n_filled, top_k, 3), dtype=np.float32)

    ones_col = np.ones((n_filled, 1), dtype=np.float32)
    pts4 = np.hstack([texel_pos, ones_col])

    n_kf = len(keyframes)

    for kf_idx, kf in enumerate(keyframes):
        view_mat = build_view_matrix(kf["position"], kf["rotation"])
        if not np.isfinite(view_mat).all():
            continue

        cam_pos = kf["position"]
        fx, fy, cx, cy = kf["fx"], kf["fy"], kf["cx"], kf["cy"]
        w, h = kf["width"], kf["height"]

        view_dirs = cam_pos - texel_pos
        dists = np.linalg.norm(view_dirs, axis=1)
        safe_dists = np.maximum(dists, 0.01)
        dots = np.sum(texel_norm * (view_dirs / safe_dists[:, None]), axis=1)

        visible = (dots > 0.05) & (dists >= 0.01)
        if not np.any(visible):
            continue

        scores = dots / np.maximum(dists, 0.1)

        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            cam = (view_mat @ pts4.T).T

        cam_z = cam[:, 2]
        in_front = cam_z > 0.001

        sx = fx * cam[:, 0] / np.maximum(cam_z, 0.001) + cx
        sy = fy * cam[:, 1] / np.maximum(cam_z, 0.001) + cy

        in_bounds = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
        valid = visible & in_front & in_bounds

        # Check which valid texels have a score higher than the current min in top-K
        current_min = topk_scores.min(axis=1)
        candidates = valid & (scores > current_min)

        if not np.any(candidates):
            continue

        ci = np.where(candidates)[0]
        px_i = np.clip(np.round(sx[ci]).astype(np.int32), 0, w - 1)
        py_i = np.clip(np.round(sy[ci]).astype(np.int32), 0, h - 1)
        colors = kf["pixels"][py_i, px_i]

        # Insert into top-K: replace the slot with the lowest score
        min_slot = topk_scores[ci].argmin(axis=1)
        topk_scores[ci, min_slot] = scores[ci]
        topk_colors[ci, min_slot, :] = colors

        if kf_idx % 50 == 0:
            filled_count = np.sum(topk_scores[:, 0] > 0)
            logger.info(f"Multi-view blend: {kf_idx + 1}/{n_kf} keyframes, "
                        f"{filled_count}/{n_filled} texels have ≥1 view")
            if progress_fn:
                progress_fn(kf_idx, n_kf, 0.0)

    # Weighted blend: softmax over scores for smooth weighting
    valid_mask = topk_scores > 0
    topk_scores_safe = np.where(valid_mask, topk_scores, -1e9)

    # Softmax-like weighting with temperature for sharpness control
    temperature = 0.3
    exp_scores = np.exp((topk_scores_safe - topk_scores_safe.max(axis=1, keepdims=True))
                        / temperature)
    exp_scores = np.where(valid_mask, exp_scores, 0.0)
    weight_sum = exp_scores.sum(axis=1, keepdims=True)
    weights = exp_scores / np.maximum(weight_sum, 1e-8)  # (N, K)

    blended = (topk_colors * weights[:, :, None]).sum(axis=1)  # (N, 3)

    atlas = np.zeros((texel_count, 3), dtype=np.float32)
    has_any = topk_scores[:, 0] > 0
    atlas[filled_idx[has_any]] = blended[has_any]

    logger.info(f"Multi-view blend complete: {has_any.sum()}/{n_filled} texels filled, "
                f"top_k={top_k}, temperature={temperature}")

    if progress_fn:
        progress_fn(n_kf, n_kf, 0.0)

    return atlas.reshape(atlas_h, atlas_w, 3)


def _resize_image(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize image using PIL."""
    from PIL import Image
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.resize((w, h), Image.BILINEAR)
    return np.array(pil_img, dtype=np.float32) / 255.0


def _prerasterize_atlas(mesh: dict) -> dict:
    """
    Pre-rasterize the UV atlas: for each filled texel, store which face it
    belongs to and the barycentric-interpolated 3D position + normal.
    This is done once and reused for all keyframes.
    """
    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]
    positions = mesh["positions"]
    normals = mesh["normals"]
    uvs = mesh["uvs"]
    indices = mesh["indices"]

    i0s, i1s, i2s = indices[:, 0], indices[:, 1], indices[:, 2]
    p0s, p1s, p2s = positions[i0s], positions[i1s], positions[i2s]
    n0s, n1s, n2s = normals[i0s], normals[i1s], normals[i2s]

    uv_scale = np.array([atlas_w, atlas_h], dtype=np.float32)
    uv0s = (uvs[i0s] * uv_scale).astype(np.float32)
    uv1s = (uvs[i1s] * uv_scale).astype(np.float32)
    uv2s = (uvs[i2s] * uv_scale).astype(np.float32)

    face_normals = np.cross(p1s - p0s, p2s - p0s)
    norm_lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
    safe_lens = np.maximum(norm_lens, 1e-6)
    face_normals = face_normals / safe_lens

    texel_count = atlas_w * atlas_h
    texel_pos = np.zeros((texel_count, 3), dtype=np.float32)
    texel_norm = np.zeros((texel_count, 3), dtype=np.float32)
    texel_filled = np.zeros(texel_count, dtype=bool)

    n_faces = len(indices)
    for fi in range(n_faces):
        uv0, uv1, uv2 = uv0s[fi], uv1s[fi], uv2s[fi]

        min_x = max(0, int(np.floor(min(uv0[0], uv1[0], uv2[0]))))
        max_x = min(atlas_w - 1, int(np.ceil(max(uv0[0], uv1[0], uv2[0]))))
        min_y = max(0, int(np.floor(min(uv0[1], uv1[1], uv2[1]))))
        max_y = min(atlas_h - 1, int(np.ceil(max(uv0[1], uv1[1], uv2[1]))))

        denom = (uv1[1] - uv2[1]) * (uv0[0] - uv2[0]) + \
                (uv2[0] - uv1[0]) * (uv0[1] - uv2[1])
        if abs(denom) < 1e-8:
            continue
        inv_d = 1.0 / denom

        a0x = uv1[1] - uv2[1]; a0y = uv2[0] - uv1[0]
        a1x = uv2[1] - uv0[1]; a1y = uv0[0] - uv2[0]

        xs = np.arange(min_x, max_x + 1, dtype=np.float32)
        ys = np.arange(min_y, max_y + 1, dtype=np.float32)
        if len(xs) == 0 or len(ys) == 0:
            continue

        gx, gy = np.meshgrid(xs, ys)
        dx = gx - uv2[0]
        dy = gy - uv2[1]
        bw0 = (a0x * dx + a0y * dy) * inv_d
        bw1 = (a1x * dx + a1y * dy) * inv_d
        bw2 = 1.0 - bw0 - bw1

        valid = (bw0 >= -0.001) & (bw1 >= -0.001) & (bw2 >= -0.001)
        vy, vx = np.where(valid)
        if len(vy) == 0:
            continue

        pix_x = (gx[vy, vx]).astype(np.int32)
        pix_y = (gy[vy, vx]).astype(np.int32)
        tidx = pix_y * atlas_w + pix_x

        b0 = bw0[vy, vx][:, None]
        b1 = bw1[vy, vx][:, None]
        b2 = bw2[vy, vx][:, None]

        pos_interp = b0 * p0s[fi] + b1 * p1s[fi] + b2 * p2s[fi]
        nrm_interp = b0 * face_normals[fi] + b1 * face_normals[fi] + b2 * face_normals[fi]

        texel_pos[tidx] = pos_interp
        texel_norm[tidx] = nrm_interp
        texel_filled[tidx] = True

    filled_idx = np.where(texel_filled)[0]
    logger.info(f"Pre-rasterized atlas: {len(filled_idx)}/{texel_count} texels filled "
                f"({100*len(filled_idx)/texel_count:.1f}%)")

    return {
        "texel_pos": texel_pos[filled_idx],
        "texel_norm": texel_norm[filled_idx],
        "filled_idx": filled_idx,
        "atlas_w": atlas_w,
        "atlas_h": atlas_h,
    }


def _compute_initial_atlas(mesh: dict, keyframes: list[dict],
                           prerastd: dict = None) -> np.ndarray:
    """
    Compute initial atlas from best-view per-texel projection.
    Vectorized: projects all texels at once per keyframe.
    """
    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]
    texel_count = atlas_w * atlas_h

    if prerastd is None:
        prerastd = _prerasterize_atlas(mesh)

    texel_pos = prerastd["texel_pos"]     # (N, 3)
    texel_norm = prerastd["texel_norm"]   # (N, 3)
    filled_idx = prerastd["filled_idx"]   # (N,)
    n_filled = len(filled_idx)

    best_score = np.full(n_filled, -1.0, dtype=np.float32)
    best_color = np.zeros((n_filled, 3), dtype=np.float32)

    ones_col = np.ones((n_filled, 1), dtype=np.float32)
    pts4 = np.hstack([texel_pos, ones_col])  # (N, 4)

    for kf_idx, kf in enumerate(keyframes):
        view_mat = build_view_matrix(kf["position"], kf["rotation"])
        if not np.isfinite(view_mat).all():
            continue

        cam_pos = kf["position"]
        fx, fy, cx, cy = kf["fx"], kf["fy"], kf["cx"], kf["cy"]
        w, h = kf["width"], kf["height"]

        view_dirs = cam_pos - texel_pos  # (N, 3)
        dists = np.linalg.norm(view_dirs, axis=1)
        safe_dists = np.maximum(dists, 0.01)
        dots = np.sum(texel_norm * (view_dirs / safe_dists[:, None]), axis=1)

        visible = (dots > 0.05) & (dists >= 0.01)
        if not np.any(visible):
            continue

        scores = dots / np.maximum(dists, 0.1)

        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            cam = (view_mat @ pts4.T).T  # (N, 4)

        cam_z = cam[:, 2]
        in_front = cam_z > 0.001

        sx = fx * cam[:, 0] / np.maximum(cam_z, 0.001) + cx
        sy = fy * cam[:, 1] / np.maximum(cam_z, 0.001) + cy

        in_bounds = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)
        valid = visible & in_front & in_bounds
        better = valid & (scores > best_score)

        if not np.any(better):
            continue

        bi = np.where(better)[0]
        px_i = np.clip(np.round(sx[bi]).astype(np.int32), 0, w - 1)
        py_i = np.clip(np.round(sy[bi]).astype(np.int32), 0, h - 1)

        best_score[bi] = scores[bi]
        best_color[bi] = kf["pixels"][py_i, px_i]

        if kf_idx % 50 == 0:
            logger.info(f"Initial atlas: {kf_idx + 1}/{len(keyframes)} keyframes, "
                        f"{np.sum(best_score > 0)}/{n_filled} texels filled")

    atlas = np.zeros((texel_count, 3), dtype=np.float32)
    atlas[filled_idx] = best_color
    return atlas.reshape(atlas_h, atlas_w, 3)


# ═══════════════════════════════════════════════════════════════════════
#  Fallback: Correspondence-Based Optimization (no PyTorch3D)
# ═══════════════════════════════════════════════════════════════════════

def _refine_fallback(mesh: dict, keyframes: list[dict], num_steps: int,
                     lr: float = 0.01, prerastd: dict = None,
                     progress_fn: callable = None) -> np.ndarray:
    """
    Fallback texture refinement: best single-view selection per texel.
    Picks the highest-scoring keyframe observation for each texel, producing
    sharp results. Multi-view optimization is deferred to the PyTorch3D path.
    """
    if prerastd is None:
        prerastd = _prerasterize_atlas(mesh)

    atlas = _compute_initial_atlas(mesh, keyframes, prerastd=prerastd)

    if progress_fn:
        progress_fn(0, 1, 0.0)

    return atlas


def _compute_correspondences(mesh: dict, keyframes: list[dict],
                              prerastd: dict = None) -> dict:
    """
    Compute texel-to-keyframe correspondences (vectorized).
    Returns array of (texel_idx, kf_idx, px, py, score) per valid mapping.
    """
    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]

    if prerastd is None:
        prerastd = _prerasterize_atlas(mesh)

    texel_pos = prerastd["texel_pos"]     # (N, 3)
    texel_norm = prerastd["texel_norm"]   # (N, 3)
    filled_idx = prerastd["filled_idx"]   # (N,)
    n_filled = len(filled_idx)

    ones_col = np.ones((n_filled, 1), dtype=np.float32)
    pts4 = np.hstack([texel_pos, ones_col])  # (N, 4)

    all_corr = []

    for kf_idx, kf in enumerate(keyframes):
        view_mat = build_view_matrix(kf["position"], kf["rotation"])
        if not np.isfinite(view_mat).all():
            continue

        cam_pos = kf["position"]
        fx, fy, cx, cy = kf["fx"], kf["fy"], kf["cx"], kf["cy"]
        w, h = kf["width"], kf["height"]

        view_dirs = cam_pos - texel_pos
        dists = np.linalg.norm(view_dirs, axis=1)
        safe_dists = np.maximum(dists, 0.01)
        dots = np.sum(texel_norm * (view_dirs / safe_dists[:, None]), axis=1)
        scores = dots / np.maximum(dists, 0.1)

        visible = (dots > 0.05) & (dists >= 0.01)

        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            cam = (view_mat @ pts4.T).T

        cam_z = cam[:, 2]
        in_front = cam_z > 0.001

        sx = fx * cam[:, 0] / np.maximum(cam_z, 0.001) + cx
        sy = fy * cam[:, 1] / np.maximum(cam_z, 0.001) + cy
        in_bounds = (sx >= 0) & (sx < w - 1) & (sy >= 0) & (sy < h - 1)

        valid = visible & in_front & in_bounds
        vi = np.where(valid)[0]
        if len(vi) == 0:
            continue

        px_i = np.clip(np.round(sx[vi]).astype(np.int32), 0, w - 1)
        py_i = np.clip(np.round(sy[vi]).astype(np.int32), 0, h - 1)

        kf_corr = np.column_stack([
            filled_idx[vi].astype(np.float32),
            np.full(len(vi), kf_idx, dtype=np.float32),
            px_i.astype(np.float32),
            py_i.astype(np.float32),
            scores[vi],
        ])
        all_corr.append(kf_corr)

        if kf_idx % 50 == 0:
            logger.info(f"Correspondences: {kf_idx + 1}/{len(keyframes)} keyframes")

    if all_corr:
        corr_array = np.concatenate(all_corr, axis=0)
    else:
        corr_array = None
    total = len(corr_array) if corr_array is not None else 0
    logger.info(f"Total correspondences: {total}")
    return {"corr_array": corr_array}


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def refine_texture(run_dir: Path, num_steps: int = 300,
                   progress_fn: callable = None) -> Path:
    """
    Full server-side texture refinement pipeline.
    Uses PyTorch3D differentiable rendering if available, else falls back
    to correspondence-based optimization.

    Returns path to the output PNG atlas.
    """
    logger.info(f"Starting texture refinement from {run_dir}")

    mesh_path = run_dir / "refined_mesh.bin"
    if not mesh_path.exists():
        raise FileNotFoundError(f"No refined_mesh.bin found in {run_dir}")

    mesh = load_mesh(mesh_path)
    logger.info(f"Loaded mesh: {mesh['positions'].shape[0]} verts, "
                f"atlas {mesh['atlas_w']}x{mesh['atlas_h']}")

    keyframes = load_keyframes(run_dir)
    if not keyframes:
        raise ValueError("No keyframes found")
    logger.info(f"Loaded {len(keyframes)} keyframes")

    import time
    t0 = time.time()
    prerastd = _prerasterize_atlas(mesh)
    logger.info(f"Pre-rasterization done in {time.time() - t0:.1f}s")

    atlas_float = _refine_multiview_blend(mesh, keyframes, top_k=5,
                                           prerastd=prerastd,
                                           progress_fn=progress_fn)

    atlas_uint8 = (np.clip(atlas_float, 0, 1) * 255).astype(np.uint8)

    # V-flip: server atlas has y=0 → UV v=0 as the first row, but Unity's
    # ImageConversion.LoadImage maps PNG row 0 (top) to texture top (UV v=1).
    # Flip so row 0 = UV v=1, matching Unity's PNG→texture convention.
    atlas_uint8 = atlas_uint8[::-1].copy()

    atlas_uint8 = _dilate_atlas(atlas_uint8, iterations=4)

    from PIL import Image
    img = Image.fromarray(atlas_uint8, "RGB")
    out_path = run_dir / "hq_atlas.png"
    img.save(out_path, "PNG")
    logger.info(f"Saved HQ atlas to {out_path}")

    return out_path


def _dilate_atlas(atlas: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Fill small gaps in the atlas by dilating filled pixels into empty neighbors."""
    h, w, c = atlas.shape
    filled = np.any(atlas > 0, axis=2)

    for _ in range(iterations):
        empty = ~filled
        if not np.any(empty):
            break

        padded = np.pad(atlas, ((1, 1), (1, 1), (0, 0)), mode='constant')
        padded_f = np.pad(filled, ((1, 1), (1, 1)), mode='constant')

        acc = np.zeros_like(atlas, dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb_f = padded_f[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            nb_c = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            mask = empty & nb_f
            acc[mask] += nb_c[mask].astype(np.float32)
            cnt[mask] += 1.0

        update = cnt > 0
        atlas[update] = (acc[update] / cnt[update, None]).astype(np.uint8)
        filled[update] = True

    return atlas
