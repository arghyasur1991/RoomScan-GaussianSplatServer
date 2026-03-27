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
        pixels = np.array(img, dtype=np.float32) / 255.0

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

def _refine_pytorch3d(mesh: dict, keyframes: list[dict], num_steps: int,
                      lr: float = 0.01, tv_weight: float = 0.001,
                      render_size: int = 512,
                      progress_fn: callable = None) -> np.ndarray:
    """
    Differentiable rendering optimization using PyTorch3D.
    Renders the mesh from each keyframe viewpoint, compares to the actual
    keyframe image via photometric loss, and backprops to the atlas texture.
    """
    import torch
    import torch.nn.functional as F
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        MeshRasterizer,
        TexturesUV,
    )

    device = get_device()
    logger.info(f"PyTorch3D refinement on {device}")

    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]

    # Unity uses left-handed coords (z forward), PyTorch3D uses right-handed (z backward)
    # Flip Z for positions
    positions = mesh["positions"].copy()
    positions[:, 2] *= -1

    verts = torch.from_numpy(positions).float().unsqueeze(0).to(device)
    faces = torch.from_numpy(mesh["indices"].astype(np.int64)).unsqueeze(0).to(device)

    # UVs: xatlas produces pixel-space UVs, normalize to [0,1]
    uvs_np = mesh["uvs"].copy()
    uvs_np[:, 0] = uvs_np[:, 0] / atlas_w
    uvs_np[:, 1] = uvs_np[:, 1] / atlas_h
    verts_uvs = torch.from_numpy(uvs_np).float().unsqueeze(0).to(device)

    # Initialize atlas from best-view projection (fallback correspondences)
    initial_atlas = _compute_initial_atlas(mesh, keyframes)
    atlas_texture = torch.nn.Parameter(
        torch.from_numpy(initial_atlas).float().unsqueeze(0).to(device)
    )

    textures = TexturesUV(
        maps=atlas_texture,
        faces_uvs=faces,
        verts_uvs=verts_uvs,
    )

    py3d_mesh = Meshes(verts=verts, faces=faces, textures=textures)

    raster_settings = RasterizationSettings(
        image_size=render_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=True,
    )

    optimizer = torch.optim.Adam([atlas_texture], lr=lr)

    n_kf = len(keyframes)
    kf_per_step = min(4, n_kf)

    for step in range(num_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        # Random subset of keyframes per step for efficiency
        kf_indices = np.random.choice(n_kf, kf_per_step, replace=False)

        for ki in kf_indices:
            kf = keyframes[ki]

            # Build PyTorch3D camera from keyframe intrinsics + extrinsics
            R_np = quat_to_rotation_matrix(kf["rotation"])
            # Flip Z axis for right-handed convention
            R_np[:, 2] *= -1
            R_np[2, :] *= -1
            T_np = kf["position"].copy()
            T_np[2] *= -1

            R = torch.from_numpy(R_np.T[np.newaxis]).float().to(device)
            T = torch.from_numpy((-R_np.T @ T_np)[np.newaxis]).float().to(device)

            # Focal length in NDC: PyTorch3D expects focal_length in screen pixels
            # then converts internally based on image_size
            focal = torch.tensor([[kf["fx"], kf["fy"]]], dtype=torch.float32, device=device)
            principal = torch.tensor([[kf["cx"], kf["cy"]]], dtype=torch.float32, device=device)

            cameras = PerspectiveCameras(
                R=R, T=T,
                focal_length=focal,
                principal_point=principal,
                image_size=torch.tensor([[render_size, render_size]], device=device),
                in_ndc=False,
                device=device,
            )

            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

            # Update textures with current atlas
            py3d_mesh.textures = TexturesUV(
                maps=atlas_texture,
                faces_uvs=faces,
                verts_uvs=verts_uvs,
            )

            fragments = rasterizer(py3d_mesh)

            # Sample texture at fragment UVs
            pix_to_face = fragments.pix_to_face[..., 0]  # (1, H, W)
            bary = fragments.bary_coords[..., 0, :]  # (1, H, W, 3)

            valid_mask = pix_to_face >= 0
            face_idx = pix_to_face.clamp(min=0)

            # Get per-pixel UV via barycentric interpolation
            face_verts_uvs = verts_uvs[0][faces[0][face_idx.view(-1)]]  # (N, 3, 2)
            face_verts_uvs = face_verts_uvs.view(1, render_size, render_size, 3, 2)
            bary_expanded = bary.unsqueeze(-1)  # (1, H, W, 3, 1)
            pixel_uvs = (face_verts_uvs * bary_expanded).sum(dim=-2)  # (1, H, W, 2)

            # Sample atlas at these UVs using grid_sample
            # grid_sample expects coords in [-1, 1]
            grid = pixel_uvs * 2.0 - 1.0  # (1, H, W, 2)
            rendered = F.grid_sample(
                atlas_texture.permute(0, 3, 1, 2),  # (1, 3, H_atlas, W_atlas)
                grid,
                mode='bilinear',
                align_corners=True,
                padding_mode='border',
            )  # (1, 3, render_size, render_size)
            rendered = rendered.permute(0, 2, 3, 1)  # (1, H, W, 3)

            # Resize GT keyframe to render_size
            gt_np = kf["pixels"]  # (H_orig, W_orig, 3)
            gt_resized = _resize_image(gt_np, render_size, render_size)
            gt = torch.from_numpy(gt_resized).float().unsqueeze(0).to(device)

            # Photometric loss (only on valid pixels)
            valid_3d = valid_mask.unsqueeze(-1).float()  # (1, H, W, 1)
            photo_loss = (torch.abs(rendered - gt) * valid_3d).sum() / (valid_3d.sum() * 3 + 1e-6)
            total_loss = total_loss + photo_loss

        total_loss = total_loss / kf_per_step

        # Total variation regularization on atlas
        if tv_weight > 0:
            tv_h = torch.abs(atlas_texture[:, 1:, :, :] - atlas_texture[:, :-1, :, :]).mean()
            tv_w = torch.abs(atlas_texture[:, :, 1:, :] - atlas_texture[:, :, :-1, :]).mean()
            total_loss = total_loss + tv_weight * (tv_h + tv_w)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            atlas_texture.data.clamp_(0, 1)

        if step % 50 == 0 or step == num_steps - 1:
            logger.info(f"Step {step}/{num_steps}, loss={total_loss.item():.6f}")
            if progress_fn:
                progress_fn(step, num_steps, total_loss.item())

    result = atlas_texture.detach().squeeze(0).cpu().numpy()
    return result


def _resize_image(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize image using PIL."""
    from PIL import Image
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.resize((w, h), Image.BILINEAR)
    return np.array(pil_img, dtype=np.float32) / 255.0


def _compute_initial_atlas(mesh: dict, keyframes: list[dict]) -> np.ndarray:
    """Compute initial atlas from best-view per-texel projection (NumPy)."""
    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]
    positions = mesh["positions"]
    normals = mesh["normals"]
    uvs = mesh["uvs"]
    indices = mesh["indices"]

    texel_count = atlas_w * atlas_h
    best_score = np.full(texel_count, -1.0, dtype=np.float32)
    best_color = np.zeros((texel_count, 3), dtype=np.float32)

    i0s, i1s, i2s = indices[:, 0], indices[:, 1], indices[:, 2]
    p0s, p1s, p2s = positions[i0s], positions[i1s], positions[i2s]

    face_normals = np.cross(p1s - p0s, p2s - p0s)
    norm_lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
    safe_lens = np.maximum(norm_lens, 1e-6)
    face_normals = face_normals / safe_lens

    centroids = (p0s + p1s + p2s) / 3.0

    uv_scale = np.array([atlas_w, atlas_h], dtype=np.float32)
    uv0s = uvs[i0s] * uv_scale
    uv1s = uvs[i1s] * uv_scale
    uv2s = uvs[i2s] * uv_scale

    for kf_idx, kf in enumerate(keyframes):
        view_mat = build_view_matrix(kf["position"], kf["rotation"])
        if not np.isfinite(view_mat).all():
            continue

        cam_pos = kf["position"]
        view_dirs = cam_pos - centroids
        dists = np.linalg.norm(view_dirs, axis=1)
        safe_dists = np.maximum(dists, 0.01)
        dots = np.sum(face_normals * (view_dirs / safe_dists[:, None]), axis=1)

        mask = (dots > 0.05) & (dists >= 0.01)
        if not np.any(mask):
            continue

        m_idx = np.where(mask)[0]
        scores = dots[m_idx] / np.maximum(dists[m_idx], 0.1)

        fx, fy, cx, cy = kf["fx"], kf["fy"], kf["cx"], kf["cy"]
        w, h = kf["width"], kf["height"]

        for ti_local, ti in enumerate(m_idx):
            score = float(scores[ti_local])
            uv0 = uv0s[ti]
            uv1 = uv1s[ti]
            uv2 = uv2s[ti]

            min_x = max(0, int(np.floor(min(uv0[0], uv1[0], uv2[0]))))
            max_x = min(atlas_w - 1, int(np.ceil(max(uv0[0], uv1[0], uv2[0]))))
            min_y = max(0, int(np.floor(min(uv0[1], uv1[1], uv2[1]))))
            max_y = min(atlas_h - 1, int(np.ceil(max(uv0[1], uv1[1], uv2[1]))))

            denom_val = (uv1[1] - uv2[1]) * (uv0[0] - uv2[0]) + \
                        (uv2[0] - uv1[0]) * (uv0[1] - uv2[1])
            if abs(denom_val) < 1e-8:
                continue
            inv_d = 1.0 / denom_val

            v0, v1, v2 = p0s[ti], p1s[ti], p2s[ti]
            ones = np.ones((3, 1), dtype=np.float32)
            pts4 = np.hstack([np.stack([v0, v1, v2]), ones])

            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                cam = (view_mat @ pts4.T).T
            cam_z = cam[:, 2]
            if np.all(cam_z <= 0):
                continue

            s_pts = np.zeros((3, 2), dtype=np.float32)
            for pi in range(3):
                if cam_z[pi] > 0.001:
                    s_pts[pi, 0] = fx * cam[pi, 0] / cam_z[pi] + cx
                    s_pts[pi, 1] = fy * cam[pi, 1] / cam_z[pi] + cy

            for y in range(min_y, max_y + 1):
                dy = float(y) - uv2[1]
                for x in range(min_x, max_x + 1):
                    dx = float(x) - uv2[0]
                    bw0 = ((uv1[1] - uv2[1]) * dx + (uv2[0] - uv1[0]) * dy) * inv_d
                    bw1 = ((uv2[1] - uv0[1]) * dx + (uv0[0] - uv2[0]) * dy) * inv_d
                    bw2 = 1.0 - bw0 - bw1
                    if bw0 < -0.001 or bw1 < -0.001 or bw2 < -0.001:
                        continue

                    sx = bw0 * s_pts[0, 0] + bw1 * s_pts[1, 0] + bw2 * s_pts[2, 0]
                    sy = bw0 * s_pts[0, 1] + bw1 * s_pts[1, 1] + bw2 * s_pts[2, 1]
                    px_i = int(round(sx))
                    py_i = int(round(sy))
                    if px_i < 0 or px_i >= w or py_i < 0 or py_i >= h:
                        continue

                    texel_idx = y * atlas_w + x
                    if score > best_score[texel_idx]:
                        best_score[texel_idx] = score
                        best_color[texel_idx] = kf["pixels"][py_i, px_i]

        if kf_idx % 20 == 0:
            logger.info(f"Initial atlas: {kf_idx + 1}/{len(keyframes)} keyframes")

    return best_color.reshape(atlas_h, atlas_w, 3)


# ═══════════════════════════════════════════════════════════════════════
#  Fallback: Correspondence-Based Optimization (no PyTorch3D)
# ═══════════════════════════════════════════════════════════════════════

def _refine_fallback(mesh: dict, keyframes: list[dict], num_steps: int,
                     lr: float = 0.01, progress_fn: callable = None) -> np.ndarray:
    """
    Fallback texture optimization using pre-computed correspondences.
    Gathers atlas texels, compares to target keyframe pixels, backprops.
    """
    import torch
    import torch.nn.functional as F

    device = get_device()
    logger.info(f"Fallback optimization on {device}")

    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]

    initial_atlas = _compute_initial_atlas(mesh, keyframes)
    atlas = torch.nn.Parameter(
        torch.from_numpy(initial_atlas.copy()).to(device)
    )

    # Build correspondence list: (texel_idx, kf_idx, px, py, score)
    correspondences = _compute_correspondences(mesh, keyframes)
    corr_arr = correspondences["corr_array"]
    if corr_arr is None or len(corr_arr) == 0:
        logger.warning("No correspondences, returning initial atlas")
        return initial_atlas

    MAX_CORR = 10_000_000
    if len(corr_arr) > MAX_CORR:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(corr_arr), MAX_CORR, replace=False)
        corr_arr = corr_arr[idx]

    n_corr = len(corr_arr)
    texel_ids = torch.from_numpy(corr_arr[:, 0].astype(np.int64)).to(device)
    kf_ids = corr_arr[:, 1].astype(np.int32)
    pixel_xs = corr_arr[:, 2].astype(np.int32)
    pixel_ys = corr_arr[:, 3].astype(np.int32)

    target_colors = np.zeros((n_corr, 3), dtype=np.float32)
    for ki in range(len(keyframes)):
        mask = kf_ids == ki
        if not np.any(mask):
            continue
        m_idx = np.where(mask)[0]
        target_colors[m_idx] = keyframes[ki]["pixels"][pixel_ys[m_idx], pixel_xs[m_idx]]

    target = torch.from_numpy(target_colors).to(device)
    texel_rows = texel_ids // atlas_w
    texel_cols = texel_ids % atlas_w

    optimizer = torch.optim.Adam([atlas], lr=lr)
    batch_size = min(n_corr, 50000)

    for step in range(num_steps):
        if n_corr > batch_size:
            perm = torch.randperm(n_corr, device=device)[:batch_size]
            b_rows = texel_rows[perm]
            b_cols = texel_cols[perm]
            b_target = target[perm]
        else:
            b_rows = texel_rows
            b_cols = texel_cols
            b_target = target

        predicted = atlas[b_rows, b_cols]
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

    return atlas.detach().cpu().numpy()


def _compute_correspondences(mesh: dict, keyframes: list[dict]) -> dict:
    """Pre-compute texel-to-keyframe-pixel correspondences for fallback path."""
    atlas_w, atlas_h = mesh["atlas_w"], mesh["atlas_h"]
    positions = mesh["positions"]
    normals = mesh["normals"]
    uvs = mesh["uvs"]
    indices = mesh["indices"]

    i0s, i1s, i2s = indices[:, 0], indices[:, 1], indices[:, 2]
    p0s, p1s, p2s = positions[i0s], positions[i1s], positions[i2s]

    face_normals = np.cross(p1s - p0s, p2s - p0s)
    norm_lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
    safe_lens = np.maximum(norm_lens, 1e-6)
    n0s = normals[i0s]
    degenerate = (norm_lens.ravel() < 1e-6)
    face_normals = np.where(degenerate[:, None], n0s, face_normals / safe_lens)
    centroids = (p0s + p1s + p2s) / 3.0

    uv_scale = np.array([atlas_w, atlas_h], dtype=np.float32)
    uv0s = uvs[i0s] * uv_scale
    uv1s = uvs[i1s] * uv_scale
    uv2s = uvs[i2s] * uv_scale

    all_corr = []

    for kf_idx, kf in enumerate(keyframes):
        view_mat = build_view_matrix(kf["position"], kf["rotation"])
        if not np.isfinite(view_mat).all():
            continue

        cam_pos = kf["position"]
        fx, fy, cx, cy = kf["fx"], kf["fy"], kf["cx"], kf["cy"]
        w, h = kf["width"], kf["height"]

        view_dirs = cam_pos - centroids
        dists = np.linalg.norm(view_dirs, axis=1)
        safe_dists = np.maximum(dists, 0.01)
        dots = np.sum(face_normals * (view_dirs / safe_dists[:, None]), axis=1)
        mask = (dots > 0.05) & (dists >= 0.01)
        if not np.any(mask):
            continue

        m_idx = np.where(mask)[0]
        scores = dots[m_idx] / np.maximum(dists[m_idx], 0.1)

        for ti_local, ti in enumerate(m_idx):
            score = float(scores[ti_local])
            uv0 = uv0s[ti]
            uv1 = uv1s[ti]
            uv2 = uv2s[ti]

            min_x = max(0, int(np.floor(min(uv0[0], uv1[0], uv2[0]))))
            max_x = min(atlas_w - 1, int(np.ceil(max(uv0[0], uv1[0], uv2[0]))))
            min_y = max(0, int(np.floor(min(uv0[1], uv1[1], uv2[1]))))
            max_y = min(atlas_h - 1, int(np.ceil(max(uv0[1], uv1[1], uv2[1]))))

            denom_val = (uv1[1] - uv2[1]) * (uv0[0] - uv2[0]) + \
                        (uv2[0] - uv1[0]) * (uv0[1] - uv2[1])
            if abs(denom_val) < 1e-8:
                continue
            inv_d = 1.0 / denom_val

            v0, v1, v2 = p0s[ti], p1s[ti], p2s[ti]
            pts = np.stack([v0, v1, v2])
            ones = np.ones((3, 1), dtype=np.float32)
            pts4 = np.hstack([pts, ones])
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                cam = (view_mat @ pts4.T).T
            cam_z = cam[:, 2]
            if np.all(cam_z <= 0):
                continue

            s_pts = np.zeros((3, 2), dtype=np.float32)
            for pi in range(3):
                if cam_z[pi] > 0.001:
                    s_pts[pi, 0] = fx * cam[pi, 0] / cam_z[pi] + cx
                    s_pts[pi, 1] = fy * cam[pi, 1] / cam_z[pi] + cy

            for y in range(min_y, max_y + 1):
                dy = float(y) - uv2[1]
                for x in range(min_x, max_x + 1):
                    dx = float(x) - uv2[0]
                    bw0 = ((uv1[1] - uv2[1]) * dx + (uv2[0] - uv1[0]) * dy) * inv_d
                    bw1 = ((uv2[1] - uv0[1]) * dx + (uv0[0] - uv2[0]) * dy) * inv_d
                    bw2 = 1.0 - bw0 - bw1
                    if bw0 < -0.001 or bw1 < -0.001 or bw2 < -0.001:
                        continue

                    sx = bw0 * s_pts[0, 0] + bw1 * s_pts[1, 0] + bw2 * s_pts[2, 0]
                    sy = bw0 * s_pts[0, 1] + bw1 * s_pts[1, 1] + bw2 * s_pts[2, 1]
                    px_i = int(round(sx))
                    py_i = int(round(sy))
                    if px_i < 0 or px_i >= w or py_i < 0 or py_i >= h:
                        continue

                    texel_idx = y * atlas_w + x
                    all_corr.append([texel_idx, kf_idx, px_i, py_i, score])

        if kf_idx % 20 == 0:
            logger.info(f"Correspondences: {kf_idx + 1}/{len(keyframes)} keyframes")

    corr_array = np.array(all_corr, dtype=np.float32) if all_corr else None
    logger.info(f"Total correspondences: {len(all_corr)}")
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

    if _check_pytorch3d():
        atlas_float = _refine_pytorch3d(mesh, keyframes, num_steps,
                                         progress_fn=progress_fn)
    else:
        atlas_float = _refine_fallback(mesh, keyframes, num_steps,
                                        progress_fn=progress_fn)

    atlas_uint8 = (np.clip(atlas_float, 0, 1) * 255).astype(np.uint8)

    from PIL import Image

    # Inpainting pass: fill any remaining gaps in the atlas
    try:
        from atlas_inpaint import _detect_gaps, _inpaint_lama, _inpaint_opencv
        gap_mask = _detect_gaps(atlas_uint8)
        if np.sum(gap_mask > 0) > 0:
            logger.info("Running inpainting on atlas gaps...")
            atlas_uint8 = _inpaint_lama(atlas_uint8, gap_mask)
            logger.info("Inpainting complete")
    except Exception as e:
        logger.warning(f"Inpainting skipped: {e}")

    img = Image.fromarray(atlas_uint8, "RGB")
    out_path = run_dir / "hq_atlas.png"
    img.save(out_path, "PNG")
    logger.info(f"Saved HQ atlas to {out_path}")

    return out_path
