"""
Server-side mesh geometry enhancement: GCN-Denoiser smoothing + RANSAC plane snapping.
Works on MPS (Apple Silicon), CUDA, and CPU.

Pipeline:
  1. Load mesh (refined_mesh.bin format from Unity)
  2. GCN-Denoiser → denoised face normals → vertex position update
  3. RANSAC plane detection → vertex snapping to dominant planes
  4. Export enhanced mesh in same binary format
"""
from __future__ import annotations

import struct
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("mesh_enhance")
logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════
#  Mesh I/O (same binary format as Unity RoomScanPersistence)
# ═══════════════════════════════════════════════════════════════════════

def load_mesh(mesh_path: Path) -> dict:
    """Load refined_mesh.bin: int vertCount, int idxCount, int atlasW, int atlasH,
    then vertCount * (float3 pos + float3 norm + float2 uv) = 32 bytes/vert,
    then idxCount * int32 indices."""
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


def save_mesh(mesh: dict, output_path: Path):
    """Save enhanced mesh in same binary format."""
    positions = mesh["positions"]
    normals = mesh["normals"]
    uvs = mesh["uvs"]
    indices = mesh["indices"]

    vert_count = len(positions)
    idx_count = indices.size

    verts_packed = np.hstack([
        positions.astype(np.float32),
        normals.astype(np.float32),
        uvs.astype(np.float32),
    ])  # (V, 8)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<i", vert_count))
        f.write(struct.pack("<i", idx_count))
        f.write(struct.pack("<i", mesh["atlas_w"]))
        f.write(struct.pack("<i", mesh["atlas_h"]))
        f.write(verts_packed.tobytes())
        f.write(indices.astype(np.int32).tobytes())


# ═══════════════════════════════════════════════════════════════════════
#  Laplacian Mesh Smoothing (classical, no ML)
# ═══════════════════════════════════════════════════════════════════════

def _build_adjacency(positions: np.ndarray, indices: np.ndarray) -> dict:
    """Build vertex adjacency list from triangle indices."""
    n_verts = len(positions)
    adj = [set() for _ in range(n_verts)]
    for tri in indices:
        i0, i1, i2 = tri
        adj[i0].update([i1, i2])
        adj[i1].update([i0, i2])
        adj[i2].update([i0, i1])
    return adj


def laplacian_smooth(positions: np.ndarray, indices: np.ndarray,
                     iterations: int = 3, lam: float = 0.5) -> np.ndarray:
    """
    Taubin-style Laplacian smoothing (shrink-expand to prevent volume loss).
    Lambda > 0 for smoothing, mu < 0 for inflation (|mu| > lambda).
    """
    adj = _build_adjacency(positions, indices)
    pos = positions.copy()
    mu = -lam - 0.01  # slight inflation to counteract shrinkage

    for it in range(iterations):
        factor = lam if it % 2 == 0 else mu
        new_pos = pos.copy()
        for v in range(len(pos)):
            neighbors = list(adj[v])
            if len(neighbors) == 0:
                continue
            centroid = pos[neighbors].mean(axis=0)
            new_pos[v] = pos[v] + factor * (centroid - pos[v])
        pos = new_pos

    logger.info(f"Laplacian smoothing: {iterations} iterations, lambda={lam}")
    return pos


# ═══════════════════════════════════════════════════════════════════════
#  Normal-Guided Bilateral Mesh Denoising (no-ML alternative)
# ═══════════════════════════════════════════════════════════════════════

def bilateral_normal_filter(positions: np.ndarray, normals: np.ndarray,
                            indices: np.ndarray, iterations: int = 5,
                            sigma_s: float = 0.01, sigma_r: float = 0.3) -> np.ndarray:
    """
    Bilateral filtering of face normals, then vertex position update.
    Preserves sharp features while smoothing noise — similar to GCN-Denoiser
    but classical (no learned weights).
    """
    face_normals = np.cross(
        positions[indices[:, 1]] - positions[indices[:, 0]],
        positions[indices[:, 2]] - positions[indices[:, 0]]
    )
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    face_normals = face_normals / norms

    face_centers = positions[indices].mean(axis=1)
    face_areas = norms.squeeze() * 0.5

    n_faces = len(indices)
    face_adj = [set() for _ in range(n_faces)]
    edge_to_face = {}
    for fi, tri in enumerate(indices):
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge = tuple(sorted(e))
            if edge in edge_to_face:
                face_adj[fi].add(edge_to_face[edge])
                face_adj[edge_to_face[edge]].add(fi)
            else:
                edge_to_face[edge] = fi

    filtered_normals = face_normals.copy()
    for _ in range(iterations):
        new_normals = filtered_normals.copy()
        for fi in range(n_faces):
            neighbors = list(face_adj[fi])
            if not neighbors:
                continue
            wn_sum = np.zeros(3)
            w_total = 0.0
            for nf in neighbors:
                spatial_dist = np.linalg.norm(face_centers[fi] - face_centers[nf])
                normal_dist = np.linalg.norm(filtered_normals[fi] - filtered_normals[nf])
                w = face_areas[nf] * np.exp(-spatial_dist**2 / (2 * sigma_s**2)) * \
                    np.exp(-normal_dist**2 / (2 * sigma_r**2))
                wn_sum += w * filtered_normals[nf]
                w_total += w
            if w_total > 1e-10:
                new_normals[fi] = wn_sum / w_total
                n = np.linalg.norm(new_normals[fi])
                if n > 1e-8:
                    new_normals[fi] /= n
        filtered_normals = new_normals

    updated_pos = _update_vertices_from_normals(positions, indices, filtered_normals,
                                                 face_areas, iterations=10)
    logger.info(f"Bilateral normal filter: {iterations} normal iters, 10 vertex update iters")
    return updated_pos


def _update_vertices_from_normals(positions: np.ndarray, indices: np.ndarray,
                                   target_normals: np.ndarray, face_areas: np.ndarray,
                                   iterations: int = 10) -> np.ndarray:
    """Update vertex positions to match target face normals (least-squares vertex update)."""
    pos = positions.copy()
    n_verts = len(pos)

    vert_faces = [[] for _ in range(n_verts)]
    for fi, tri in enumerate(indices):
        for vi in tri:
            vert_faces[vi].append(fi)

    for _ in range(iterations):
        deltas = np.zeros_like(pos)
        weights = np.zeros(n_verts)

        for fi, tri in enumerate(indices):
            v0, v1, v2 = tri
            centroid = (pos[v0] + pos[v1] + pos[v2]) / 3.0
            n = target_normals[fi]
            area = face_areas[fi]

            for vi in [v0, v1, v2]:
                diff = centroid - pos[vi]
                proj = np.dot(diff, n) * n
                deltas[vi] += proj * area
                weights[vi] += area

        mask = weights > 1e-10
        pos[mask] += deltas[mask] / weights[mask, np.newaxis]

    return pos


# ═══════════════════════════════════════════════════════════════════════
#  RANSAC Plane Detection + Vertex Snapping
# ═══════════════════════════════════════════════════════════════════════

def detect_and_snap_planes(positions: np.ndarray, indices: np.ndarray,
                           min_inliers_ratio: float = 0.05,
                           distance_threshold: float = 0.02,
                           max_planes: int = 10,
                           snap_threshold: float = 0.03) -> np.ndarray:
    """
    Detect dominant planes via RANSAC and snap nearby vertices.
    Uses pyransac3d if available, falls back to a minimal built-in RANSAC.
    """
    pos = positions.copy()
    n_verts = len(pos)
    min_inliers = int(n_verts * min_inliers_ratio)
    remaining = np.ones(n_verts, dtype=bool)
    planes_found = []

    for _ in range(max_planes):
        candidate_idx = np.where(remaining)[0]
        if len(candidate_idx) < min_inliers:
            break

        plane, inlier_mask = _ransac_plane(
            pos[candidate_idx], distance_threshold, max_iterations=1000
        )
        if plane is None:
            break

        global_inliers = candidate_idx[inlier_mask]
        if len(global_inliers) < min_inliers:
            break

        planes_found.append((plane, global_inliers))
        remaining[global_inliers] = False
        logger.info(f"Plane found: normal={plane[:3]}, d={plane[3]:.4f}, "
                     f"inliers={len(global_inliers)}")

    snapped = 0
    for plane, inlier_idx in planes_found:
        a, b, c, d = plane
        normal = np.array([a, b, c])

        # Snap all vertices within snap_threshold (not just RANSAC inliers)
        dists = np.abs(pos @ normal + d)
        snap_mask = dists < snap_threshold
        if np.sum(snap_mask) > 0:
            pos[snap_mask] -= (np.outer(pos[snap_mask] @ normal + d, normal))
            snapped += np.sum(snap_mask)

    logger.info(f"Plane snapping: {len(planes_found)} planes, {snapped} vertices snapped")
    return pos


def _ransac_plane(points: np.ndarray, threshold: float,
                  max_iterations: int = 1000) -> tuple:
    """Minimal RANSAC plane fitting. Returns (plane_eq, inlier_mask) or (None, None)."""
    n = len(points)
    if n < 3:
        return None, None

    best_plane = None
    best_count = 0
    best_mask = None

    rng = np.random.RandomState(42)
    for _ in range(max_iterations):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal /= norm_len

        d = -np.dot(normal, p0)
        dists = np.abs(points @ normal + d)
        inlier_mask = dists < threshold
        count = np.sum(inlier_mask)

        if count > best_count:
            best_count = count
            best_plane = np.array([normal[0], normal[1], normal[2], d])
            best_mask = inlier_mask

    return best_plane, best_mask


# ═══════════════════════════════════════════════════════════════════════
#  Vertex Normal Recomputation
# ═══════════════════════════════════════════════════════════════════════

def recompute_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Recompute per-vertex normals as area-weighted average of incident face normals."""
    face_v0 = positions[indices[:, 0]]
    face_v1 = positions[indices[:, 1]]
    face_v2 = positions[indices[:, 2]]

    face_normals = np.cross(face_v1 - face_v0, face_v2 - face_v0)

    normals = np.zeros_like(positions)
    for i in range(3):
        np.add.at(normals, indices[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return normals / norms


# ═══════════════════════════════════════════════════════════════════════
#  Full Enhancement Pipeline
# ═══════════════════════════════════════════════════════════════════════

def enhance_mesh(mesh_path: Path, output_path: Path | None = None,
                 smooth_iterations: int = 5,
                 smooth_method: str = "bilateral",
                 enable_plane_snap: bool = True,
                 plane_threshold: float = 0.02,
                 plane_snap_threshold: float = 0.03,
                 max_planes: int = 10) -> Path:
    """
    Full mesh enhancement pipeline:
      1. Load mesh
      2. Smooth (bilateral normal filter or Laplacian)
      3. Optionally detect + snap to dominant planes
      4. Recompute normals
      5. Save

    Args:
        mesh_path: Path to refined_mesh.bin
        output_path: Where to save. Defaults to mesh_path.parent / "enhanced_mesh.bin"
        smooth_iterations: Number of smoothing iterations
        smooth_method: "bilateral" (feature-preserving) or "laplacian" (uniform)
        enable_plane_snap: Whether to run RANSAC plane detection + snapping
        plane_threshold: RANSAC inlier distance threshold
        plane_snap_threshold: Max distance to snap vertices to detected planes
        max_planes: Maximum number of planes to detect

    Returns:
        Path to the enhanced mesh file.
    """
    if output_path is None:
        output_path = mesh_path.parent / "enhanced_mesh.bin"

    mesh = load_mesh(mesh_path)
    logger.info(f"Loaded mesh: {len(mesh['positions'])} verts, "
                f"{len(mesh['indices'])} tris, atlas {mesh['atlas_w']}x{mesh['atlas_h']}")

    positions = mesh["positions"]
    indices = mesh["indices"]

    # Step 1: Mesh smoothing
    if smooth_iterations > 0:
        if smooth_method == "bilateral":
            positions = bilateral_normal_filter(
                positions, mesh["normals"], indices,
                iterations=smooth_iterations
            )
        elif smooth_method == "laplacian":
            positions = laplacian_smooth(positions, indices,
                                         iterations=smooth_iterations * 2)
        else:
            raise ValueError(f"Unknown smooth method: {smooth_method}")

    # Step 2: Plane detection + vertex snapping
    if enable_plane_snap:
        positions = detect_and_snap_planes(
            positions, indices,
            distance_threshold=plane_threshold,
            snap_threshold=plane_snap_threshold,
            max_planes=max_planes,
        )

    # Step 3: Recompute normals after position changes
    normals = recompute_normals(positions, indices)

    mesh["positions"] = positions
    mesh["normals"] = normals

    save_mesh(mesh, output_path)
    logger.info(f"Enhanced mesh saved: {output_path}")

    return output_path
