"""
Atlas inpainting: fill gaps/holes in baked texture atlases using LaMa.
Falls back to OpenCV-based inpainting if LaMa is unavailable.
Works on MPS (Apple Silicon), CUDA, and CPU.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("atlas_inpaint")
logger.setLevel(logging.INFO)


def _detect_gaps(atlas_rgb: np.ndarray, atlas_alpha: np.ndarray | None = None,
                 black_threshold: int = 5) -> np.ndarray:
    """
    Detect unfilled regions in the atlas.
    Returns a binary mask (255 = gap, 0 = filled).
    """
    h, w = atlas_rgb.shape[:2]

    if atlas_alpha is not None:
        mask = (atlas_alpha < 128).astype(np.uint8) * 255
    else:
        # No alpha channel — detect near-black pixels as gaps
        gray = atlas_rgb.mean(axis=2)
        mask = (gray < black_threshold).astype(np.uint8) * 255

    gap_count = np.sum(mask > 0)
    total = h * w
    logger.info(f"Gap detection: {gap_count}/{total} texels "
                f"({100 * gap_count / max(total, 1):.1f}%)")

    return mask


def _inpaint_opencv(image: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """OpenCV Navier-Stokes inpainting (classical, fast, no ML)."""
    try:
        import cv2
        result = cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
        logger.info("Inpainted with OpenCV (Navier-Stokes)")
        return result
    except ImportError:
        logger.warning("OpenCV not available, filling with dilation")
        return _inpaint_dilation(image, mask)


def _inpaint_dilation(image: np.ndarray, mask: np.ndarray, iterations: int = 16) -> np.ndarray:
    """Simple dilation-based gap filling (no dependencies)."""
    result = image.copy()
    gap = mask > 0

    for _ in range(iterations):
        if not np.any(gap):
            break

        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode='edge')
        gap_padded = np.pad(gap, ((1, 1), (1, 1)), mode='constant', constant_values=True)

        color_sum = np.zeros_like(result, dtype=np.float64)
        count = np.zeros(result.shape[:2], dtype=np.float64)

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                oy, ox = 1 + dy, 1 + dx
                neighbor_color = padded[oy:oy + result.shape[0], ox:ox + result.shape[1]]
                neighbor_filled = ~gap_padded[oy:oy + result.shape[0], ox:ox + result.shape[1]]

                mask_2d = neighbor_filled & gap
                color_sum[mask_2d] += neighbor_color[mask_2d].astype(np.float64)
                count[mask_2d] += 1

        fillable = count > 0
        result[fillable] = (color_sum[fillable] / count[fillable, np.newaxis]).astype(np.uint8)
        gap[fillable] = False

    filled = np.sum(mask > 0) - np.sum(gap)
    logger.info(f"Dilation inpainting: filled {filled} texels in {iterations} iterations")
    return result


def _inpaint_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LaMa deep inpainting model (Apache 2.0, works on MPS/CUDA/CPU)."""
    try:
        import torch
        from lama_cleaner.model.lama import LaMa
        from lama_cleaner.schema import Config
    except ImportError:
        logger.info("lama-cleaner not installed, falling back to OpenCV inpainting")
        return _inpaint_opencv(image, mask)

    try:
        device_str = "cpu"
        try:
            if torch.cuda.is_available():
                device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"
        except Exception:
            pass

        device = torch.device(device_str)
        model = LaMa(device)

        config = Config(
            ldm_steps=1,
            hd_strategy="Original",
            hd_strategy_crop_margin=128,
            hd_strategy_crop_trigger_size=2048,
            hd_strategy_resize_limit=2048,
        )

        result = model(image, mask, config)
        logger.info(f"Inpainted with LaMa on {device_str}")
        return result

    except Exception as e:
        logger.warning(f"LaMa failed ({e}), falling back to OpenCV")
        return _inpaint_opencv(image, mask)


def inpaint_atlas(input_path: Path, output_path: Path | None = None,
                  method: str = "auto") -> Path:
    """
    Fill gaps in an atlas texture.

    Args:
        input_path: Path to atlas PNG (RGB or RGBA).
        output_path: Where to write result. Defaults to input_path.stem + "_inpainted.png".
        method: "lama" (ML), "opencv" (classical), "dilation" (minimal), or "auto" (try in order).

    Returns:
        Path to the inpainted atlas PNG.
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_inpainted.png"

    img = Image.open(input_path)

    if img.mode == "RGBA":
        atlas_rgb = np.array(img.convert("RGB"))
        atlas_alpha = np.array(img.split()[-1])
        mask = _detect_gaps(atlas_rgb, atlas_alpha)
    else:
        atlas_rgb = np.array(img.convert("RGB"))
        mask = _detect_gaps(atlas_rgb)

    if np.sum(mask > 0) == 0:
        logger.info("No gaps detected, saving as-is")
        img.convert("RGB").save(str(output_path), "PNG")
        return output_path

    if method == "auto":
        result = _inpaint_lama(atlas_rgb, mask)
    elif method == "lama":
        result = _inpaint_lama(atlas_rgb, mask)
    elif method == "opencv":
        result = _inpaint_opencv(atlas_rgb, mask)
    elif method == "dilation":
        result = _inpaint_dilation(atlas_rgb, mask)
    else:
        raise ValueError(f"Unknown inpainting method: {method}")

    Image.fromarray(result).save(str(output_path), "PNG")
    logger.info(f"Inpainted atlas saved: {output_path}")
    return output_path
