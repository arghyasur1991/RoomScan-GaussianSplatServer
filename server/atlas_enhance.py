"""
Atlas enhancement utilities: super-resolution, inpainting, seam blending.
Works on MPS (Apple Silicon), CUDA, and CPU.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("atlas_enhance")
logger.setLevel(logging.INFO)

_realesrgan_model = None


def _get_device_str() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _load_realesrgan(scale: int = 4):
    """Lazy-load Real-ESRGAN upsampler. Caches across calls."""
    global _realesrgan_model
    if _realesrgan_model is not None and _realesrgan_model._scale == scale:
        return _realesrgan_model

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise ImportError(
            "Real-ESRGAN not installed. Run: pip install realesrgan basicsr"
        )

    import torch
    device = _get_device_str()

    if scale == 4:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_name = "RealESRGAN_x4plus"
    elif scale == 2:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model_name = "RealESRGAN_x2plus"
    else:
        raise ValueError(f"Unsupported scale: {scale}. Use 2 or 4.")

    # RealESRGANer handles weight download automatically
    gpu_id = 0 if device == "cuda" else None
    upsampler = RealESRGANer(
        scale=scale,
        model_path=None,  # auto-downloads
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=device == "cuda",
        gpu_id=gpu_id,
        device=torch.device(device),
    )
    upsampler._scale = scale
    _realesrgan_model = upsampler
    logger.info(f"Loaded Real-ESRGAN x{scale} on {device}")
    return upsampler


def upscale_atlas(input_path: Path, output_path: Path | None = None, scale: int = 4) -> Path:
    """
    Upscale an atlas image using Real-ESRGAN.

    Args:
        input_path: Path to input atlas (PNG or raw RGBA).
        output_path: Where to write the result. Defaults to input_path.stem + f"_sr{scale}x.png".
        scale: Upscale factor (2 or 4).

    Returns:
        Path to the upscaled atlas PNG.
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_sr{scale}x.png"

    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)

    # Real-ESRGAN expects BGR (OpenCV convention)
    img_bgr = img_np[:, :, ::-1]

    upsampler = _load_realesrgan(scale)
    output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)

    output_rgb = output_bgr[:, :, ::-1]
    output_img = Image.fromarray(output_rgb)
    output_img.save(str(output_path), "PNG")

    logger.info(f"Atlas upscaled: {img.size} -> {output_img.size} (x{scale}), saved to {output_path}")
    return output_path
