from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore
except Exception:  # pragma: no cover
    ssim = None


def compare_grayscale_images(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        raise ValueError("image shape mismatch")
    diff = cand - ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    out = {"rmse": rmse, "mae": mae}
    if ssim is not None:
        data_range = float(max(ref.max(), cand.max()) - min(ref.min(), cand.min()) + 1e-12)
        out["ssim"] = float(ssim(ref, cand, data_range=data_range))
    return out


def load_image_gray(path: str | Path) -> np.ndarray:
    import matplotlib.image as mpimg

    img = mpimg.imread(path)
    if img.ndim == 2:
        return img.astype(float)
    if img.shape[-1] >= 3:
        rgb = img[..., :3].astype(float)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    raise ValueError("unsupported image shape")

