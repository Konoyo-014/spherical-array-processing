from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import stft

from .foa_from_stereo import FOAEstimate, StereoFOAConfig, estimate_incomplete_foa_from_stereo


@dataclass
class StereoFOADLConfig:
    model_name: str = "linear_stft"
    feature_set: str = "mid_side_ipd_ild"
    checkpoint_path: str | None = None
    return_uncertainty: bool = True
    nperseg: int = 1024
    noverlap: int = 512
    c: float = 343.0
    mic_spacing_m: float = 0.18
    uncertainty_floor: float = 0.1


def _extract_features(zl: np.ndarray, zr: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mid = 0.5 * (zl + zr)
    side = 0.5 * (zl - zr)
    ipd = np.angle(zl * np.conj(zr))
    ild = np.log(np.maximum(np.abs(zl), 1e-9) / np.maximum(np.abs(zr), 1e-9))
    feat = np.stack(
        [
            np.real(mid),
            np.imag(mid),
            np.real(side),
            np.imag(side),
            ipd,
            ild,
            np.abs(mid),
            np.abs(side),
        ],
        axis=-1,
    )
    aux = {"mid": mid, "side": side, "ipd": ipd, "ild": ild}
    return feat, aux


def _load_linear_checkpoint(path: str | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    data = np.load(ckpt_path, allow_pickle=False)
    out = {
        "weights": np.asarray(data["weights"], dtype=float),
        "bias": np.asarray(data["bias"], dtype=float),
    }
    if "residual_var" in data:
        out["residual_var"] = np.asarray(data["residual_var"], dtype=float)
    return out


def estimate_incomplete_foa_from_stereo_dl(
    stereo: ArrayLike,
    fs: float,
    config: StereoFOADLConfig | None = None,
) -> FOAEstimate:
    cfg = config or StereoFOADLConfig()
    x = np.asarray(stereo, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("stereo must have shape [n_samples, 2]")

    left = x[:, 0]
    right = x[:, 1]
    f, t, zl = stft(left, fs=fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    _, _, zr = stft(right, fs=fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    feat, aux = _extract_features(zl, zr)  # [F,T,D]

    checkpoint = _load_linear_checkpoint(cfg.checkpoint_path)
    if checkpoint is None:
        # Fallback to robust direct estimator and expose as DL baseline mode.
        base_cfg = StereoFOAConfig(
            nperseg=cfg.nperseg,
            noverlap=cfg.noverlap,
            c=cfg.c,
            mic_spacing_m=cfg.mic_spacing_m,
        )
        est = estimate_incomplete_foa_from_stereo(stereo, fs, base_cfg)
        est.metadata["model_name"] = cfg.model_name
        est.metadata["feature_set"] = cfg.feature_set
        est.metadata["dl_mode"] = "fallback_direct"
        return est

    w = checkpoint["weights"]  # [D,8]
    b = checkpoint["bias"]  # [8]
    if w.shape != (feat.shape[-1], 8) or b.shape != (8,):
        raise ValueError(f"invalid checkpoint shape: weights={w.shape}, bias={b.shape}")

    flat = feat.reshape(-1, feat.shape[-1])
    pred = flat @ w + b
    pred = pred.reshape(feat.shape[0], feat.shape[1], 8)
    foa = np.empty((4, feat.shape[0], feat.shape[1]), dtype=np.complex128)
    for ch in range(4):
        foa[ch] = pred[:, :, 2 * ch] + 1j * pred[:, :, 2 * ch + 1]

    # Confidence/uncertainty from side visibility and optional residual variance.
    side_rel = np.clip(np.abs(aux["side"]) / np.maximum(np.percentile(np.abs(aux["side"]), 75), 1e-9), 0.0, 1.0)
    confidence = np.clip(0.2 + 0.8 * side_rel, 0.0, 1.0)
    if cfg.return_uncertainty:
        resid = checkpoint.get("residual_var")
        if resid is None:
            uncertainty = np.clip(1.0 - confidence, cfg.uncertainty_floor, 1.0)
        else:
            resid = np.asarray(resid, dtype=float).reshape(4)
            u = np.mean(resid)[None, None] * np.ones_like(confidence)
            u = np.clip(u / np.maximum(np.percentile(u, 90), 1e-9), 0.0, 1.0)
            uncertainty = np.clip(0.5 * u + 0.5 * (1.0 - confidence), cfg.uncertainty_floor, 1.0)
    else:
        uncertainty = np.clip(1.0 - confidence, 0.0, 1.0)

    return FOAEstimate(
        foa_stft=foa,
        freqs_hz=f,
        times_s=t,
        observability_mask=np.array([True, True, True, True], dtype=bool),
        confidence=confidence,
        uncertainty=uncertainty,
        metadata={
            "model_name": cfg.model_name,
            "feature_set": cfg.feature_set,
            "checkpoint_path": cfg.checkpoint_path,
            "dl_mode": "linear_checkpoint",
        },
    )

