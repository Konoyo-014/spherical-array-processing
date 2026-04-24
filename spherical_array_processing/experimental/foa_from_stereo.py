from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import stft


@dataclass
class StereoFOAConfig:
    nperseg: int = 1024
    noverlap: int = 512
    geometry: str = "lr_pair"
    c: float = 343.0
    mic_spacing_m: float = 0.18
    max_azimuth_deg: float = 80.0
    pressure_weight: float = 1.0
    confidence_floor: float = 0.05
    low_freq_hz: float = 150.0
    ipd_clip_mode: Literal["hard", "soft"] = "soft"


@dataclass
class FOAEstimate:
    foa_stft: np.ndarray  # [4, F, T] complex, channels [W, X, Y, Z]
    freqs_hz: np.ndarray
    times_s: np.ndarray
    observability_mask: np.ndarray  # [4] bool
    confidence: np.ndarray  # [F, T]
    uncertainty: np.ndarray  # [F, T]
    metadata: dict = field(default_factory=dict)


def _estimate_sin_azimuth(ipd: np.ndarray, denom: np.ndarray, mode: str) -> np.ndarray:
    if mode == "hard":
        return np.clip(ipd / denom, -1.0, 1.0)
    if mode == "soft":
        # Smooth bounded estimate reduces low-frequency phase-wrap instability.
        return np.tanh(ipd / np.maximum(denom, 1e-9))
    raise ValueError("ipd_clip_mode must be 'hard' or 'soft'")


def estimate_incomplete_foa_from_stereo(
    stereo: ArrayLike,
    fs: float,
    config: StereoFOAConfig | None = None,
) -> FOAEstimate:
    """Experimental stereo -> incomplete FOA estimator.

    Assumption: a horizontal left-right pair with known spacing.
    We estimate W and an approximate horizontal X component from mid/side.
    Y and Z are marked unobservable by default.
    """
    cfg = config or StereoFOAConfig()
    x = np.asarray(stereo, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("stereo must have shape [n_samples, 2]")
    left = x[:, 0]
    right = x[:, 1]
    f, t, zl = stft(left, fs=fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    _, _, zr = stft(right, fs=fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    mid = 0.5 * (zl + zr)
    side = 0.5 * (zl - zr)

    cross = zl * np.conj(zr)
    ipd = np.angle(cross)
    k = 2 * np.pi * f[:, None] / cfg.c
    denom = np.maximum(k * cfg.mic_spacing_m, 1e-9)
    sin_az = _estimate_sin_azimuth(ipd, denom, cfg.ipd_clip_mode)
    az = np.arcsin(sin_az)
    az_limit = np.deg2rad(cfg.max_azimuth_deg)
    base_conf = 1.0 - np.clip(np.abs(az) / max(az_limit, 1e-9), 0.0, 1.0)
    freq_conf = np.clip((f[:, None] - cfg.low_freq_hz) / max(cfg.low_freq_hz, 1e-9), 0.0, 1.0)
    confidence = cfg.confidence_floor + (1.0 - cfg.confidence_floor) * (base_conf * freq_conf)
    # Down-weight bins with negligible pressure energy to prevent false certainty.
    mid_mag = np.abs(mid)
    mid_ref = np.maximum(np.percentile(mid_mag, 80), 1e-9)
    energy_conf = np.clip(mid_mag / mid_ref, 0.0, 1.0)
    confidence = confidence * energy_conf
    confidence[0, :] = cfg.confidence_floor * 0.5  # DC bin is weakly informative
    confidence = np.clip(confidence, 0.0, 1.0)

    # Incomplete FOA mapping (horizontal-only proxy):
    # W ~ mid, X ~ side stabilized by confidence, Y/Z unobservable
    w = cfg.pressure_weight * mid
    x_est = side * confidence
    y_est = np.zeros_like(mid)
    z_est = np.zeros_like(mid)
    foa = np.stack([w, x_est, y_est, z_est], axis=0)
    mask = np.array([True, True, False, False], dtype=bool)
    side_mag = np.abs(side)
    side_ref = np.maximum(np.percentile(side_mag, 75), 1e-9)
    side_rel = np.clip(side_mag / side_ref, 0.0, 1.0)
    uncertainty = np.clip((1.0 - confidence) + 0.35 * (1.0 - side_rel), 0.0, 1.0)
    return FOAEstimate(
        foa_stft=foa,
        freqs_hz=f,
        times_s=t,
        observability_mask=mask,
        confidence=confidence,
        uncertainty=uncertainty,
        metadata={
            "geometry": cfg.geometry,
            "assumption": "horizontal left-right pair, incomplete FOA (W/X observable proxy only)",
            "mic_spacing_m": cfg.mic_spacing_m,
            "ipd_clip_mode": cfg.ipd_clip_mode,
            "low_freq_hz": cfg.low_freq_hz,
            "confidence_floor": cfg.confidence_floor,
        },
    )
