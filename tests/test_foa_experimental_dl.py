from __future__ import annotations

import numpy as np

from spherical_array_processing.experimental import StereoFOADLConfig, estimate_incomplete_foa_from_stereo_dl


def _synthetic_stereo(fs: int = 16000, seconds: float = 1.0) -> np.ndarray:
    n = int(fs * seconds)
    t = np.arange(n, dtype=float) / fs
    l = 0.8 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 900 * t + 0.3)
    r = 0.8 * np.sin(2 * np.pi * 440 * t + 0.12) + 0.2 * np.sin(2 * np.pi * 900 * t - 0.2)
    return np.stack([l, r], axis=1)


def test_stereo_to_foa_dl_fallback_mode():
    x = _synthetic_stereo()
    est = estimate_incomplete_foa_from_stereo_dl(x, fs=16000.0, config=StereoFOADLConfig(checkpoint_path=None))
    assert est.foa_stft.shape[0] == 4
    assert est.observability_mask.tolist() == [True, True, False, False]
    assert est.metadata["dl_mode"] == "fallback_direct"
    assert np.isfinite(est.confidence).all()
    assert np.isfinite(est.uncertainty).all()


def test_stereo_to_foa_dl_with_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.1, size=(8, 8))
    b = rng.normal(scale=0.01, size=(8,))
    rv = np.array([0.2, 0.3, 0.5, 0.6], dtype=float)
    ckpt = tmp_path / "ckpt.npz"
    np.savez(ckpt, weights=w, bias=b, residual_var=rv)
    x = _synthetic_stereo()
    est = estimate_incomplete_foa_from_stereo_dl(
        x,
        fs=16000.0,
        config=StereoFOADLConfig(checkpoint_path=str(ckpt), return_uncertainty=True),
    )
    assert est.foa_stft.shape[0] == 4
    assert est.observability_mask.tolist() == [True, True, True, True]
    assert est.metadata["dl_mode"] == "linear_checkpoint"
    assert est.metadata["checkpoint_path"] == str(ckpt)
    assert np.isfinite(est.foa_stft).all()
