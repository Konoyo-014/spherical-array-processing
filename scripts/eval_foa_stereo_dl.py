#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spherical_array_processing.experimental import StereoFOADLConfig, estimate_incomplete_foa_from_stereo_dl


def _build_stereo_eval_signal(fs: int, seconds: float) -> np.ndarray:
    n = int(fs * seconds)
    t = np.arange(n, dtype=float) / fs
    left = (
        0.65 * np.sin(2 * np.pi * 440 * t)
        + 0.35 * np.sin(2 * np.pi * 880 * t + 0.2)
        + 0.15 * np.sin(2 * np.pi * 220 * t + 0.7)
    )
    right = (
        0.65 * np.sin(2 * np.pi * 440 * t + 0.15)
        + 0.30 * np.sin(2 * np.pi * 880 * t - 0.1)
        + 0.20 * np.sin(2 * np.pi * 220 * t + 0.4)
    )
    return np.stack([left, right], axis=1)


def _summarize(est) -> dict[str, float]:
    w = est.foa_stft[0]
    x = est.foa_stft[1]
    y = est.foa_stft[2]
    z = est.foa_stft[3]
    energy = np.abs(w) ** 2 + np.abs(x) ** 2 + np.abs(y) ** 2 + np.abs(z) ** 2
    az = np.arctan2(np.real(y), np.real(x) + 1e-9)
    return {
        "foa_energy_mean": float(np.mean(energy)),
        "foa_energy_std": float(np.std(energy)),
        "azimuth_std_deg": float(np.rad2deg(np.std(az))),
        "confidence_mean": float(np.mean(est.confidence)),
        "uncertainty_mean": float(np.mean(est.uncertainty)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate stereo->FOA DL estimator on deterministic synthetic audio.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/foa_dl/eval_metrics.json"))
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args(argv)

    stereo = _build_stereo_eval_signal(args.fs, args.seconds)
    est = estimate_incomplete_foa_from_stereo_dl(
        stereo,
        fs=float(args.fs),
        config=StereoFOADLConfig(checkpoint_path=str(args.checkpoint), return_uncertainty=True),
    )
    metrics = _summarize(est)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote eval metrics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
