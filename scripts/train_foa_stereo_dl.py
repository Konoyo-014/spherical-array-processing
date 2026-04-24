#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _make_synthetic_dataset(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Features roughly mimic [Re(mid), Im(mid), Re(side), Im(side), ipd, ild, |mid|, |side|]
    x = rng.normal(size=(n_samples, 8))
    x[:, 4] = np.tanh(x[:, 4]) * np.pi  # ipd
    x[:, 5] = np.tanh(x[:, 5]) * 3.0  # ild
    x[:, 6] = np.abs(x[:, 6])
    x[:, 7] = np.abs(x[:, 7])

    # Target is [Re/Im of W,X,Y,Z] with limited Y/Z observability.
    w_true = np.array(
        [
            [0.90, 0.05, 0.10, -0.04, 0.02, 0.03, 0.20, -0.02],
            [0.05, 0.85, -0.01, 0.12, -0.03, 0.01, 0.11, 0.02],
            [0.12, -0.04, 0.82, 0.03, 0.10, -0.08, -0.02, 0.23],
            [0.04, 0.10, -0.05, 0.79, -0.06, 0.11, 0.03, 0.18],
            [0.03, 0.02, 0.08, -0.07, 0.22, 0.10, 0.05, 0.07],
            [0.02, 0.01, -0.07, 0.06, -0.18, 0.16, -0.04, 0.05],
            [0.01, -0.03, 0.02, 0.01, 0.08, -0.06, 0.03, 0.04],
            [0.00, 0.02, 0.01, 0.03, -0.05, 0.07, 0.02, 0.01],
        ],
        dtype=float,
    )
    y = x @ w_true.T
    noise = rng.normal(scale=np.array([0.03, 0.03, 0.04, 0.04, 0.09, 0.09, 0.12, 0.12]), size=y.shape)
    y = y + noise
    return x, y


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc + alpha * np.eye(x.shape[1])
    w = np.linalg.solve(gram, xc.T @ yc)
    b = (y_mean - x_mean @ w).reshape(-1)
    return w, b


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    energy_true = np.sum(y_true.reshape(-1, 4, 2) ** 2, axis=(1, 2))
    energy_pred = np.sum(y_pred.reshape(-1, 4, 2) ** 2, axis=(1, 2))
    energy_mae = float(np.mean(np.abs(energy_pred - energy_true)))
    wx_true = y_true[:, :4]
    wx_pred = y_pred[:, :4]
    wx_corr = float(np.corrcoef(wx_true.reshape(-1), wx_pred.reshape(-1))[0, 1])
    yz_rmse = float(np.sqrt(np.mean((y_pred[:, 4:] - y_true[:, 4:]) ** 2)))
    uncertainty = np.clip(np.linalg.norm(err, axis=1) / np.maximum(np.percentile(np.linalg.norm(err, axis=1), 95), eps), 0.0, 1.0)
    calib = float(np.mean(np.abs(uncertainty - (np.linalg.norm(err, axis=1) / np.maximum(np.max(np.linalg.norm(err, axis=1)), eps)))))
    return {
        "mae": mae,
        "rmse": rmse,
        "energy_mae": energy_mae,
        "wx_corr": wx_corr,
        "yz_rmse": yz_rmse,
        "uncertainty_calibration_error": calib,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight linear stereo->FOA DL checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/foa_dl/linear_checkpoint.npz"))
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/foa_dl/train_metrics.json"))
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    x_train, y_train = _make_synthetic_dataset(args.train_samples, rng)
    x_val, y_val = _make_synthetic_dataset(args.val_samples, rng)
    w, b = _fit_ridge(x_train, y_train, args.alpha)
    y_pred = x_val @ w + b
    resid_var = np.var(y_train - (x_train @ w + b), axis=0).reshape(4, 2).mean(axis=1)
    metrics = _evaluate(y_val, y_pred)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        weights=w,
        bias=b,
        residual_var=resid_var,
        feature_names=np.array(["re_mid", "im_mid", "re_side", "im_side", "ipd", "ild", "abs_mid", "abs_side"]),
    )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote checkpoint: {args.output}")
    print(f"wrote metrics: {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

