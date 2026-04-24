#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.types import FigureReproConfig


def _legendre_poly_matrix_ascending(max_order: int = 4) -> np.ndarray:
    p = np.zeros((max_order + 1, max_order + 1), dtype=float)
    p[0] = np.array([1, 0, 0, 0, 0], dtype=float)
    p[1] = np.array([0, 1, 0, 0, 0], dtype=float)
    p[2] = 0.5 * np.array([-1, 0, 3, 0, 0], dtype=float)
    p[3] = 0.5 * np.array([0, -3, 0, 5, 0], dtype=float)
    p[4] = (1 / 8) * np.array([3, 0, -30, 0, 35], dtype=float)
    return p


def compute_supercardioid_patterns(max_order: int = 4, n_samples: int = 512):
    p = _legendre_poly_matrix_ascending(max_order)
    theta = np.linspace(0.0, 2 * np.pi, n_samples)
    x = np.cos(theta)
    Y = {}
    F_db = {}

    for N in range(max_order + 1):
        A = np.zeros((N + 1, N + 1), dtype=float)
        B = np.zeros((N + 1, N + 1), dtype=float)
        for n in range(N + 1):
            for nn in range(N + 1):
                a = 0.0
                b = 0.0
                for k in range(n + 1):
                    for l in range(nn + 1):
                        pk = p[n, k]
                        pl = p[nn, l]
                        a += (1.0 / (k + l + 1)) * pk * pl
                        b += (((-1) ** (k + l)) / (k + l + 1)) * pk * pl
                A[n, nn] = (1.0 / (8 * np.pi)) * (2 * n + 1) * (2 * nn + 1) * a
                B[n, nn] = (1.0 / (8 * np.pi)) * (2 * n + 1) * (2 * nn + 1) * b

        vals, vecs = eigh(A, B)
        idx = int(np.argmax(vals))
        lam = float(vals[idx])
        dn = vecs[:, idx]
        pp = np.diag(dn) @ np.diag((2 * np.arange(N + 1) + 1) / (4 * np.pi)) @ p[: N + 1, :]
        yp = pp.sum(axis=0)
        y = np.zeros_like(x)
        for i, c in enumerate(yp):
            y += c * x**i
        y = y / np.maximum(np.abs(y[0]), 1e-12) - 1e-3

        Y[N] = y
        F_db[N] = 10 * np.log10(max(lam, 1e-12))

    return theta, Y, F_db


def main(show: bool = True):
    axis_font_size = 14
    theta, Y, F_db = compute_supercardioid_patterns()
    with figure_repro_context(FigureReproConfig(font_size=axis_font_size, line_width=2.0)):
        fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "polar"})
        for ax, N in zip(axs.ravel(), (1, 2, 3, 4)):
            ax.plot(theta, np.abs(Y[N]), "-", linewidth=2, color=(0, 0, 0.5))
            ax.tick_params(labelsize=axis_font_size - 2)
            ax.set_title(rf"$N={N}, F={F_db[N]:.1f}\,$dB", fontsize=axis_font_size)
        fig.tight_layout()
        if show:
            plt.show()
    return theta, Y, F_db, fig


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
