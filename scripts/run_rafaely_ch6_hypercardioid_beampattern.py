#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.types import FigureReproConfig


def compute_hypercardioid_curves(n_samples: int = 512):
    theta = np.linspace(0.0, 2 * np.pi, n_samples)
    z = np.cos(theta)
    curves = {
        1: (1 / 4) * (3 * z + 1),
        2: (1 / 6) * (5 * z**2 + 2 * z - 1),
        3: (1 / 32) * (35 * z**3 + 15 * z**2 - 15 * z - 3),
        4: (1 / 40) * (63 * z**4 + 28 * z**3 - 42 * z**2 - 12 * z + 3),
        5: (1 / 96) * (231 * z**5 + 105 * z**4 - 210 * z**3 - 70 * z**2 + 35 * z + 5),
    }
    return theta, curves


def main(show: bool = True):
    axis_font_size = 14
    theta, curves = compute_hypercardioid_curves()
    with figure_repro_context(FigureReproConfig(font_size=axis_font_size, line_width=2.0)):
        fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "polar"})
        for ax, N in zip(axs.ravel(), (1, 2, 3, 4)):
            ax.plot(theta, np.abs(curves[N]), "-", linewidth=2, color=(0, 0, 0.5))
            ax.tick_params(labelsize=axis_font_size - 2)
            ax.set_title(rf"$N={N}$", fontsize=axis_font_size)
        fig.tight_layout()
        if show:
            plt.show()
    return theta, curves, fig


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
