#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lpmv

from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.types import FigureReproConfig


def compute_pnm_data(max_order: int = 4, n_samples: int = 256):
    x = np.linspace(-1.0, 1.0, n_samples)
    pnm: dict[tuple[int, int], np.ndarray] = {}
    for n in range(max_order + 1):
        for m in range(n + 1):
            pnm[(n, m)] = lpmv(m, n, x)
    return x, pnm


def main(show: bool = True):
    axis_font_size = 12
    x, pnm = compute_pnm_data()
    with figure_repro_context(FigureReproConfig(font_size=axis_font_size, line_width=1.5)):
        fig = plt.figure()
        for n in range(5):
            for m in range(n + 1):
                ax = fig.add_subplot(5, 5, n * 5 + m + 1)
                ax.plot(x, pnm[(n, m)], "-", color=(0, 0, 0.5), linewidth=1.5)
                ax.grid(True)
                ax.set_title(f"({n},{m})", fontsize=axis_font_size)
                ax.tick_params(labelsize=axis_font_size)
                if n == 4:
                    ax.set_xlabel(r"$x$", fontsize=axis_font_size)
                if n == 4 and m == 4:
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(0, 150)
        fig.tight_layout()
        if show:
            plt.show()
    return x, pnm, fig


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
