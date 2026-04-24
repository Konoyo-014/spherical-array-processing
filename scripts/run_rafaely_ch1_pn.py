#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre

from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.types import FigureReproConfig


def compute_pn_data(max_order: int = 4, n_samples: int = 256):
    x = np.linspace(-1.0, 1.0, n_samples)
    polys = {n: eval_legendre(n, x) for n in range(max_order + 1)}
    return x, polys


def main(show: bool = True):
    axis_font_size = 16
    x, polys = compute_pn_data()
    with figure_repro_context(FigureReproConfig(font_size=axis_font_size, line_width=1.5)):
        fig = plt.figure()
        positions = {0: 1, 1: 3, 2: 4, 3: 5, 4: 6}
        for n in range(5):
            ax = fig.add_subplot(3, 2, positions[n])
            ax.plot(x, polys[n], "-", color=(0, 0, 0.5), linewidth=1.5)
            y_text = 2.3 if n == 0 else 1.3
            ax.text(-0.95, y_text, rf"$n={n}$", fontsize=axis_font_size)
            ax.tick_params(labelsize=axis_font_size)
            if n in (0, 1, 2):
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$x$", fontsize=axis_font_size)
        fig.tight_layout()
        if show:
            plt.show()
    return x, polys, fig


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
