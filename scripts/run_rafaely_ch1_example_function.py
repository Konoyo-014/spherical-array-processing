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
from spherical_array_processing.repro.rafaely import plot_balloon, plot_contour, plot_sphere
from spherical_array_processing.types import FigureReproConfig


def compute_example_function_coeffs(order: int = 2) -> np.ndarray:
    fnm = np.zeros((order + 1) ** 2, dtype=np.complex128)
    # Rafaely ACN indexing with rows=(N+1)^2 and 0-based Python indexing.
    fnm[4] = 2 * np.sqrt(2 * np.pi / 15)  # (n,m)=(2,-2)
    fnm[8] = 2 * np.sqrt(2 * np.pi / 15)  # (n,m)=(2, 2)
    return fnm


def main(show: bool = True):
    axis_font_size = 14
    fnm = compute_example_function_coeffs(order=2)
    figs: list[plt.Figure] = []
    with figure_repro_context(FigureReproConfig(font_size=axis_font_size, line_width=2.0)):
        ax1 = plot_sphere(fnm)
        figs.append(ax1.figure)

        ax2 = plot_contour(fnm, normalization=0, absolute=0)
        figs.append(ax2.figure)

        ax3 = plot_balloon(fnm, transparency=0.2)
        ax3.axis("on")
        ax3.set_xlabel(r"$x$", fontsize=axis_font_size)
        ax3.set_ylabel(r"$y$", fontsize=axis_font_size)
        ax3.set_zlabel(r"$z$", fontsize=axis_font_size)
        figs.append(ax3.figure)

        for fig in figs:
            fig.tight_layout()
        if show:
            plt.show()
    return fnm, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
