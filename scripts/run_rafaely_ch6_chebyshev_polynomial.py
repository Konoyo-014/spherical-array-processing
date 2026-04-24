#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.types import FigureReproConfig


def compute_chebyshev_demo():
    x = np.linspace(0.0, 1.07, 512)
    y8 = 128 * x**8 - 256 * x**6 + 160 * x**4 - 32 * x**2 + 1

    M = 8
    th0 = np.pi / 4
    x0 = np.cos(np.pi / (2 * M)) / np.cos(th0 / 2)
    R = np.cosh(M * np.arccosh(x0))
    th = np.linspace(-np.pi, np.pi, 512)
    z = x0 * np.cos(th / 2)
    b8 = (1.0 / R) * (128 * z**8 - 256 * z**6 + 160 * z**4 - 32 * z**2 + 1)
    return {"x": x, "y8": y8, "M": M, "th0": th0, "x0": x0, "R": R, "th": th, "b8": b8}


def main(show: bool = True):
    d = compute_chebyshev_demo()
    figs: list[plt.Figure] = []
    with figure_repro_context(FigureReproConfig(font_size=16, line_width=2.0)):
        fig1, ax1 = plt.subplots()
        ax1.plot(d["x"], d["y8"], "-", color=(0, 0, 0.5))
        ax1.plot(d["x"], np.ones_like(d["x"]), "k--", linewidth=1)
        ax1.plot(d["x"], -np.ones_like(d["x"]), "k--", linewidth=1)
        ax1.plot([d["x0"]], [d["R"]], "ko", linewidth=2)
        ax1.text(float(d["x0"]) - 0.15, float(d["R"]), r"$(x_0,R)$")
        ax1.set_xlim(float(np.min(d["x"])), float(np.max(d["x"]) + 0.1))
        ax1.set_ylim(-2, 10)
        ax1.set_xlabel(r"$x$")
        figs.append(fig1)

        fig2, ax2 = plt.subplots()
        th_deg = np.rad2deg(d["th"])
        ax2.plot(th_deg, d["b8"], "-", color=(0, 0, 0.5))
        ax2.plot(th_deg, (1.0 / d["R"]) * np.ones_like(th_deg), "k--", linewidth=1)
        ax2.plot(th_deg, -(1.0 / d["R"]) * np.ones_like(th_deg), "k--", linewidth=1)
        ax2.set_xlim(float(np.min(th_deg)), float(np.max(th_deg)))
        ax2.set_ylim(-0.2, 1.2)
        ax2.set_xlabel(r"$\theta$ (degrees)")
        figs.append(fig2)

        for fig in figs:
            fig.tight_layout()
        if show:
            plt.show()
    return d, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
