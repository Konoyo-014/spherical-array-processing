#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.types import FigureReproConfig


def _make_fig1():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([0, 1], [0, 0], [0, 0], linewidth=2, color="k")
    ax.plot([0, 0], [0, 1], [0, 0], linewidth=2, color="k")
    ax.plot([0, 0], [0, 0], [0, 1], linewidth=2, color="k")
    ax.text(1.15, 0.07, -0.03, r"$x$", fontsize=24)
    ax.text(-0.02, 1.38, -0.13, r"$y$", fontsize=24)
    ax.text(-0.12, 0.0, 1.3, r"$z$", fontsize=24)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    ax.view_init(elev=30.0, azim=-37.5)
    return fig, ax


def _make_fig2():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 0], linewidth=2, color="k")
    ax.plot([0, 0], [0, 1], linewidth=3, color="k")
    ax.text(1.06, 0.01, r"$x$", fontsize=24)
    ax.text(-0.09, 1.20, r"$y$", fontsize=24)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def _make_fig3():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 0], linewidth=2, color="k")
    ax.plot([1, 1], [0, 1], linewidth=2, color="k")
    ax.text(-0.25, 0.02, r"$x$", fontsize=24)
    ax.text(0.93, 1.20, r"$z$", fontsize=24)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def _make_fig4():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 0], linewidth=2, color="k")
    ax.plot([0, 0], [0, 1], linewidth=3, color="k")
    ax.text(1.05, -0.01, r"$y$", fontsize=24)
    ax.text(-0.07, 1.18, r"$z$", fontsize=24)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def main(show: bool = True):
    with figure_repro_context(FigureReproConfig(font_size=18, line_width=2.0)):
        figs_axes = [_make_fig1(), _make_fig2(), _make_fig3(), _make_fig4()]
        for fig, _ in figs_axes:
            fig.tight_layout()
        if show:
            plt.show()
    return figs_axes


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
