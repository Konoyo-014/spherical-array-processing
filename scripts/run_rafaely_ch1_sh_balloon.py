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
from spherical_array_processing.repro.rafaely import c2s, s2c, sh2
from spherical_array_processing.types import FigureReproConfig


VIEW_PRESETS = {
    "angle": (-37.5, 30.0),
    "z": (0.0, 90.0),
    "x": (90.0, 0.0),
    "y": (180.0, 0.0),
}


def compute_sh_balloon_grid(max_order: int = 4, resolution: int = 40):
    u = np.linspace(0.0, 2 * np.pi, resolution + 1)
    v = np.linspace(0.0, np.pi, resolution + 1)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    th, ph, _ = c2s(x, y, z)
    thf = th.reshape(-1)
    phf = ph.reshape(-1)
    ysh = sh2(max_order, thf, phf)
    return {"x": x, "y": y, "z": z, "th": th, "ph": ph, "Y": ysh, "resolution": resolution}


def _panel_surface(ax, th, ph, c):
    X, Y, Z = s2c(th, ph, np.abs(c))
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=plt.cm.cool((np.sign(c) + 1) / 2),
        linewidth=0.15,
        antialiased=True,
        shade=False,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")


def main(show: bool = True, max_order: int = 4, resolution: int = 40, views: tuple[str, ...] = ("angle", "z", "x", "y")):
    grid = compute_sh_balloon_grid(max_order=max_order, resolution=resolution)
    th = grid["th"]
    ph = grid["ph"]
    Y = grid["Y"]
    figs = []
    with figure_repro_context(FigureReproConfig(font_size=8, line_width=1.0)):
        for view_name in views:
            if view_name not in VIEW_PRESETS:
                raise ValueError(f"Unknown view preset: {view_name}")
            azim, elev = VIEW_PRESETS[view_name]
            fig = plt.figure(figsize=(14, 8))
            for n in range(max_order + 1):
                for m in range(-n, n + 1):
                    idx = 9 * n + 5 + m
                    ax = fig.add_subplot(5, 9, idx, projection="3d")
                    q = n * n + n + m
                    comp = np.imag(Y[q, :]) if m < 0 else np.real(Y[q, :])
                    c = comp.reshape(th.shape)
                    _panel_surface(ax, th, ph, c)
                    ax.view_init(elev=elev, azim=azim)
            fig.tight_layout(pad=0.05)
            figs.append(fig)
        if show:
            plt.show()
    return grid, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
