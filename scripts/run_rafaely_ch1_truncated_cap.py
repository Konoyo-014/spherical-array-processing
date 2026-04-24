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
from spherical_array_processing.repro.rafaely import c2s, s2c, sh2
from spherical_array_processing.types import FigureReproConfig


def spherical_cap_fnm(max_order: int, alpha_rad: float, add_dc: float = 0.0) -> np.ndarray:
    fnm = np.zeros((max_order + 1) ** 2, dtype=np.complex128)
    fnm[0] = np.sqrt(np.pi) * (1 - np.cos(alpha_rad)) + add_dc
    for n in range(1, max_order + 1):
        fn0 = np.sqrt(np.pi / (2 * n + 1)) * (eval_legendre(n - 1, np.cos(alpha_rad)) - eval_legendre(n + 1, np.cos(alpha_rad)))
        q = n * n + n
        fnm[q] = fn0
    return fnm


def _balloon_on_axis(ax, fnm: np.ndarray, resolution: int = 50, viewangle: tuple[float, float] = (-37.5, 30.0), transparency: float = 0.01):
    N = int(round(np.sqrt(fnm.size) - 1))
    u = np.linspace(0.0, 2 * np.pi, resolution + 1)
    v = np.linspace(0.0, np.pi, resolution + 1)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    th, ph, _ = c2s(x, y, z)
    Y = sh2(N, th.reshape(-1), ph.reshape(-1))
    F = (fnm @ Y).reshape(th.shape)
    X, Yc, Z = s2c(th, ph, np.abs(F))
    surf = ax.plot_surface(
        X,
        Yc,
        Z,
        facecolors=plt.cm.cool((np.sign(np.real(F)) + 1) / 2),
        linewidth=0.15,
        antialiased=True,
        shade=False,
    )
    surf.set_alpha(max(0.0, 1.0 - transparency))
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    ax.view_init(elev=viewangle[1], azim=viewangle[0])
    return ax


def compute_truncated_cap_data(max_order: int = 40, alpha_deg: float = 30.0, add_dc: float = 10.0):
    alpha = np.deg2rad(alpha_deg)
    fnm = spherical_cap_fnm(max_order, alpha, add_dc=add_dc)
    return {"max_order": max_order, "alpha_deg": alpha_deg, "add_dc": add_dc, "fnm": fnm}


def main(
    show: bool = True,
    max_order: int = 40,
    alpha_deg: float = 30.0,
    add_dc: float = 10.0,
    orders: tuple[int, ...] = (4, 10, 20, 40),
    resolution: int = 36,
):
    d = compute_truncated_cap_data(max_order=max_order, alpha_deg=alpha_deg, add_dc=add_dc)
    axis_font_size = 14
    figs: list[plt.Figure] = []
    with figure_repro_context(FigureReproConfig(font_size=axis_font_size, line_width=1.5)):
        fig = plt.figure(figsize=(10, 8))
        for i, N in enumerate(orders, start=1):
            ax = fig.add_subplot(2, 2, i, projection="3d")
            _balloon_on_axis(ax, d["fnm"][: (N + 1) ** 2], resolution=resolution, viewangle=(-37.5, 30.0))
            ax.set_title(rf"$N={N}$", fontsize=axis_font_size)
        fig.tight_layout()
        figs.append(fig)
        if show:
            plt.show()
    return d, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
