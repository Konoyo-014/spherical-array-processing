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


def compute_sinc_and_cap_data():
    axis_font_size = 16
    orders = np.array([8, 20], dtype=int)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 500)
    z = np.cos(theta)
    s1 = np.zeros((orders.size, z.size), dtype=float)
    for i, n in enumerate(orders):
        num = eval_legendre(n + 1, z) - eval_legendre(n, z)
        den = np.where(np.abs(z - 1.0) < 1e-12, np.nan, (z - 1.0))
        s1[i, :] = ((n + 1) / (4 * np.pi)) * (num / den)
        mask = ~np.isfinite(s1[i, :])
        if np.any(mask):
            # removable singularity at z=1; use nearest finite value for plotting stability
            finite = np.flatnonzero(np.isfinite(s1[i, :]))
            if finite.size:
                s1[i, mask] = s1[i, finite[0]]

    N = 20
    alpha = np.deg2rad(np.array([15.0, 45.0]))
    s2 = np.zeros((alpha.size, N + 1), dtype=float)
    s2[:, 0] = np.sqrt(np.pi) * (1 - np.cos(alpha))
    for n in range(1, N + 1):
        s2[:, n] = np.sqrt(np.pi / (2 * n + 1)) * (eval_legendre(n - 1, np.cos(alpha)) - eval_legendre(n + 1, np.cos(alpha)))

    th = np.linspace(0.0, np.pi / 2, 200)
    f1 = (th <= alpha[0]).astype(float)
    f2 = (th <= alpha[1]).astype(float)
    return {
        "axis_font_size": axis_font_size,
        "orders": orders,
        "Theta": theta,
        "S1": s1,
        "Ncap": N,
        "alpha": alpha,
        "S2": s2,
        "TH": th,
        "f1": f1,
        "f2": f2,
    }


def main(show: bool = True):
    d = compute_sinc_and_cap_data()
    fs = d["axis_font_size"]
    figs: list[plt.Figure] = []
    with figure_repro_context(FigureReproConfig(font_size=fs, line_width=2.0)):
        fig1, ax1 = plt.subplots()
        ax1.plot(np.rad2deg(d["Theta"]), d["S1"][0], "--", linewidth=1.5, color="k")
        ax1.plot(np.rad2deg(d["Theta"]), d["S1"][1], "-", linewidth=2, color=(0, 0, 0.5))
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel(r"$\Theta$ (degrees)")
        ax1.legend(["N=8", "N=20"])
        ax1.set_xlim(-90, 90)
        ax1.set_ylim(-10, 40)
        fig1.tight_layout()
        figs.append(fig1)

        fig2, ax2 = plt.subplots()
        n_axis = np.arange(0, d["Ncap"] + 1)
        ax2.plot(n_axis, d["S2"][0], "x--", linewidth=1.5, color="k", markersize=8)
        ax2.plot(n_axis, d["S2"][1], "o-", linewidth=2, color=(0, 0, 0.5), markersize=6)
        ax2.set_ylabel("Amplitude")
        ax2.set_xlabel(r"$n$")
        ax2.legend([r"$\alpha=15^\circ$", r"$\alpha=45^\circ$"])
        fig2.tight_layout()
        figs.append(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(np.rad2deg(d["TH"]), d["f1"], "--", linewidth=1.5, color="k")
        ax3.plot(np.rad2deg(d["TH"]), d["f2"], "-", linewidth=2, color=(0, 0, 0.5))
        ax3.set_ylabel("Amplitude")
        ax3.set_xlabel(r"$\theta$ (degrees)")
        ax3.legend([r"$\alpha=15^\circ$", r"$\alpha=45^\circ$"])
        ax3.set_xlim(0, 90)
        ax3.set_ylim(0, 1.1)
        fig3.tight_layout()
        figs.append(fig3)

        if show:
            plt.show()
    return d, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
