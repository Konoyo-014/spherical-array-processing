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
from spherical_array_processing.repro.rafaely import wigner_d_matrix
from spherical_array_processing.types import FigureReproConfig

from scripts.run_rafaely_ch1_truncated_cap import _balloon_on_axis, spherical_cap_fnm


def compute_rotation_data(order: int = 2, alpha_deg: float = 30.0):
    alpha = np.deg2rad(alpha_deg)
    fnm = spherical_cap_fnm(order, alpha, add_dc=0.0)
    D1 = wigner_d_matrix(order, 0.0, np.deg2rad(45.0), 0.0)
    D2 = wigner_d_matrix(order, np.deg2rad(180.0), np.deg2rad(45.0), 0.0)
    D3 = wigner_d_matrix(order, 0.0, np.deg2rad(180.0), 0.0)
    fnm1 = D1 @ fnm
    fnm2 = D2 @ fnm
    fnm3 = D3 @ fnm
    return {"order": order, "alpha_deg": alpha_deg, "fnm": fnm, "fnm1": fnm1, "fnm2": fnm2, "fnm3": fnm3}


def main(show: bool = True):
    d = compute_rotation_data()
    axis_font_size = 12
    figs: list[plt.Figure] = []
    with figure_repro_context(FigureReproConfig(font_size=axis_font_size, line_width=1.5)):
        fig = plt.figure(figsize=(10, 8))
        titles = [
            "Original",
            r"$\Lambda(0,45^\circ,0)$",
            r"$\Lambda(180^\circ,45^\circ,0)$",
            r"$\Lambda(0,180^\circ,0)$",
        ]
        coeffs = [d["fnm"], d["fnm1"], d["fnm2"], d["fnm3"]]
        for i, (title, fnm) in enumerate(zip(titles, coeffs), start=1):
            ax = fig.add_subplot(2, 2, i, projection="3d")
            _balloon_on_axis(ax, fnm, resolution=36, viewangle=(180.0, 0.0), transparency=0.05)
            ax.set_title(title, fontsize=axis_font_size)
        fig.tight_layout()
        figs.append(fig)
        if show:
            plt.show()
    return d, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
