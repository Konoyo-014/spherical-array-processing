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


def main(show: bool = True):
    with figure_repro_context():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        u = np.linspace(0.0, 2 * np.pi, 17)
        v = np.linspace(0.0, np.pi, 17)
        uu, vv = np.meshgrid(u, v)
        x = np.cos(uu) * np.sin(vv)
        y = np.sin(uu) * np.sin(vv)
        z = np.cos(vv)
        surf = ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            color="white",
            edgecolor=(0.0, 0.0, 0.5),
            linewidth=1.5,
            antialiased=True,
            alpha=0.8,
        )
        surf.set_edgecolor((0.0, 0.0, 0.5))
        ax.set_box_aspect((1, 1, 1))
        ax.axis("off")
        try:
            ax.dist = 8  # approximate MATLAB zoom(1) appearance on older mpl
        except Exception:
            pass
        fig.tight_layout()
        if show:
            plt.show()
    return fig, ax


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
