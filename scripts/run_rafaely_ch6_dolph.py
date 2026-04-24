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
from spherical_array_processing.beamforming.fixed import axisymmetric_pattern
from spherical_array_processing.repro.politis import beamWeightsDolphChebyshev2Spherical
from spherical_array_processing.types import FigureReproConfig


def _db(x: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    y = 20 * np.log10(np.maximum(np.abs(x), 10 ** (floor_db / 20.0)))
    return y


def compute_dolph_demo():
    theta = np.linspace(-np.pi, np.pi, 4001)
    # Plot 1: sidelobe level 25 dB
    sidelobe_amp = 10 ** (-25 / 20)
    b1a = beamWeightsDolphChebyshev2Spherical(4, "sidelobe", sidelobe_amp)
    b1b = beamWeightsDolphChebyshev2Spherical(9, "sidelobe", sidelobe_amp)
    y1a = axisymmetric_pattern(theta, b1a)
    y1b = axisymmetric_pattern(theta, b1b)

    # Plot 2: mainlobe width parameter theta0 = 45 deg
    theta0 = np.deg2rad(45.0)
    b2a = beamWeightsDolphChebyshev2Spherical(4, "width", theta0)
    b2b = beamWeightsDolphChebyshev2Spherical(9, "width", theta0)
    y2a = axisymmetric_pattern(theta, b2a)
    y2b = axisymmetric_pattern(theta, b2b)
    return {
        "theta": theta,
        "plot1": {"N4": y1a, "N9": y1b},
        "plot2": {"N4": y2a, "N9": y2b},
        "theta0_deg": 45.0,
        "sidelobe_db": 25.0,
    }


def main(show: bool = True):
    data = compute_dolph_demo()
    theta_deg = np.rad2deg(data["theta"])
    figs: list[plt.Figure] = []
    with figure_repro_context(FigureReproConfig(font_size=14, line_width=2.0)):
        fig1, ax1 = plt.subplots()
        ax1.plot(theta_deg, _db(data["plot1"]["N4"]), "-", color=(0, 0, 0.5), linewidth=2, label="N=4")
        ax1.plot(theta_deg, _db(data["plot1"]["N9"]), "--", color="k", linewidth=1.5, label="N=9")
        ax1.set_xlabel(r"$\theta$ (degrees)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_xlim(-180, 180)
        ax1.set_ylim(-35, 5)
        ax1.legend()
        fig1.tight_layout()
        figs.append(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(theta_deg, _db(data["plot2"]["N4"]), "-", color=(0, 0, 0.5), linewidth=2, label="N=4")
        ax2.plot(theta_deg, _db(data["plot2"]["N9"]), "--", color="k", linewidth=1.5, label="N=9")
        ax2.set_xlabel(r"$\theta$ (degrees)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlim(-180, 180)
        ax2.set_ylim(-80, 5)
        ax2.legend()
        fig2.tight_layout()
        figs.append(fig2)

        if show:
            plt.show()
    return data, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
