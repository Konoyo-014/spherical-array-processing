#!/usr/bin/env python3
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from spherical_array_processing.acoustics import bn_matrix
from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.repro.rafaely import sh2, uniform_sampling
from spherical_array_processing.types import FigureReproConfig


def compute_wng_open_and_rigid(order: int = 3, n_points: int = 512):
    a, th, ph = uniform_sampling(order)
    q = len(a)
    kr = np.linspace(0.0, float(order), n_points)
    y0 = sh2(order, np.array([0.0]), np.array([0.0]))[:, 0]  # front look direction in Rafaely convention

    b_open = bn_matrix(order, kr, ka=kr, sphere=0, repeat_per_order=True)  # [K, (N+1)^2]
    v_open = b_open * np.conj(y0)[None, :]
    wng_open = (q / (4 * np.pi)) * np.sum(np.abs(v_open) ** 2, axis=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        b_rigid = bn_matrix(order, kr, ka=kr, sphere=1, repeat_per_order=True)
    v_rigid = b_rigid * np.conj(y0)[None, :]
    wng_rigid = (q / (4 * np.pi)) * np.sum(np.abs(v_rigid) ** 2, axis=1)
    # The rigid-sphere modal expression can be numerically unstable exactly at kr=0.
    wng_rigid = np.asarray(wng_rigid, dtype=np.float64)
    if not np.isfinite(wng_rigid[0]):
        finite_idx = np.flatnonzero(np.isfinite(wng_rigid))
        if finite_idx.size:
            wng_rigid[0] = wng_rigid[finite_idx[0]]

    return {"order": order, "Q": q, "kr": kr, "WNG_open": wng_open, "WNG_rigid": wng_rigid}


def main(show: bool = True, order: int = 3):
    d = compute_wng_open_and_rigid(order=order)
    with figure_repro_context(FigureReproConfig(font_size=16, line_width=2.0)):
        fig, ax = plt.subplots()
        ax.plot(d["kr"], 10 * np.log10(np.maximum(np.real(d["WNG_open"]), 1e-12)), "-", linewidth=2, color=(0, 0, 0.5), label="Open")
        ax.plot(d["kr"], 10 * np.log10(np.maximum(np.real(d["WNG_rigid"]), 1e-12)), "-", linewidth=1, color="k", label="Rigid")
        ax.plot(d["kr"], np.full_like(d["kr"], 10 * np.log10(d["Q"])), "--", linewidth=2, color=(0, 0, 0.5), label="Q")
        ax.set_xlim(0, float(order))
        ax.set_ylim(14, 19)
        ax.set_xlabel(r"$kr$")
        ax.set_ylabel(r"$WNG\,\,$ (dB)")
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
    return d, fig


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
