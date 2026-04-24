#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from spherical_array_processing.acoustics import besseljs
from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.repro.rafaely import uniform_sampling
from spherical_array_processing.types import FigureReproConfig


def _safe_db(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 10 * np.log10(np.maximum(np.real_if_close(x), floor))


def compute_wng_di_example(order: int = 4, n_points: int = 1024):
    a, th, ph = uniform_sampling(order)
    q = len(a)
    kr = np.linspace(0.0, order + 1.0, n_points)
    n = np.arange(order + 1)[:, None]
    w = (2 * np.arange(order + 1) + 1)[:, None]

    bn = np.zeros((order + 1, n_points), dtype=np.complex128)
    for nn in range(order + 1):
        bn[nn, :] = 4 * np.pi * ((1j) ** nn) * besseljs(nn, kr)

    eps = 1e-12
    dn_max_di = np.ones((order + 1, n_points), dtype=np.complex128)
    num_max_di = np.abs(np.sum(w * dn_max_di, axis=0)) ** 2
    den_di_max_di = np.sum(w * np.abs(dn_max_di) ** 2, axis=0)
    den_wng_max_di = np.sum(w * np.abs(dn_max_di / np.where(np.abs(bn) < eps, eps + 0j, bn)) ** 2, axis=0)
    di_max_di = num_max_di / np.maximum(den_di_max_di, eps)
    wng_max_di = (q / ((4 * np.pi) ** 2)) * num_max_di / np.maximum(den_wng_max_di, eps)

    dn_max_wng = np.abs(bn) ** 2
    num_max_wng = np.abs(np.sum(w * dn_max_wng, axis=0)) ** 2
    den_di_max_wng = np.sum(w * np.abs(dn_max_wng) ** 2, axis=0)
    den_wng_max_wng = np.sum(w * np.abs(dn_max_wng / np.where(np.abs(bn) < eps, eps + 0j, bn)) ** 2, axis=0)
    di_max_wng = num_max_wng / np.maximum(den_di_max_wng, eps)
    wng_max_wng = (q / ((4 * np.pi) ** 2)) * num_max_wng / np.maximum(den_wng_max_wng, eps)

    return {
        "order": order,
        "Q": q,
        "kr": kr,
        "DI_maxDI": di_max_di,
        "DI_maxWNG": di_max_wng,
        "WNG_maxDI": wng_max_di,
        "WNG_maxWNG": wng_max_wng,
    }


def main(show: bool = True, order: int = 4):
    d = compute_wng_di_example(order=order)
    figs: list[plt.Figure] = []
    with figure_repro_context(FigureReproConfig(font_size=16, line_width=2.0)):
        fig1, ax1 = plt.subplots()
        ax1.plot(d["kr"], _safe_db(d["DI_maxDI"]), "-", linewidth=2, color=(0, 0, 0.5), label="Max DI")
        ax1.plot(d["kr"], _safe_db(d["DI_maxWNG"]), "--", linewidth=1.5, color="k", label="Max WNG")
        ax1.set_xlim(0, float(np.max(d["kr"])))
        ax1.set_ylim(0, 20)
        ax1.set_xlabel(r"$kr$")
        ax1.set_ylabel(r"$DI\,$ (dB)")
        ax1.legend()
        fig1.tight_layout()
        figs.append(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(d["kr"], _safe_db(d["WNG_maxDI"]), "-", linewidth=2, color=(0, 0, 0.5), label="Max DI")
        ax2.plot(d["kr"], _safe_db(d["WNG_maxWNG"]), "--", linewidth=1.5, color="k", label="Max WNG")
        ax2.set_xlim(0, float(np.max(d["kr"])))
        ax2.set_ylim(-30, 30)
        ax2.set_xlabel(r"$kr$")
        ax2.set_ylabel(r"$WNG\,\,$ (dB)")
        ax2.legend()
        fig2.tight_layout()
        figs.append(fig2)

        if show:
            plt.show()
    return d, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
