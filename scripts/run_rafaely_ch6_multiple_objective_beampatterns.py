#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_legendre

from spherical_array_processing.plotting import figure_repro_context
from spherical_array_processing.repro.rafaely import bn_mat, uniform_sampling
from spherical_array_processing.types import FigureReproConfig


def _legendre_stack(order: int, theta: np.ndarray) -> np.ndarray:
    x = np.cos(theta)
    return np.stack([eval_legendre(n, x) for n in range(order + 1)], axis=0)


def _solve_slsqp_design(vn: np.ndarray, A: np.ndarray, B: np.ndarray, wng_min_lin: float, Vn: np.ndarray | None = None, sl2_max: float | None = None):
    n = vn.size
    x0 = np.ones(n, dtype=float)
    x0 = x0 / np.maximum(vn @ x0, 1e-12)

    def obj(x):
        return float(x @ B @ x)

    cons = [
        {"type": "eq", "fun": lambda x: float(vn @ x - 1.0)},
        {"type": "ineq", "fun": lambda x: float((1.0 / wng_min_lin) - (x @ A @ x))},
    ]
    if Vn is not None and sl2_max is not None:
        cons.append({"type": "ineq", "fun": lambda x, Vn=Vn, sl2_max=sl2_max: sl2_max - (Vn.T @ x) ** 2})

    res = minimize(obj, x0, method="SLSQP", constraints=cons, options={"maxiter": 600, "ftol": 1e-10, "disp": False})
    if not res.success:
        # Try again from a DI-favoring initialization.
        x1 = np.linalg.pinv(B + 1e-8 * np.eye(n)) @ vn
        x1 = x1 / np.maximum(vn @ x1, 1e-12)
        res = minimize(obj, x1, method="SLSQP", constraints=cons, options={"maxiter": 800, "ftol": 1e-10, "disp": False})
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def compute_multiple_objective_beampatterns(
    order: int = 4,
    kr: float = 2.0,
    sphere: int = 1,
    wng_min_db: float = 10.0,
    sidelobe_db: float = -30.0,
    n_sl_angles: int = 50,
    n_plot_angles: int = 1024,
):
    a, th, ph = uniform_sampling(order)
    Q = len(a)
    nn = (2 * np.arange(order + 1) + 1).astype(float)
    vn = (1 / (4 * np.pi)) * nn

    bn_full = bn_mat(order, np.array([kr]), np.array([kr]), sphere)
    acn_m0 = np.array([n * n + n for n in range(order + 1)], dtype=int)
    bn = bn_full[0, acn_m0]
    A = (4 * np.pi / Q) * np.diag(vn) @ np.diag(1.0 / np.maximum(np.abs(bn) ** 2, 1e-12))
    B = (1 / (4 * np.pi)) * np.diag(vn)
    wng_min_lin = 10 ** (wng_min_db / 10.0)
    sl2_max = 10 ** (sidelobe_db / 10.0)

    THi = np.deg2rad(np.linspace(60, 180, n_sl_angles))
    Pn2 = _legendre_stack(order, THi)
    Vn = np.diag(vn) @ Pn2

    dn1 = _solve_slsqp_design(vn, A, B, wng_min_lin)
    dn2 = _solve_slsqp_design(vn, A, B, wng_min_lin, Vn=Vn, sl2_max=sl2_max)

    def metrics(dn: np.ndarray):
        DI = float(np.abs(dn @ vn) ** 2 / np.maximum(dn @ B @ dn, 1e-12))
        WNG = float(np.abs(dn @ vn) ** 2 / np.maximum(dn @ A @ dn, 1e-12))
        SL = float(np.max(np.abs(dn @ Vn)))
        return DI, WNG, SL

    DI1, WNG1, SL1 = metrics(dn1)
    DI2, WNG2, SL2 = metrics(dn2)

    thp = np.deg2rad(np.linspace(-180, 180, n_plot_angles))
    Pn = _legendre_stack(order, thp)
    y1 = dn1 @ np.diag(vn) @ Pn
    y2 = dn2 @ np.diag(vn) @ Pn

    return {
        "order": order,
        "kr": kr,
        "sphere": sphere,
        "Q": Q,
        "vn": vn,
        "dn1": dn1,
        "dn2": dn2,
        "thp": thp,
        "y1": y1,
        "y2": y2,
        "DI1": DI1,
        "WNG1": WNG1,
        "SL1_db": 20 * np.log10(max(SL1, 1e-12)),
        "DI2": DI2,
        "WNG2": WNG2,
        "SL2_db": 20 * np.log10(max(SL2, 1e-12)),
        "wng_min_db": wng_min_db,
        "sidelobe_db": sidelobe_db,
    }


def main(
    show: bool = True,
    order: int = 4,
    kr: float = 2.0,
    sphere: int = 1,
    wng_min_db: float = 10.0,
    sidelobe_db: float = -30.0,
    n_sl_angles: int = 50,
    n_plot_angles: int = 1024,
):
    d = compute_multiple_objective_beampatterns(
        order=order,
        kr=kr,
        sphere=sphere,
        wng_min_db=wng_min_db,
        sidelobe_db=sidelobe_db,
        n_sl_angles=n_sl_angles,
        n_plot_angles=n_plot_angles,
    )
    th_deg = np.rad2deg(d["thp"])
    figs: list[plt.Figure] = []
    with figure_repro_context(FigureReproConfig(font_size=16, line_width=2.0)):
        for y in (d["y1"], d["y2"]):
            fig, ax = plt.subplots()
            ax.plot(th_deg, 20 * np.log10(np.maximum(np.abs(y), 1e-12)), "-", linewidth=2, color=(0, 0, 0.5))
            ax.set_xlabel(r"$\Theta$ (degrees)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-70, 10)
            fig.tight_layout()
            figs.append(fig)
        if show:
            plt.show()
    return d, figs


if __name__ == "__main__":
    raise SystemExit(0 if main(show=True) else 1)
