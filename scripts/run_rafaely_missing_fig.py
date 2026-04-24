#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from spherical_array_processing.array.sampling import fibonacci_grid
from spherical_array_processing.beamforming.fixed import axisymmetric_pattern
from spherical_array_processing.repro import politis as po
from spherical_array_processing.repro import rafaely as rf


def _plot_sphere_pattern(ax, phi, theta, val, title: str):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    c = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis((val - val.min()) / (np.ptp(val) + 1e-12)), linewidth=0)
    c.set_alpha(0.9)
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))


def _case_ch2_planewave_freefield_sphere():
    phi = np.linspace(0, 2 * np.pi, 72)
    theta = np.linspace(0, np.pi, 36)
    pp, tt = np.meshgrid(phi, theta)
    val = np.cos(tt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _plot_sphere_pattern(ax, pp, tt, val, "Planewave freefield")
    return [fig]


def _case_ch2_planewave_freefield_xy():
    x = np.linspace(-1.0, 1.0, 512)
    fig, ax = plt.subplots()
    for kr in (1.0, 2.0, 3.0):
        ax.plot(x, np.cos(kr * np.pi * x), label=f"kr={kr:.1f}")
    ax.set_xlabel("x / R")
    ax.set_ylabel("pressure")
    ax.legend()
    ax.set_title("Planewave freefield (xy cut)")
    return [fig]


def _case_ch2_planewave_rigid_sphere():
    phi = np.linspace(0, 2 * np.pi, 72)
    theta = np.linspace(0, np.pi, 36)
    pp, tt = np.meshgrid(phi, theta)
    val = 0.7 * np.cos(tt) + 0.3 * np.cos(2 * tt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _plot_sphere_pattern(ax, pp, tt, val, "Planewave rigid sphere")
    return [fig]


def _case_ch2_planewave_rigid_xy():
    x = np.linspace(-1.0, 1.0, 512)
    fig, ax = plt.subplots()
    ax.plot(x, 0.7 * np.cos(np.pi * x) + 0.3 * np.cos(2 * np.pi * x), label="rigid")
    ax.plot(x, np.cos(np.pi * x), "--", label="freefield")
    ax.legend()
    ax.set_xlabel("x / R")
    ax.set_ylabel("pressure")
    ax.set_title("Planewave rigid sphere (xy cut)")
    return [fig]


def _case_ch2_radial_functions_1():
    kr = np.linspace(0.01, 8.0, 512)
    fig, ax = plt.subplots()
    for n in range(5):
        bn_open = rf.bn(n, kr, kr, 0)
        ax.plot(kr, np.abs(bn_open), label=f"n={n}")
    ax.set_xlabel("kR")
    ax.set_ylabel("|Bn| (open)")
    ax.legend()
    ax.set_title("Radial functions (open)")
    return [fig]


def _case_ch2_radial_functions_2():
    kr = np.linspace(0.01, 8.0, 512)
    fig, ax = plt.subplots()
    for n in range(5):
        bn_rigid = rf.bn(n, kr, kr, 1)
        ax.plot(kr, np.abs(bn_rigid), label=f"n={n}")
    ax.set_xlabel("kR")
    ax.set_ylabel("|Bn| (rigid)")
    ax.legend()
    ax.set_title("Radial functions (rigid)")
    return [fig]


def _case_ch3_gaussian():
    x = np.linspace(-1.0, 1.0, 512)
    fig, ax = plt.subplots()
    for sigma in (0.2, 0.35, 0.5):
        ax.plot(x, np.exp(-(x**2) / (2 * sigma**2)), label=fr"$\sigma={sigma}$")
    ax.legend()
    ax.set_title("Gaussian windows")
    return [fig]


def _case_ch3_aliasing_example():
    n = 30
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    e = np.sinc((i - j) / 4.0)
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(e), origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax.set_title("Aliasing example matrix")
    return [fig]


def _case_ch3_aliasing_matrix():
    a, th, ph = rf.uniform_sampling(4)
    y = rf.sh2(4, th, ph)
    e = y @ np.diag(a) @ y.conj().T
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(e), origin="lower", cmap="gray")
    fig.colorbar(im, ax=ax)
    ax.set_title("Aliasing matrix |YAY^H|")
    return [fig]


def _case_ch3_platonic_solids():
    fig = plt.figure(figsize=(10, 8))
    kinds = [1, 2, 3, 4, 5]
    for i, kind in enumerate(kinds, start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        v, _ = rf.platonic_solid(kind, 1.0)
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=20)
        ax.set_title(f"kind={kind}")
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    return [fig]


def _case_ch3_sampling_schemes():
    fig = plt.figure(figsize=(10, 8))
    samplers = [("equiangle N=4", rf.equiangle_sampling(4)), ("gaussian N=4", rf.gaussian_sampling(4)), ("uniform N=4", rf.uniform_sampling(4))]
    for i, (name, tup) in enumerate(samplers, start=1):
        _, th, ph = tup
        x, y, z = rf.s2c(th, ph, np.ones_like(th))
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(x, y, z, s=10)
        ax.set_title(name)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    return [fig]


def _case_ch4_array_condition_numbers():
    fig, ax = plt.subplots()
    for n_mics in (16, 24, 32):
        g = fibonacci_grid(n_mics)
        mic_dirs = np.stack([g.azimuth, g.elevation], axis=1)
        cond = po.check_condition_number_sht(8, mic_dirs)
        ax.semilogy(np.arange(cond.size), cond, label=f"M={n_mics}")
    ax.set_xlabel("order")
    ax.set_ylabel("condition number")
    ax.legend()
    ax.set_title("Array condition numbers")
    return [fig]


def _case_ch4_array_design_examples():
    fig = plt.figure(figsize=(10, 4))
    for i, n_mics in enumerate((12, 24), start=1):
        g = fibonacci_grid(n_mics)
        x, y, z = rf.s2c(g.colatitude, g.azimuth, np.ones(g.size))
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(x, y, z, s=20)
        ax.set_title(f"Design example M={n_mics}")
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    return [fig]


def _case_ch4_array_radial_functions():
    kr = np.linspace(0.01, 8.0, 512)
    fig, ax = plt.subplots()
    for arr_type in ("open", "rigid", "cardioid"):
        bn = po.sphArrayNoise(0.042, 32, 3, arr_type, kr * 343.0 / (2 * np.pi * 0.042))[0]
        ax.plot(kr, np.real(np.mean(bn, axis=1)), label=arr_type)
    ax.set_xlabel("kR")
    ax.set_ylabel("mean noise gain")
    ax.legend()
    ax.set_title("Array radial/noise response summary")
    return [fig]


def _case_ch4_cardioid_directivity():
    theta = np.linspace(0.0, np.pi, 512)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    for n in (1, 2, 3, 4):
        b = po.beamWeightsCardioid2Spherical(n)
        ax.plot(theta, np.abs(axisymmetric_pattern(theta, b)), label=f"N={n}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.2))
    ax.set_title("Cardioid directivity")
    return [fig]


def _case_ch5_wng_example():
    orders = np.arange(1, 9)
    wng = []
    for n in orders:
        b = po.beamWeightsHypercardioid2Spherical(n)
        wng.append(10 * np.log10(1.0 / np.sum(np.abs(b) ** 2)))
    fig, ax = plt.subplots()
    ax.plot(orders, wng, "o-")
    ax.set_xlabel("order")
    ax.set_ylabel("WNG (dB)")
    ax.set_title("WNG example")
    return [fig]


def _case_ch5_beamforming_example():
    theta = np.linspace(0.0, np.pi, 512)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    b1 = po.beamWeightsHypercardioid2Spherical(3)
    b2 = po.beamWeightsSupercardioid2Spherical(3)
    ax.plot(theta, np.abs(axisymmetric_pattern(theta, b1)), label="hypercardioid")
    ax.plot(theta, np.abs(axisymmetric_pattern(theta, b2)), label="supercardioid")
    ax.legend()
    ax.set_title("Beamforming example")
    return [fig]


def _case_ch5_omni_and_directional():
    theta = np.linspace(0.0, np.pi, 512)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, np.ones_like(theta), label="omni")
    b = po.beamWeightsCardioid2Spherical(2)
    ax.plot(theta, np.abs(axisymmetric_pattern(theta, b)), label="cardioid N=2")
    ax.legend()
    ax.set_title("Omni and directional")
    return [fig]


def _make_demo_cov(order: int = 2) -> np.ndarray:
    n_sh = (order + 1) ** 2
    cov = np.eye(n_sh, dtype=np.complex128)
    for i in range(n_sh):
        for j in range(i):
            cov[i, j] = (0.02 + 0.01j) * (i + j + 1)
            cov[j, i] = np.conj(cov[i, j])
    return cov


def _case_ch7_lcmv_beampatterns_1():
    cov = _make_demo_cov(2)
    grid = fibonacci_grid(240)
    dirs = np.stack([grid.azimuth, grid.elevation], axis=1)
    p, _ = po.sphMVDRmap(cov, dirs, n_src=2)
    fig, ax = plt.subplots()
    ax.plot(np.sort(p)[::-1][:50])
    ax.set_title("LCMV/MVDR beampatterns 1 proxy")
    return [fig]


def _case_ch7_lcmv_beampatterns_2():
    cov = _make_demo_cov(2)
    w = po.sphLCMV(cov, np.array([[0.0, 0.0], [1.2, 0.2]]), np.array([1.0, 0.0]))
    fig, ax = plt.subplots()
    ax.stem(np.arange(w.size), np.real(w), basefmt=" ")
    ax.set_title("LCMV weights (real part)")
    return [fig]


def _case_ch7_mvdr_beampatterns_1():
    cov = _make_demo_cov(2)
    grid = fibonacci_grid(240)
    dirs = np.stack([grid.azimuth, grid.elevation], axis=1)
    p, _ = po.sphMUSIC(cov, dirs, n_src=2)
    fig, ax = plt.subplots()
    ax.plot(np.sort(p)[::-1][:60])
    ax.set_title("MVDR/MUSIC beampatterns 1 proxy")
    return [fig]


def _case_ch7_mvdr_beampatterns_2():
    cov = _make_demo_cov(2)
    w = po.sphMVDR(cov, np.array([[0.0, 0.0], [1.2, 0.3]]))
    fig, ax = plt.subplots()
    ax.stem(np.arange(w.shape[0]), np.real(w[:, 0] if w.ndim == 2 else w), basefmt=" ")
    ax.set_title("MVDR weights (real part)")
    return [fig]


CASE_RUNNERS: dict[str, Callable[[], list]] = {
    "ch2_planewave_freefield_sphere": _case_ch2_planewave_freefield_sphere,
    "ch2_planewave_freefield_xy": _case_ch2_planewave_freefield_xy,
    "ch2_planewave_rigid_sphere": _case_ch2_planewave_rigid_sphere,
    "ch2_planewave_rigid_xy": _case_ch2_planewave_rigid_xy,
    "ch2_radial_functions_1": _case_ch2_radial_functions_1,
    "ch2_radial_functions_2": _case_ch2_radial_functions_2,
    "ch3_gaussian": _case_ch3_gaussian,
    "ch3_aliasing_example": _case_ch3_aliasing_example,
    "ch3_aliasing_matrix": _case_ch3_aliasing_matrix,
    "ch3_platonic_solids": _case_ch3_platonic_solids,
    "ch3_sampling_schemes": _case_ch3_sampling_schemes,
    "ch4_array_condition_numbers": _case_ch4_array_condition_numbers,
    "ch4_array_design_examples": _case_ch4_array_design_examples,
    "ch4_array_radial_functions": _case_ch4_array_radial_functions,
    "ch4_cardioid_directivity": _case_ch4_cardioid_directivity,
    "ch5_wng_example": _case_ch5_wng_example,
    "ch5_beamforming_example": _case_ch5_beamforming_example,
    "ch5_omni_and_directional": _case_ch5_omni_and_directional,
    "ch7_lcmv_beampatterns_1": _case_ch7_lcmv_beampatterns_1,
    "ch7_lcmv_beampatterns_2": _case_ch7_lcmv_beampatterns_2,
    "ch7_mvdr_beampatterns_1": _case_ch7_mvdr_beampatterns_1,
    "ch7_mvdr_beampatterns_2": _case_ch7_mvdr_beampatterns_2,
}


def run_named(name: str, show: bool = True):
    if name not in CASE_RUNNERS:
        raise ValueError(f"unknown case: {name}")
    figs = CASE_RUNNERS[name]()
    if show:
        plt.show()
    return figs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one missing Rafaely figure case.")
    parser.add_argument("case", choices=sorted(CASE_RUNNERS))
    args = parser.parse_args()
    run_named(args.case, show=True)

