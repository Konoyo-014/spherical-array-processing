from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from ...array.sampling import equiangle_sampling as equiangle_grid
from ...coords import sph_to_cart
from .math import c2s, s2c, sh2


def plot_balloon(fnm: ArrayLike, viewangle: tuple[float, float] | None = None, transparency: float | None = None):
    fnm = np.asarray(fnm, dtype=np.complex128).reshape(-1)
    if transparency is None:
        transparency = 0.01
    if viewangle is None:
        viewangle = (-37.5, 30.0)
    N = int(round(np.sqrt(fnm.size) - 1))
    if (N + 1) ** 2 != fnm.size:
        raise ValueError("fnm length must be (N+1)^2")
    n = 100
    u = np.linspace(0, 2 * np.pi, n + 1)
    v = np.linspace(0, np.pi, n + 1)
    uu, vv = np.meshgrid(u, v)
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    th, ph, _ = c2s(x, y, z)
    thp = th.reshape(-1)
    php = ph.reshape(-1)
    Yp = sh2(N, thp, php)
    f = fnm @ Yp
    F = f.reshape(th.shape)
    X, Y, Z = s2c(th, ph, np.abs(F))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = np.sign(np.real(F))
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.cool((colors + 1) / 2), linewidth=0.2, antialiased=True)
    surf.set_alpha(max(0.0, 1.0 - transparency))
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    ax.view_init(elev=viewangle[1], azim=viewangle[0])
    return ax


def plot_sphere(fnm: ArrayLike):
    fnm = np.asarray(fnm, dtype=np.complex128).reshape(-1)
    N = int(round(np.sqrt(fnm.size) - 1))
    grid = equiangle_grid(max(N, 8))
    th = grid.colatitude
    ph = grid.azimuth
    Y = sh2(N, th, ph)
    f = (fnm @ Y).reshape(-1)
    x, y, z = sph_to_cart(ph, (np.pi / 2) - th, 1.0, convention="az_el")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=np.real(f), s=12, cmap="coolwarm")
    fig.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    return ax


def plot_contour(fnm: ArrayLike, normalization: int | bool = 0, absolute: int | bool = 1):
    fnm = np.asarray(fnm, dtype=np.complex128).reshape(-1)
    N = int(round(np.sqrt(fnm.size) - 1))
    Np = 29
    n_side = 2 * (Np + 1)
    phi_vals = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    theta_colat_vals = np.linspace(0, np.pi, n_side, endpoint=True)
    ph_grid, th_colat_grid = np.meshgrid(phi_vals, theta_colat_vals, indexing="xy")
    ph = ph_grid.reshape(-1)
    th = th_colat_grid.reshape(-1)
    Yp = sh2(N, th, ph)
    f = fnm @ Yp
    if normalization:
        f = f / np.maximum(np.max(np.abs(f)), 1e-12)
    if absolute:
        f = np.abs(f)
    n_colat = n_side
    n_azi = n_side
    F = np.asarray(f).reshape(n_colat, n_azi).T
    if np.max(np.abs(np.imag(F))) < 1e-10:
        F = np.real(F)
    phi_deg = np.rad2deg(phi_vals)
    the_deg = np.rad2deg((np.pi / 2) - theta_colat_vals)
    fig, ax = plt.subplots()
    c = ax.contourf(phi_deg, the_deg, F, levels=20, cmap="bone_r")
    fig.colorbar(c, ax=ax)
    ax.set_xlabel(r"$\phi$ (degrees)")
    ax.set_ylabel(r"$\theta$ (degrees)")
    return ax


def plot_sampling(theta: ArrayLike, phi: ArrayLike):
    th = np.asarray(theta, dtype=float).reshape(-1)
    ph = np.asarray(phi, dtype=float).reshape(-1)
    x, y, z = s2c(th, ph, np.ones_like(th))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 48)
    uu, vv = np.meshgrid(u, v)
    X = np.cos(uu) * np.sin(vv)
    Y = np.sin(uu) * np.sin(vv)
    Z = np.cos(vv)
    ax1.plot_surface(X, Y, Z, cmap="bone", alpha=0.2, linewidth=0)
    ax1.scatter(x, y, z, c="k", s=20)
    ax1.set_box_aspect((1, 1, 1))
    ax1.axis("off")

    fig2, ax2 = plt.subplots()
    ax2.plot(np.rad2deg(ph), np.rad2deg(th), ".", color=(0, 0, 0.5))
    ax2.set_xlabel(r"$\phi$ (degrees)")
    ax2.set_ylabel(r"$\theta$ (degrees)")
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 180)
    return ax1, ax2


def plot_aliasing(E: ArrayLike):
    E = np.asarray(E, dtype=float)
    E = np.abs(E)
    E[E < 1e-10] = 0.0
    if np.max(E) > 0:
        E = E / np.max(E)
    nz = E[E != 0]
    norm_factor = np.min(nz) if nz.size else 1.0
    E2 = E + norm_factor / 100.0
    norm_factor_db = np.floor(20 * np.log10(norm_factor) / 10) * 10 if norm_factor > 0 else -120
    clims = (norm_factor_db - 10, 0)
    fig, ax = plt.subplots()
    im = ax.imshow(20 * np.log10(np.abs(E2)), origin="lower", cmap="gray", aspect="equal", vmin=clims[0], vmax=clims[1])
    fig.colorbar(im, ax=ax, label="(dB)")
    ax.set_xlabel(r"$n'^2+n'+m'$")
    ax.set_ylabel(r"$n^2+n+m$")
    return ax
