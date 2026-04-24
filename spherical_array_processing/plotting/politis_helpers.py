from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from ..coords import sph_to_cart


def plot_mic_array(mic_dirs_deg: ArrayLike, radius_m: float, ax: Axes | None = None) -> Axes:
    dirs = np.asarray(mic_dirs_deg, dtype=float)
    if dirs.ndim != 2 or dirs.shape[1] != 2:
        raise ValueError("mic_dirs_deg must be [M,2] in [az_deg, el_deg]")
    az = np.deg2rad(dirs[:, 0])
    el = np.deg2rad(dirs[:, 1])
    x, y, z = sph_to_cart(az, el, radius_m, convention="az_el")
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=25)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.2 * radius_m
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax


def plot_directional_map_from_grid(
    fgrid: ArrayLike,
    azi_res_deg: float,
    polar_res_deg: float,
    ax: Axes | None = None,
    polar_or_elev: str = "elev",
    zeroed_or_centered: str = "centered",
) -> Axes:
    vals = np.asarray(fgrid, dtype=float).reshape(-1)
    n_azi = int(round(360 / azi_res_deg)) + 1
    n_pol = int(round(180 / polar_res_deg)) + 1
    if vals.size != n_azi * n_pol:
        # fallback: infer square-ish image and display directly
        side = int(np.sqrt(vals.size))
        img = vals[: side * side].reshape(side, side)
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(img, origin="lower", aspect="auto")
        return ax
    img = vals.reshape(n_pol, n_azi)
    if zeroed_or_centered == "centered":
        img = np.roll(img, n_azi // 2, axis=1)
    if ax is None:
        _, ax = plt.subplots()
    extent_y = (-90, 90) if polar_or_elev == "elev" else (0, 180)
    ax.imshow(img, origin="lower", aspect="auto", extent=[-180, 180, extent_y[0], extent_y[1]])
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)" if polar_or_elev == "elev" else "Polar (deg)")
    return ax

