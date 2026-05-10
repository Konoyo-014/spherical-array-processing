from __future__ import annotations

import math

import numpy as np

from ..types import SphericalGrid


def fibonacci_grid(n_points: int) -> SphericalGrid:
    """Generate a near-uniform spherical grid using the Fibonacci spiral method.

    Parameters
    ----------
    n_points : int
        Number of grid points to distribute on the sphere. Must be positive.

    Returns
    -------
    SphericalGrid
        Grid with ``azimuth``, ``angle2`` (colatitude), and equal-area
        ``weights`` that sum to 4 pi.

    Raises
    ------
    ValueError
        If *n_points* < 1.

    Examples
    --------
    >>> g = fibonacci_grid(100)
    >>> g.azimuth.shape
    (100,)
    >>> bool(abs(g.weights.sum() - 4 * 3.141592653589793) < 1e-10)
    True
    """
    if n_points < 1:
        raise ValueError("n_points must be positive")
    i = np.arange(n_points)
    golden = (1 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n_points
    colat = np.arccos(np.clip(z, -1.0, 1.0))
    az = (2 * np.pi * i / golden) % (2 * np.pi)
    w = np.full(n_points, 4 * np.pi / n_points)
    return SphericalGrid(azimuth=az, angle2=colat, weights=w, convention="az_colat")


def get_tdesign_fallback(order: int, n_points: int | None = None) -> SphericalGrid:
    """Return a Fibonacci grid as a fallback when t-design data is unavailable.

    Parameters
    ----------
    order : int
        Spherical harmonic order the grid should support.
    n_points : int or None, optional
        Explicit number of grid points.  When *None* (default), the count
        is chosen as ``max(2 * (order + 1)**2, 32)``.

    Returns
    -------
    SphericalGrid
        Fibonacci-spiral grid suitable as a coarse substitute for a
        proper t-design.

    Examples
    --------
    >>> g = get_tdesign_fallback(3)
    >>> g.azimuth.shape[0] >= 32
    True
    """
    if n_points is None:
        n_points = max(2 * (order + 1) ** 2, 32)
    return fibonacci_grid(n_points)


def gauss_legendre_sampling(order: int) -> SphericalGrid:
    """Gauss-Legendre x equi-azimuth grid with exact quadrature for SH up to *order*.

    Uses Gauss-Legendre nodes in colatitude (exact polynomial integration)
    and uniformly spaced azimuth points (exact for trigonometric polynomials).

    Previously named ``equiangle_sampling``; the old name is kept as an alias.

    Parameters
    ----------
    order : int
        Maximum spherical harmonic order for which the quadrature is exact.
        Produces ``(order + 1) * 2 * (order + 1)`` grid points.

    Returns
    -------
    SphericalGrid
        Tensor-product grid with ``azimuth``, ``angle2`` (colatitude), and
        quadrature ``weights`` that integrate any spherical harmonic product
        of degree up to *order* exactly.

    Examples
    --------
    >>> g = gauss_legendre_sampling(4)
    >>> bool(g.azimuth.shape == (5 * 10,))
    True
    >>> bool(abs(g.weights.sum() - 4 * 3.141592653589793) < 1e-10)
    True
    """
    n_theta = order + 1
    n_phi = 2 * (order + 1)
    # Gauss-Legendre nodes on [-1, 1] → colatitude via arccos
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    colat = np.arccos(x_gl[::-1])  # ascending colatitude [0..π]
    w_gl = w_gl[::-1]              # match reordering

    d_phi = 2 * np.pi / n_phi
    az = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    aa, tt = np.meshgrid(az, colat, indexing="xy")

    # Combined weight: GL weight * Δφ  (the sin(θ) factor is already
    # absorbed into the GL quadrature via the substitution x = cos θ).
    w_2d = np.broadcast_to(w_gl[:, None] * d_phi, (n_theta, n_phi))
    return SphericalGrid(
        azimuth=aa.reshape(-1),
        angle2=tt.reshape(-1),
        weights=w_2d.reshape(-1),
        convention="az_colat",
    )


# Backward-compatible alias.
equiangle_sampling = gauss_legendre_sampling


def spatial_aliasing_frequency(
    array_radius_m: float,
    max_order: int,
    c: float = 343.0,
) -> float:
    """Estimate the spatial-aliasing frequency for a spherical array.

    Uses the usual ``kR <= N`` rule, giving
    ``f_alias = N c / (2 pi R)``.
    """
    if float(array_radius_m) <= 0:
        raise ValueError("array_radius_m must be positive")
    if int(max_order) < 0:
        raise ValueError("max_order must be non-negative")
    return float(int(max_order) * float(c) / (2.0 * math.pi * float(array_radius_m)))


def max_sh_order(
    array_radius_m: float,
    freq_hz_max: float,
    c: float = 343.0,
) -> int:
    """Estimate the largest usable SH order at a frequency limit."""
    if float(array_radius_m) <= 0:
        raise ValueError("array_radius_m must be positive")
    if float(freq_hz_max) <= 0:
        return 0
    kr_max = 2.0 * math.pi * float(freq_hz_max) * float(array_radius_m) / float(c)
    return max(0, int(math.floor(kr_max)))
