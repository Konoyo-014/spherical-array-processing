from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import eval_jacobi, lpmv

from ...acoustics.radial import bn_matrix as _bn_matrix
from ...acoustics.radial import plane_wave_radial_bn as _bn
from ...coords import cart_to_sph, sph_to_cart
from ...array.sampling import fibonacci_grid
from ...sh import matrix as sh_matrix
from ...types import SHBasisSpec, SphericalGrid


def sh2(order: int, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """Rafaely MATLAB-compatible complex SH matrix with rows=(N+1)^2, cols=L.

    This follows the original `sh2.m` construction exactly (including
    coefficient ordering and negative-order construction).
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    phi = np.asarray(phi, dtype=float).reshape(-1)
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have same shape")
    l = theta.size
    y = np.sqrt(1.0 / (4.0 * np.pi)) * np.ones((1, l), dtype=np.complex128)
    for n in range(1, order + 1):
        m_pos = np.arange(0, n + 1, dtype=int)
        a = np.sqrt(
            ((2 * n + 1) / (4 * np.pi))
            * np.array([math.factorial(n - m) / math.factorial(n + m) for m in m_pos], dtype=float)
        )
        p = np.vstack([lpmv(m, n, np.cos(theta)) for m in m_pos])
        y1 = a[:, None] * p * np.exp(1j * m_pos[:, None] * phi[None, :])
        m_neg = np.arange(-n, 0, dtype=int)
        y2 = ((-1.0) ** m_neg)[:, None] * np.conj(y1[:0:-1, :])
        y = np.vstack([y, y2, y1])
    return y


def bn(order: int, kr: ArrayLike, ka: ArrayLike, sphere: int | str) -> np.ndarray:
    return _bn(order, kr=kr, ka=ka, sphere=sphere)


def bn_mat(order: int, kr: ArrayLike, ka: ArrayLike, sphere: int | str) -> np.ndarray:
    return _bn_matrix(order, kr=kr, ka=ka, sphere=sphere, repeat_per_order=True)


def chebyshev_coefficients(order: int) -> np.ndarray:
    coeffs = np.polynomial.chebyshev.Chebyshev.basis(order).convert(kind=np.polynomial.Polynomial).coef
    return coeffs[::-1]


def legendre_coefficients(order: int) -> np.ndarray:
    out = np.zeros(order + 1, dtype=float)
    for r in range(order // 2 + 1):
        out[2 * r] = (
            (1 / 2**order)
            * (-1) ** r
            * math.factorial(2 * order - 2 * r)
            / (math.factorial(r) * math.factorial(order - r) * math.factorial(order - 2 * r))
        )
    return out


def wigner_d_matrix(order: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Rafaely-style Wigner-D block matrix up to order N in ACN ordering."""
    size = (order + 1) ** 2
    dmat = np.zeros((size, size), dtype=np.complex128)
    for n in range(order + 1):
        for m in range(-n, n + 1):
            for mm in range(-n, n + 1):
                mu = abs(mm - m)
                ni = abs(mm + m)
                s = int(n - (mu + ni) / 2)
                xi = ((-1) ** (m - mm)) if (m < mm) else 1.0
                c = math.sqrt(
                    math.factorial(s)
                    * math.factorial(s + mu + ni)
                    / (math.factorial(s + mu) * math.factorial(s + ni))
                )
                small_d = xi * c * (math.sin(beta / 2) ** mu) * (math.cos(beta / 2) ** ni) * eval_jacobi(
                    s, mu, ni, math.cos(beta)
                )
                val = np.exp(-1j * mm * alpha) * small_d * np.exp(-1j * m * gamma)
                dmat[n * n + n + mm, n * n + n + m] = val
    return dmat


def derivative_ph(vnm: ArrayLike) -> np.ndarray:
    v = np.asarray(vnm, dtype=np.complex128).reshape(-1)
    n = int(round(np.sqrt(v.size) - 1))
    if (n + 1) ** 2 != v.size:
        raise ValueError("vnm length must be a square SH count")
    out = np.zeros_like(v)
    for nn in range(n + 1):
        for m in range(-nn, nn + 1):
            q = nn * nn + nn + m
            out[q] = -1j * m * v[q]
    return out


def derivative_th(vnm: ArrayLike, th: float, ph: float) -> np.ndarray:
    v = np.asarray(vnm, dtype=np.complex128).reshape(-1)
    n = int(round(np.sqrt(v.size) - 1))
    if (n + 1) ** 2 != v.size:
        raise ValueError("vnm length must be a square SH count")
    out = np.zeros_like(v)
    cot_th = np.cos(th) / np.maximum(np.sin(th), 1e-12)
    for nn in range(n + 1):
        for m in range(-nn, nn + 1):
            q = nn * nn + nn + m
            g1 = m * cot_th
            val = g1 * v[q]
            if m < nn:
                g2 = np.sqrt((nn - m) * (nn + m + 1)) * np.exp(1j * ph)
                val = val + g2 * v[q + 1]
            out[q] = val
    return out


def c2s(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Rafaely names: theta=colatitude, phi=azimuth
    phi, theta, r = cart_to_sph(x, y, z, convention="az_colat")
    return theta, phi, r


def s2c(theta: ArrayLike, phi: ArrayLike, r: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sph_to_cart(phi, theta, r, convention="az_colat")


def equiangle_sampling(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rafaely-compatible equal-angle grid and weights.

    Returns `(a, th, ph)` where `th` is colatitude and `ph` is azimuth.
    """
    L = 2 * (order + 1)
    theta = np.arange(L) * np.pi / L + (np.pi / (2 * L))
    phi = np.arange(L) * 2 * np.pi / L
    th = np.reshape(np.tile(theta, (L, 1)), (-1,), order="F")
    ph = np.tile(phi, L)
    q = np.arange(order + 1)[:, None]
    S0 = np.sin((2 * q + 1) * th[None, :])
    S = (S0 / (2 * q + 1)).sum(axis=0)
    a = ((8 * np.pi) / (L**2)) * S * np.sin(th)
    return a, th, ph


def gaussian_sampling(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roots = np.roots(legendre_coefficients(order + 1))
    x = np.sort(np.real_if_close(roots))
    th = np.arccos(np.clip(x, -1.0, 1.0))
    th = np.sort(th)
    lc_next = legendre_coefficients(order + 2)
    a = (np.pi / (order + 1)) * 2 / (order + 2) ** 2 * (1 - np.cos(th) ** 2) / (np.polyval(lc_next, np.cos(th)) ** 2)
    ph = np.arange(0, 2 * np.pi, np.pi / (order + 1))
    th_full = np.repeat(th, 2 * order + 2)
    a_full = np.repeat(a, 2 * order + 2)
    ph_full = np.tile(ph, order + 1)
    return a_full, th_full, ph_full


def uniform_sampling(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fallback near-uniform sampling (not exact t-design tables for all N)
    n_points = {
        2: 12,
        3: 32,
        4: 36,
        6: 84,
        8: 144,
        10: 240,
    }.get(order, max(12, 2 * (order + 1) ** 2))
    g = fibonacci_grid(n_points)
    th = g.colatitude
    ph = g.azimuth
    a = np.full(n_points, 4 * np.pi / n_points)
    return a, th, ph


def platonic_solid(kind: int, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and faces for Platonic solids.

    `kind`: 1 tetrahedron, 2 cube, 3 octahedron, 4 icosahedron, 5 dodecahedron.
    Faces are 0-based indices for Python.
    """
    phi = (1 + np.sqrt(5.0)) / 2.0
    if kind == 1:
        v1 = np.array([-0.5, 0.5, 0.0, 0.0])
        v2 = np.array([-np.sqrt(3) / 6, -np.sqrt(3) / 6, np.sqrt(3) / 3, 0.0])
        v3 = np.array([-0.25 * np.sqrt(2 / 3), -0.25 * np.sqrt(2 / 3), -0.25 * np.sqrt(2 / 3), 0.75 * np.sqrt(2 / 3)])
        f = np.array([[1, 2, 3], [1, 2, 4], [2, 3, 4], [1, 3, 4]], dtype=int) - 1
    elif kind == 2:
        v1 = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
        v2 = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
        v3 = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)
        f = np.array([[1, 2, 3, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7], [4, 1, 5, 8], [5, 6, 7, 8]], dtype=int) - 1
    elif kind == 3:
        v1 = np.array([-1, 1, 1, -1, 0, 0], dtype=float)
        v2 = np.array([-1, -1, 1, 1, 0, 0], dtype=float)
        v3 = np.array([0, 0, 0, 0, -1, 1], dtype=float)
        f = np.array([[1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5], [1, 2, 6], [2, 3, 6], [3, 4, 6], [4, 1, 6]], dtype=int) - 1
    elif kind == 4:
        v1 = np.array([0, 0, 0, 0, -1, -1, 1, 1, -phi, phi, phi, -phi], dtype=float)
        v2 = np.array([-1, -1, 1, 1, -phi, phi, phi, -phi, 0, 0, 0, 0], dtype=float)
        v3 = np.array([-phi, phi, phi, -phi, 0, 0, 0, 0, -1, -1, 1, 1], dtype=float)
        f = np.array(
            [
                [1, 4, 9], [1, 5, 9], [1, 8, 5], [1, 8, 10], [1, 10, 4], [12, 2, 5], [12, 2, 3], [12, 3, 6],
                [12, 6, 9], [12, 9, 5], [11, 7, 10], [11, 10, 8], [11, 8, 2], [11, 2, 3], [11, 3, 7],
                [2, 5, 8], [10, 4, 7], [3, 6, 7], [6, 7, 4], [6, 4, 9],
            ],
            dtype=int,
        ) - 1
    elif kind == 5:
        v1 = np.array([1, (1 / phi), -phi, phi, -1, 0, -phi, 1, -1, -1, 1, (1 / phi), -1, 0, 0, -(1 / phi), phi, -(1 / phi), 1, 0], dtype=float)
        v2 = np.array([1, 0, -(1 / phi), (1 / phi), 1, -phi, (1 / phi), -1, 1, -1, -1, 0, -1, -phi, phi, 0, -(1 / phi), 0, 1, phi], dtype=float)
        v3 = np.array([1, phi, 0, 0, -1, -(1 / phi), 0, 1, 1, 1, -1, -phi, -1, (1 / phi), -(1 / phi), phi, 0, -phi, -1, (1 / phi)], dtype=float)
        f = np.array(
            [
                [1, 2, 16, 9, 20], [2, 16, 10, 14, 8], [16, 9, 7, 3, 10], [7, 9, 20, 15, 5],
                [5, 7, 3, 13, 18], [3, 13, 6, 14, 10], [6, 13, 18, 12, 11], [6, 11, 17, 8, 14],
                [11, 12, 19, 4, 17], [1, 2, 8, 17, 4], [1, 4, 19, 15, 20], [12, 18, 5, 15, 19],
            ],
            dtype=int,
        ) - 1
    else:
        raise ValueError("kind must be 1..5")
    _, _, rr = cart_to_sph(v1, v2, v3, convention="az_el")
    scale = np.where(rr == 0, 0.0, radius / rr)
    v = np.stack([v1 * scale, v2 * scale, v3 * scale], axis=1)
    return v, f
