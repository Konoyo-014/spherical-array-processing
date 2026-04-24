from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import loadmat
from scipy.special import eval_legendre
from scipy.special import spherical_jn

from ..._measured_sht_filters import (
    arraySHTfiltersMeas_regLS as _arraySHTfiltersMeas_regLS,
    arraySHTfiltersMeas_regLSHD as _arraySHTfiltersMeas_regLSHD,
)
from ...acoustics import sph_modal_coeffs
from ...array.sampling import fibonacci_grid, get_tdesign_fallback
from ...beamforming.fixed import beam_weights_cardioid
from ...beamforming.fixed import (
    axisymmetric_pattern,
    beam_weights_hypercardioid,
    beam_weights_maxev,
    beam_weights_supercardioid,
)
from ...coords import azel_to_az_colat, cart_to_sph, unit_sph_to_cart
from ...sh import complex_to_real_coeffs, direct_sht, matrix as sh_matrix, real_to_complex_coeffs
from ...sh import replicate_per_order
from ...types import SHBasisSpec, SphericalGrid


_ROOT = Path(__file__).resolve().parents[3]
_SHT_DESIGNS_MAT = _ROOT / "src" / "Spherical-Harmonic-Transform" / "t_designs_1_21.mat"


def sorted_eig(x: ArrayLike, direction: str = "descend") -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("input matrix should be square")
    vals, vecs = np.linalg.eig(arr)
    perm = np.argsort(vals)
    if direction == "descend":
        perm = perm[::-1]
    elif direction != "ascend":
        raise ValueError("direction must be 'ascend' or 'descend'")
    vals = vals[perm]
    vecs = vecs[:, perm]
    return vecs, np.diag(vals)


def beam_weights_cardioid_to_differential(order: int) -> np.ndarray:
    return np.array([(0.5**order) * math.comb(order, n) for n in range(order + 1)], dtype=float)


def _legendre_poly_coeffs_ascending(order: int) -> np.ndarray:
    # Coeffs in ascending powers of x, matching matrix construction in MATLAB helper.
    coeffs = np.zeros(order + 1, dtype=float)
    for k in range(order // 2 + 1):
        c = (
            (1 / 2**order)
            * (-1) ** k
            * math.factorial(2 * order - 2 * k)
            / (math.factorial(k) * math.factorial(order - k) * math.factorial(order - 2 * k))
        )
        power = order - 2 * k
        coeffs[power] = c
    return coeffs


def beam_weights_differential_to_spherical(a_n: ArrayLike) -> np.ndarray:
    a = np.asarray(a_n, dtype=float).reshape(-1)
    order = a.size - 1
    p_mat = np.zeros((order + 1, order + 1), dtype=float)
    for n in range(order + 1):
        c = _legendre_poly_coeffs_ascending(n)
        p_mat[: c.size, n] = c
    w = np.sqrt((2 * np.arange(order + 1) + 1) / (4 * np.pi))
    w_inv = np.diag(1.0 / w)
    b = w_inv @ np.linalg.solve(p_mat, a)
    return b


def sph_array_noise(radius_m: float, n_mics: int, max_order: int, array_type: str, freqs_hz: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    c = 343.0
    kR = 2 * np.pi * f * radius_m / c
    b_n = sph_modal_coeffs(max_order, kR, array_type=array_type).T / (4 * np.pi)  # [N+1,K]
    g2 = (1.0 / (n_mics * np.maximum(np.abs(b_n) ** 2, 1e-20))).T  # [K,N+1]
    if max_order < 1:
        return g2, np.zeros((f.size, 0))
    p = -(6 / 10) * np.arange(1, max_order + 1) / np.log10(2)
    b_lim0 = sph_modal_coeffs(max_order, np.array([1.0]), array_type=array_type)[0] / (4 * np.pi)
    a = 1.0 / (n_mics * np.maximum(np.abs(b_lim0[1:]) ** 2, 1e-20))
    g2_lin = np.zeros((f.size, max_order), dtype=float)
    for n in range(1, max_order + 1):
        g2_lin[:, n - 1] = a[n - 1] * np.maximum(kR, 1e-20) ** p[n - 1]
    return g2, g2_lin


def sph_array_noise_threshold(
    radius_m: float,
    n_mics: int,
    max_g_db: float,
    max_order: int,
    array_type: str,
) -> np.ndarray:
    c = 343.0
    out = np.zeros(max_order, dtype=float)
    max_g = 10 ** (max_g_db / 10.0)
    for n in range(1, max_order + 1):
        bn = sph_modal_coeffs(n, np.array([1.0]), array_type=array_type)[0, -1] / (4 * np.pi)
        kR_lim = (max_g * n_mics * (abs(bn) ** 2)) ** (-10 * np.log10(2) / (6 * n))
        out[n - 1] = kR_lim * c / (2 * np.pi * radius_m)
    return out


def check_condition_number_sht(max_order: int, mic_dirs_az_el_rad: ArrayLike, weights: ArrayLike | None = None) -> np.ndarray:
    dirs = np.asarray(mic_dirs_az_el_rad, dtype=float)
    if dirs.ndim != 2 or dirs.shape[1] != 2:
        raise ValueError("mic_dirs_az_el_rad must be [M,2] az/elev")
    az, colat = azel_to_az_colat(dirs[:, 0], dirs[:, 1])
    dirs_colat = np.column_stack([az, colat])
    # Keep parity with MATLAB checkCondNumberSHT: condition number of Y'Y (or Y'WY).
    return checkCondNumberSHT(max_order, dirs_colat, basisType="real", W=weights)


def sph_array_alias_lim(
    radius_m: float,
    n_mics: int,
    max_order: int,
    mic_dirs_az_el_rad: ArrayLike,
    mic_weights: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    c = 343.0
    f_alias = np.zeros(3, dtype=float)
    f_alias[0] = c * max_order / (2 * np.pi * radius_m)
    f_alias[1] = c * np.floor(np.sqrt(n_mics) - 1) / (2 * np.pi * radius_m)
    max_n = int(np.ceil(np.sqrt(n_mics) - 1))
    cond_n = check_condition_number_sht(max_n, mic_dirs_az_el_rad, mic_weights)
    good = np.where(cond_n < 1e4)[0]
    # cond_n index maps directly to order n (0..max_n), unlike MATLAB 1-based indexing.
    true_max_order = int(good[-1]) if good.size else 0
    f_alias[2] = c * true_max_order / (2 * np.pi * radius_m)
    return f_alias, cond_n


def default_eigenmike_like_dirs() -> np.ndarray:
    """Fallback nearly uniform 32-point grid used when exact Eigenmike geometry is not specified."""
    g = fibonacci_grid(32)
    elev = g.elevation
    return np.stack([g.azimuth, elev], axis=1)


def beam_weights_cardioid_to_spherical(order: int) -> np.ndarray:
    # Exact closed-form from Politis `beamWeightsCardioid2Spherical.m`.
    b_n = np.zeros(order + 1, dtype=float)
    for n in range(order + 1):
        b_n[n] = (
            np.sqrt(4 * np.pi * (2 * n + 1))
            * math.factorial(order)
            * math.factorial(order + 1)
            / (math.factorial(order + n + 1) * math.factorial(order - n))
            / (order + 1)
        )
    return b_n


def beamWeightsHypercardioid2Spherical(order: int) -> np.ndarray:
    # Match MATLAB beamWeightsHypercardioid2Spherical.m exactly:
    # c_n = 4*pi/(N+1)^2 * getSH(N,[0 0],'real'); then pick m=0 entries.
    dirs = np.array([[0.0, 0.0]], dtype=float)  # [azimuth, inclination]
    c = (4.0 * np.pi / float((order + 1) ** 2)) * getSH(order, dirs, "real").reshape(-1)
    b = np.zeros(order + 1, dtype=float)
    for n in range(order + 1):
        q = (n + 1) ** 2 - n - 1  # MATLAB 1-based ((n+1)^2 - n) -> Python 0-based
        b[n] = float(np.real(c[q]))
    return b


def beamWeightsSupercardioid2Spherical(order: int) -> np.ndarray:
    table = {
        1: np.array([1.2975, 1.2975], dtype=float),
        2: np.array([0.8372, 0.9591, 0.4680], dtype=float),
        3: np.array([0.6267, 0.7861, 0.5021, 0.1641], dtype=float),
        4: np.array([0.5013, 0.6674, 0.4942, 0.2314, 0.0569], dtype=float),
    }
    if order not in table:
        raise ValueError("Coefficients available up to 4")
    return table[order].copy()


def beamWeightsMaxEV(order: int) -> np.ndarray:
    d_n = np.zeros(order + 1, dtype=float)
    x = np.cos(2.4068 / (order + 1.51))
    for n in range(order + 1):
        d_n[n] = np.sqrt((2 * n + 1) / (4 * np.pi)) * eval_legendre(n, x)
    norm = np.sum(d_n * np.sqrt((2 * np.arange(order + 1) + 1) / (4 * np.pi)))
    return d_n / np.maximum(norm, 1e-12)


def _cheby_poly_coeffs_ascending(order: int) -> np.ndarray:
    # Chebyshev T_n coefficients in ascending powers of x
    coeffs_desc = np.polynomial.chebyshev.Chebyshev.basis(order).convert(kind=np.polynomial.Polynomial).coef[::-1]
    return coeffs_desc[::-1]


def returnLegePolyCoeffs(order: int) -> np.ndarray:
    return _legendre_poly_coeffs_ascending(order).reshape(-1, 1)


def returnChebyPolyCoeffs(order: int) -> np.ndarray:
    return _cheby_poly_coeffs_ascending(order).reshape(-1, 1)


def beamWeightsCardioid2Differential(N: int) -> np.ndarray:
    return beam_weights_cardioid_to_differential(N)


def beamWeightsCardioid2Spherical(N: int) -> np.ndarray:
    return beam_weights_cardioid_to_spherical(N)


def beamWeightsDifferential2Spherical(a_n: ArrayLike) -> np.ndarray:
    return beam_weights_differential_to_spherical(a_n)


def beamWeightsPressureVelocity(basisType: str = "real") -> np.ndarray:
    return beam_weights_pressure_velocity(basisType)


def differentialGains() -> dict[str, dict[int, np.ndarray]]:
    """Tabulated differential-form coefficients from Politis `differentialGains.m` script."""
    cardioid = {
        1: np.array([1 / 2, 1 / 2], dtype=float),
        2: np.array([1 / 4, 2 / 4, 1 / 4], dtype=float),
        3: np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8], dtype=float),
        4: np.array([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16], dtype=float),
    }
    supercardioid = {
        1: np.array([np.sqrt(3) - 1, 3 - np.sqrt(3)], dtype=float) / 2,
        2: np.array([1, 2 * np.sqrt(7), 5], dtype=float) / (2 * (3 + np.sqrt(7))),
        3: None,  # filled below
        4: np.array([0.0036, 0.0670, 0.2870, 0.4318, 0.2107], dtype=float),
    }
    b = np.sqrt(2)
    c = np.sqrt(21)
    d = np.sqrt(21 - c)
    supercardioid[3] = np.array(
        [b * d - c - 1, 21 + 9 * c - b * (6 + c) * d, 3 * (b * (4 + c) * d - 25 - 5 * c), 63 + 7 * c - b * (7 + 2 * c) * d],
        dtype=float,
    ) / 8
    hypercardioid = {
        1: np.array([1 / 4, 3 / 4], dtype=float),
        2: np.array([-1 / 6, 2 / 6, 5 / 6], dtype=float),
        3: np.array([-3 / 32, -15 / 32, 15 / 32, 35 / 32], dtype=float),
        4: np.array([0.075, -0.3, -1.05, 0.7, 1.575], dtype=float),
    }
    return {"cardioid": cardioid, "supercardioid": supercardioid, "hypercardioid": hypercardioid}


def beam_weights_pressure_velocity(basis_type: str = "real") -> np.ndarray:
    if basis_type == "real":
        return np.sqrt(4 * np.pi) * np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1 / np.sqrt(3)],
                [0, 1 / np.sqrt(3), 0, 0],
                [0, 0, 1 / np.sqrt(3), 0],
            ],
            dtype=np.complex128,
        )
    if basis_type == "complex":
        return np.sqrt(4 * np.pi) * np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(6), 0, -1 / np.sqrt(6)],
                [0, -1j / np.sqrt(6), 0, -1j / np.sqrt(6)],
                [0, 0, 1 / np.sqrt(3), 0],
            ],
            dtype=np.complex128,
        )
    raise ValueError("basis_type must be 'real' or 'complex'")


def beamWeightsDolphChebyshev2Spherical(N: int, paramType: str, arrayParam: float) -> np.ndarray:
    M = 2 * N
    if paramType == "sidelobe":
        R = 1.0 / arrayParam
        x0 = np.cosh((1.0 / M) * np.arccosh(R))
    elif paramType == "width":
        a0 = arrayParam / 2.0
        x0 = np.cos(np.pi / (2 * M)) / np.cos(a0 / 2.0)
        R = np.cosh(M * np.arccosh(x0))
    else:
        raise ValueError("paramType must be 'sidelobe' or 'width'")

    t_2N = _cheby_poly_coeffs_ascending(2 * N)
    P_N = np.zeros((N + 1, N + 1), dtype=float)
    for n in range(N + 1):
        c = _legendre_poly_coeffs_ascending(n)
        P_N[: c.size, n] = c

    d_n = np.zeros(N + 1, dtype=float)
    for n in range(N + 1):
        temp = 0.0
        for i in range(n + 1):
            for j in range(N + 1):
                for m in range(j + 1):
                    temp += (
                        (1 - (-1) ** (m + i + 1))
                        / (m + i + 1)
                        * math.factorial(j)
                        / (math.factorial(m) * math.factorial(j - m))
                        * (1 / (2**j))
                        * t_2N[2 * j]
                        * P_N[i, n]
                        * (x0 ** (2 * j))
                    )
        d_n[n] = (2 * np.pi / R) * temp
    norm = np.sum(d_n * np.sqrt(2 * np.arange(N + 1) + 1))
    return np.sqrt(4 * np.pi) * d_n / norm


def beamWeightsLinear2Spherical(a_n: ArrayLike, PLOT_ON: bool = False) -> np.ndarray:
    a = np.asarray(a_n, dtype=float).reshape(-1)
    N = a.size - 1
    W_lin = np.eye(N + 1)
    if N >= 1:
        W_lin[1:, 1:] *= 2
    T_N = np.zeros((N + 1, N + 1), dtype=float)
    P_N = np.zeros((N + 1, N + 1), dtype=float)
    for n in range(N + 1):
        tc = _cheby_poly_coeffs_ascending(n)
        pc = _legendre_poly_coeffs_ascending(n)
        T_N[: tc.size, n] = tc
        P_N[: pc.size, n] = pc
    w_sph = np.sqrt((2 * np.arange(N + 1) + 1) / (4 * np.pi))
    b = np.diag(1 / w_sph) @ np.linalg.solve(P_N, T_N @ W_lin @ a)
    if PLOT_ON:
        # plotting omitted here by default; helper kept for interface parity
        pass
    return b


def beamWeightsFromFunction(fHandleArray: Callable | Sequence[Callable], order: int) -> np.ndarray:
    grid = get_tdesign_fallback(order=20, n_points=512)
    # MATLAB grid is az/elev; fallback grid may be az/colat
    dirs_az = grid.azimuth
    dirs_el = grid.elevation
    funcs: list[Callable]
    if callable(fHandleArray):
        funcs = [fHandleArray]
    else:
        funcs = list(fHandleArray)
    y = np.asarray(sh_matrix(SHBasisSpec(max_order=order, basis="real"), SphericalGrid(dirs_az, dirs_el, weights=grid.weights, convention="az_el")))
    out = np.zeros(((order + 1) ** 2, len(funcs)), dtype=np.complex128)
    for i, f in enumerate(funcs):
        vals = np.asarray(f(dirs_az, dirs_el), dtype=float).reshape(-1)
        out[:, i] = direct_sht(vals, y, weights=grid.weights)
    return out[:, 0].real if len(funcs) == 1 else out.real


def extractAxisCoeffs(a_nm: ArrayLike) -> np.ndarray:
    a = np.asarray(a_nm)
    if a.ndim == 1:
        a = a[:, None]
    n_sh, n_sets = a.shape
    N = int(round(np.sqrt(n_sh) - 1))
    if (N + 1) ** 2 != n_sh:
        raise ValueError("a_nm first dimension is not a valid SH channel count")
    out = np.zeros((N + 1, n_sets), dtype=a.dtype)
    for n in range(N + 1):
        out[n, :] = a[n * n + n, :]
    return out[:, 0] if np.asarray(a_nm).ndim == 1 else out


def beamWeightsTorus2Spherical(N: int) -> np.ndarray:
    table = {
        1: np.array([2.7842, 0.0000, -0.7781, 0.0000, -0.1303]),
        2: np.array([2.3633, -0.0000, -1.0569]),
        3: np.array([2.0881, -0.0000, -1.1673, 0.0000, 0.1468]),
        4: np.array([1.8906, -0.0000, -1.2079, 0.0000, 0.2701]),
    }
    if N not in table:
        raise ValueError("Torus weights are tabulated only for N=1..4")
    return table[N].copy()


def plotAxisymPatternFromCoeffs(b_n: ArrayLike, ax=None):
    import matplotlib.pyplot as plt

    b = np.asarray(b_n, dtype=float).reshape(-1)
    theta = np.deg2rad(np.arange(0, 361))
    f = axisymmetric_pattern(theta, b)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
    pos = f >= 0
    ax.plot(theta[pos], np.abs(f[pos]), "b")
    ax.plot(theta[~pos], np.abs(f[~pos]), "r")
    return ax


def plotMicArray(mic_dirs_deg: ArrayLike, R: float):
    from ...plotting import plot_mic_array

    return plot_mic_array(mic_dirs_deg, R)


def plotDirectionalMapFromGrid(
    fgrid: ArrayLike,
    aziRes: float,
    polarRes: float,
    h_ax=None,
    POLAR_OR_ELEV: str = "elev",
    ZEROED_OR_CENTERED: str = "centered",
):
    from ...plotting import plot_directional_map_from_grid

    return plot_directional_map_from_grid(
        fgrid,
        azi_res_deg=aziRes,
        polar_res_deg=polarRes,
        ax=h_ax,
        polar_or_elev=POLAR_OR_ELEV,
        zeroed_or_centered=ZEROED_OR_CENTERED,
    )


def sphNullformer_pwd(order: int, beam_dirs_az_el_rad: ArrayLike) -> np.ndarray:
    dirs = np.asarray(beam_dirs_az_el_rad, dtype=float)
    if dirs.ndim == 1:
        dirs = dirs[None, :]
    grid = SphericalGrid(azimuth=dirs[:, 0], angle2=dirs[:, 1], convention="az_el")
    y = np.asarray(sh_matrix(SHBasisSpec(max_order=order, basis="real"), grid))  # [K,nSH]
    return np.linalg.pinv(y)  # [nSH,K]


def sphNullformer_diff(order: int, src_dirs_az_el_rad: ArrayLike) -> np.ndarray:
    dirs = np.asarray(src_dirs_az_el_rad, dtype=float)
    if dirs.ndim == 1:
        dirs = dirs[None, :]
    n_sh = (order + 1) ** 2
    n_src = dirs.shape[0]
    grid = SphericalGrid(azimuth=dirs[:, 0], angle2=dirs[:, 1], convention="az_el")
    y = np.asarray(sh_matrix(SHBasisSpec(max_order=order, basis="real"), grid))  # [K,nSH]
    c = np.zeros((n_src + 1,), dtype=float)
    c[0] = 1.0
    g = np.zeros((n_sh,), dtype=float)
    g[0] = 1.0
    a = np.concatenate([g[:, None], y.T], axis=1)
    # MATLAB: pinv(A') * c
    return (np.linalg.pinv(a.T) @ c).reshape(-1)


def getDiffCohMtxMeas(H_array: ArrayLike, w_grid: ArrayLike | None = None) -> np.ndarray:
    h = np.asarray(H_array, dtype=np.complex128)
    if h.ndim != 3:
        raise ValueError("H_array must be [n_bins, n_mics, n_grid]")
    n_bins, n_mics, n_grid = h.shape
    if w_grid is None:
        w = np.full(n_grid, 1.0 / n_grid, dtype=float)
    else:
        w = np.asarray(w_grid, dtype=float).reshape(-1)
        if w.size != n_grid:
            raise ValueError("w_grid length mismatch")
    out = np.zeros((n_mics, n_mics, n_bins), dtype=np.complex128)
    wdiag = np.diag(w)
    for nb in range(n_bins):
        h_nb = h[nb, :, :]  # [M,G]
        out[:, :, nb] = h_nb @ wdiag @ h_nb.conj().T
    return out


def getDiffCohMtxTheory(
    mic_dirs_rad: ArrayLike,
    array_type: str,
    radius_m: float,
    n_max: int,
    freqs_hz: ArrayLike,
    dir_coeff: float | None = None,
) -> np.ndarray:
    dirs = np.asarray(mic_dirs_rad, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    xyz = unit_sph_to_cart(dirs[:, 0], dirs[:, 1], convention="az_el")
    n_mics = xyz.shape[0]
    bn = sph_modal_coeffs(n_max, 2 * np.pi * freqs * radius_m / 343.0, array_type=array_type)
    bn2 = np.abs(bn / (4 * np.pi)) ** 2  # [F, N+1]
    out = np.zeros((n_mics, n_mics, freqs.size), dtype=np.complex128)
    for cidx in range(n_mics):
        for ridx in range(cidx, n_mics):
            cosang = float(np.clip(np.dot(xyz[ridx], xyz[cidx]), -1.0, 1.0))
            pn = np.array([eval_legendre(n, cosang) for n in range(n_max + 1)], dtype=float)
            weight = (2 * np.arange(n_max + 1) + 1) * pn
            vals = 4 * np.pi * (bn2 @ weight)
            out[ridx, cidx, :] = vals
            out[cidx, ridx, :] = vals
    return out


def diffCoherence(
    k: ArrayLike,
    r_A: ArrayLike,
    r_B: ArrayLike,
    a_nm: ArrayLike,
    b_nm: ArrayLike,
    G_mtx=None,  # retained for interface parity, unused in numeric implementation
) -> np.ndarray:
    """Numerical diffuse coherence integration for arbitrary directional patterns.

    This is a quadrature-based implementation replacing the original Gaunt/translation
    closed-form derivation. It preserves the MATLAB-style interface.
    """
    k = np.asarray(k, dtype=float).reshape(-1)
    rA = np.asarray(r_A, dtype=float).reshape(3)
    rB = np.asarray(r_B, dtype=float).reshape(3)
    a = np.asarray(a_nm, dtype=np.complex128).reshape(-1)
    b = np.asarray(b_nm, dtype=np.complex128).reshape(-1)
    Na = int(round(np.sqrt(a.size) - 1))
    Nb = int(round(np.sqrt(b.size) - 1))
    if (Na + 1) ** 2 != a.size or (Nb + 1) ** 2 != b.size:
        raise ValueError("a_nm/b_nm must be valid SH coefficient vectors")
    if a.size != b.size:
        # pad to common order
        N = max(Na, Nb)
        a_pad = np.zeros((N + 1) ** 2, dtype=np.complex128)
        b_pad = np.zeros((N + 1) ** 2, dtype=np.complex128)
        a_pad[: a.size] = a
        b_pad[: b.size] = b
        a, b = a_pad, b_pad
    N = int(round(np.sqrt(a.size) - 1))
    grid = get_tdesign_fallback(order=max(4 * N, 8), n_points=max(256, 4 * (N + 1) ** 2))
    Yc = np.asarray(sh_matrix(SHBasisSpec(max_order=N, basis="complex"), grid))  # [G,C]
    fa = Yc @ a
    fb = Yc @ b
    u = unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention)  # [G,3]
    dr = rB - rA
    phase_arg = u @ dr
    w = grid.weights if grid.weights is not None else np.full(grid.size, 4 * np.pi / grid.size)
    den = np.sqrt(np.vdot(a, a) * np.vdot(b, b))
    if abs(den) < 1e-15:
        return np.zeros_like(k, dtype=np.complex128)
    out = np.zeros_like(k, dtype=np.complex128)
    for i, kk in enumerate(k):
        phase = np.exp(-1j * kk * phase_arg)
        num = np.sum(w * fa * np.conj(fb) * phase)
        out[i] = num / den
    return out


def computeVelCoeffsMtx(sectorOrder: int) -> np.ndarray:
    """Numerical projection version of Politis `computeVelCoeffsMtx`.

    Returns complex SH conversion matrices A_xyz with shape [C_out, C_in, 3].
    """
    Ns = int(sectorOrder)
    Nxyz = Ns + 1
    Cin = (Ns + 1) ** 2
    Cout = (Nxyz + 1) ** 2
    grid = get_tdesign_fallback(order=max(4 * Nxyz, 10), n_points=max(512, 6 * Cout))
    Y_in = np.asarray(sh_matrix(SHBasisSpec(max_order=Ns, basis="complex"), grid))  # [G,Cin]
    Y_out = np.asarray(sh_matrix(SHBasisSpec(max_order=Nxyz, basis="complex"), grid))  # [G,Cout]
    x, y, z = unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention).T
    w = grid.weights if grid.weights is not None else np.full(grid.size, 4 * np.pi / grid.size)
    A_xyz = np.zeros((Cout, Cin, 3), dtype=np.complex128)
    for j in range(Cin):
        f0 = Y_in[:, j]
        for ax_idx, coord in enumerate((x, y, z)):
            prod = coord * f0
            # weighted projection to output SH basis
            A_xyz[:, j, ax_idx] = (Y_out.conj().T @ (w * prod))
    return A_xyz


def beamWeightsVelocityPatterns(b_n: ArrayLike, orientation: ArrayLike, A_xyz: ArrayLike | None = None, basisType: str = "real") -> np.ndarray:
    """Generate x/y/z weighted velocity patterns from an axisymmetric sector.

    If `A_xyz` is provided it is used directly; otherwise a numerical projection path
    is used (slower but self-contained).
    """
    b = np.asarray(b_n, dtype=float).reshape(-1)
    N = b.size - 1
    ori = np.asarray(orientation, dtype=float).reshape(2)
    # Build axisymmetric complex coefficients (m=0 only) and rotate numerically by evaluating on grid.
    c_axis = np.zeros((N + 1) ** 2, dtype=np.complex128)
    for n in range(N + 1):
        c_axis[n * n + n] = b[n]  # m=0 ACN index in complex basis

    if A_xyz is not None:
        A = np.asarray(A_xyz, dtype=np.complex128)
        x_nm = A[: (N + 2) ** 2, : (N + 1) ** 2, 0] @ c_axis
        y_nm = A[: (N + 2) ** 2, : (N + 1) ** 2, 1] @ c_axis
        z_nm = A[: (N + 2) ** 2, : (N + 1) ** 2, 2] @ c_axis
        vel = np.stack([x_nm, y_nm, z_nm], axis=1)
    else:
        grid = get_tdesign_fallback(order=max(4 * (N + 1), 10), n_points=max(512, 6 * (N + 2) ** 2))
        # rotate look direction by re-centering axisymmetric pattern relative to orientation
        u = unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention)
        u0 = unit_sph_to_cart(np.array([ori[0]]), np.array([ori[1]]), convention="az_el")[0]
        cosang = np.clip(u @ u0, -1.0, 1.0)
        theta_rel = np.arccos(cosang)
        f_sector = axisymmetric_pattern(theta_rel, b)
        x, y, z = u.T
        Y_out = np.asarray(sh_matrix(SHBasisSpec(max_order=N + 1, basis="complex"), grid))
        w = grid.weights if grid.weights is not None else np.full(grid.size, 4 * np.pi / grid.size)
        vel = np.zeros(((N + 2) ** 2, 3), dtype=np.complex128)
        for ax_idx, coord in enumerate((x, y, z)):
            vel[:, ax_idx] = Y_out.conj().T @ (w * (coord * f_sector))
    if basisType == "complex":
        return vel
    if basisType == "real":
        return complex_to_real_coeffs(vel.T, max_order=N + 1, axis=-1).T
    raise ValueError("basisType must be 'complex' or 'real'")


def getSH(N: int, dirs: ArrayLike, basisType: str = "real") -> np.ndarray:
    """Polarch-compatible SH matrix.

    `dirs` follows MATLAB convention [azimuth, inclination], where inclination is colatitude.
    Output shape is [n_dirs, (N+1)^2].
    """
    d = np.asarray(dirs, dtype=float)
    if d.ndim != 2:
        raise ValueError("dirs must be a 2D array of [azimuth, inclination]")
    if d.shape[1] != 2 and d.shape[0] == 2:
        d = d.T
    if d.shape[1] != 2:
        raise ValueError("dirs must have two columns: [azimuth, inclination]")
    basis = "complex" if basisType == "complex" else "real"
    grid = SphericalGrid(azimuth=d[:, 0], angle2=d[:, 1], convention="az_colat")
    return np.asarray(sh_matrix(SHBasisSpec(max_order=int(N), basis=basis), grid))


def unitSph2cart(aziElev: ArrayLike) -> np.ndarray:
    d = np.asarray(aziElev, dtype=float)
    if d.ndim != 2:
        raise ValueError("aziElev must be [n,2] or [2,n]")
    if d.shape[1] != 2 and d.shape[0] == 2:
        d = d.T
    if d.shape[1] != 2:
        raise ValueError("aziElev must have two columns")
    return unit_sph_to_cart(d[:, 0], d[:, 1], convention="az_el")


def sphModalCoeffs(N: int, kr: ArrayLike, arrayType: str, dirCoeff: float | None = None) -> np.ndarray:
    kr_arr = np.asarray(kr, dtype=float).reshape(-1)
    if arrayType in ("open", "rigid"):
        return sph_modal_coeffs(int(N), kr_arr, array_type=arrayType)
    if arrayType == "directional":
        coeff = 0.5 if dirCoeff is None else float(dirCoeff)
        out = np.zeros((kr_arr.size, int(N) + 1), dtype=np.complex128)
        for n in range(int(N) + 1):
            jn = spherical_jn(n, kr_arr)
            jnd = spherical_jn(n, kr_arr, derivative=True)
            out[:, n] = 4 * np.pi * (1j**n) * (coeff * jn - 1j * (1 - coeff) * jnd)
        out[np.isnan(out)] = 0.0
        return out
    raise ValueError("arrayType must be one of {'open', 'rigid', 'directional'}")


def getTdesign(degree: int) -> tuple[np.ndarray, np.ndarray]:
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be at least 1")
    if _SHT_DESIGNS_MAT.exists() and degree <= 21:
        data = loadmat(_SHT_DESIGNS_MAT)
        t_cell = data["t_designs"]
        vecs = np.asarray(t_cell[0, degree - 1], dtype=float)
        az, el, _ = cart_to_sph(vecs[:, 0], vecs[:, 1], vecs[:, 2], convention="az_el")
        dirs = np.stack([az, el], axis=1)
        return vecs, dirs
    n_points = max(2 * (degree // 2 + 1) ** 2, 32)
    g = fibonacci_grid(n_points)
    vecs = unit_sph_to_cart(g.azimuth, g.elevation, convention="az_el")
    dirs = np.stack([g.azimuth, g.elevation], axis=1)
    return vecs, dirs


def checkCondNumberSHT(N: int, dirs: ArrayLike, basisType: str = "real", W: ArrayLike | None = None) -> np.ndarray:
    y = getSH(N, dirs, basisType=basisType)
    if W is not None:
        w = np.asarray(W, dtype=float).reshape(-1)
        if w.size != y.shape[0]:
            raise ValueError("weights length mismatch")
    out = np.zeros(int(N) + 1, dtype=float)
    for n in range(int(N) + 1):
        yn = y[:, : (n + 1) ** 2]
        if W is None:
            yy = yn.conj().T @ yn
        else:
            yy = yn.conj().T @ np.diag(w) @ yn
        out[n] = float(np.linalg.cond(yy))
    return out


def conjCoeffs(f_nm: ArrayLike) -> np.ndarray:
    f = np.asarray(f_nm, dtype=np.complex128).reshape(-1)
    N = int(round(np.sqrt(f.size) - 1))
    if (N + 1) ** 2 != f.size:
        raise ValueError("f_nm length must be (N+1)^2")
    g = np.zeros_like(f)
    for n in range(N + 1):
        for m in range(-n, n + 1):
            qg = n * (n + 1) + m
            qf = n * (n + 1) - m
            g[qg] = ((-1) ** m) * np.conj(f[qf])
    return g


def rotateAxisCoeffs(c_n: ArrayLike, theta_0: float, phi_0: float, basisType: str = "real") -> np.ndarray:
    c = np.asarray(c_n, dtype=np.complex128).reshape(-1)
    N = c.size - 1
    y = getSH(N, np.array([[phi_0, theta_0]], dtype=float), basisType="complex")[0]
    out = np.zeros((N + 1) ** 2, dtype=np.complex128)
    for n in range(N + 1):
        for m in range(-n, n + 1):
            q = n * (n + 1) + m
            out[q] = np.sqrt(4 * np.pi / (2 * n + 1)) * c[n] * np.conj(y[q])
    if basisType == "complex":
        return out
    if basisType == "real":
        return np.asarray(complex_to_real_coeffs(out, max_order=N, axis=-1))
    raise ValueError("basisType must be 'complex' or 'real'")


def gaunt_mtx(N1: int, N2: int, N: int) -> np.ndarray:
    N1 = int(N1)
    N2 = int(N2)
    N = int(N)
    deg = min(21, max(1, 2 * (N1 + N2 + N)))
    _vecs, dirs = getTdesign(deg)
    w = np.full(dirs.shape[0], 4 * np.pi / dirs.shape[0], dtype=float)
    Y1 = getSH(N1, np.column_stack([dirs[:, 0], (np.pi / 2) - dirs[:, 1]]), basisType="complex")
    Y2 = getSH(N2, np.column_stack([dirs[:, 0], (np.pi / 2) - dirs[:, 1]]), basisType="complex")
    Y = getSH(N, np.column_stack([dirs[:, 0], (np.pi / 2) - dirs[:, 1]]), basisType="complex")
    A = np.zeros(((N1 + 1) ** 2, (N2 + 1) ** 2, (N + 1) ** 2), dtype=np.complex128)
    for q in range((N + 1) ** 2):
        A[:, :, q] = (Y1.conj().T * w[None, :]) @ (Y2 * np.conj(Y[:, [q]]))
    return A


def chebyshevPoly(n: int) -> np.ndarray:
    return _cheby_poly_coeffs_ascending(int(n))


def sphArrayNoise(R: float, Nmic: int, maxN: int, arrayType: str, f: ArrayLike):
    return sph_array_noise(R, Nmic, maxN, arrayType, f)


def sphArrayNoiseThreshold(R: float, Nmic: int, maxG_db: float, maxN: int, arrayType: str, dirCoeff=None):
    return sph_array_noise_threshold(R, Nmic, maxG_db, maxN, arrayType)


def sphArrayAliasLim(R: float, Nmic: int, maxN: int, mic_dirs_rad: ArrayLike, mic_weights: ArrayLike | None = None):
    return sph_array_alias_lim(R, Nmic, maxN, mic_dirs_rad, mic_weights)


def _ifft_half_spectrum_to_fir(h_half: np.ndarray) -> np.ndarray:
    """Mirror one-sided spectrum [K, ...] to real FIR [L, ...] along axis 0."""
    h = np.asarray(h_half, dtype=np.complex128)
    h2 = h.copy()
    h2[-1, ...] = np.real(h2[-1, ...])
    full = np.concatenate([h2, np.conj(h2[-2:0:-1, ...])], axis=0)
    fir = np.fft.ifft(full, axis=0).real
    return np.fft.fftshift(fir, axes=0)


def _fix_modal_dc_limits(bN: np.ndarray) -> np.ndarray:
    """Apply analytical kR->0 limits to modal responses.

    For rigid/open arrays in the normalized convention used here (bN/(4*pi)),
    n=0 tends to 1 and n>0 tends to 0 at DC.
    """
    out = np.asarray(bN, dtype=np.complex128).copy()
    if out.shape[0] > 0 and out.shape[1] > 0:
        out[0, 0] = 1.0 + 0.0j
        if out.shape[1] > 1:
            out[0, 1:] = 0.0 + 0.0j
    return out


def arraySHTfiltersTheory_radInverse(
    radius_m: float, n_mics: int, order_sht: int, fft_len: int, fs: float, amp_threshold_db: float
) -> tuple[np.ndarray, np.ndarray]:
    c = 343.0
    freqs = np.arange(fft_len // 2 + 1, dtype=float) * fs / fft_len
    order_sht = min(order_sht, int(np.floor(np.sqrt(n_mics) - 1)))
    kR = 2 * np.pi * freqs * radius_m / c
    bN = sph_modal_coeffs(order_sht, kR, array_type="rigid") / (4 * np.pi)  # [F,N+1]
    bN = np.nan_to_num(bN, nan=0.0, posinf=0.0, neginf=0.0)
    bN = _fix_modal_dc_limits(bN)
    alpha = np.sqrt(n_mics) * 10 ** (amp_threshold_db / 20.0)
    beta = np.sqrt((1 - np.sqrt(1 - 1 / alpha**2)) / (1 + np.sqrt(1 - 1 / alpha**2)))
    h_f = np.conj(bN) / (np.abs(bN) ** 2 + beta**2)
    h_t = _ifft_half_spectrum_to_fir(h_f)
    return h_f, h_t


def arraySHTfiltersTheory_softLim(
    radius_m: float, n_mics: int, order_sht: int, fft_len: int, fs: float, amp_threshold_db: float
) -> tuple[np.ndarray, np.ndarray]:
    c = 343.0
    freqs = np.arange(fft_len // 2 + 1, dtype=float) * fs / fft_len
    order_sht = min(order_sht, int(np.floor(np.sqrt(n_mics) - 1)))
    kR = 2 * np.pi * freqs * radius_m / c
    bN = sph_modal_coeffs(order_sht, kR, array_type="rigid") / (4 * np.pi)
    bN = np.nan_to_num(bN, nan=0.0, posinf=0.0, neginf=0.0)
    bN = _fix_modal_dc_limits(bN)
    inv_bN = np.zeros_like(bN)
    mask = np.abs(bN) > 1e-15
    inv_bN[mask] = 1.0 / bN[mask]
    alpha = np.sqrt(n_mics) * 10 ** (amp_threshold_db / 20.0)
    h_f = (2 * alpha / np.pi) * (np.abs(bN) * inv_bN) * np.arctan((np.pi / (2 * alpha)) * np.abs(inv_bN))
    h_t = _ifft_half_spectrum_to_fir(h_f)
    return h_f, h_t


def arraySHTfiltersTheory_regLS(
    radius_m: float,
    mic_dirs_az_el_rad: ArrayLike,
    order_sht: int,
    fft_len: int,
    fs: float,
    amp_threshold_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate Politis regLS encoding filter design.

    This implementation preserves the main interface and regularized LS structure,
    while using the local SH/radial model in place of the original external MATLAB dependencies.
    """
    dirs = np.asarray(mic_dirs_az_el_rad, dtype=float)
    n_mics = dirs.shape[0]
    freqs = np.arange(fft_len // 2 + 1, dtype=float) * fs / fft_len
    kR = 2 * np.pi * freqs * radius_m / 343.0
    order_array = min(30, int(np.floor(2 * kR[-1])))
    order_sht = min(order_sht, int(np.floor(np.sqrt(n_mics) - 1)))
    mic_dirs_azi_incl = np.column_stack([dirs[:, 0], (np.pi / 2.0) - dirs[:, 1]])
    y_array = np.sqrt(4.0 * np.pi) * getSH(order_array, mic_dirs_azi_incl, "real")  # [M,Ca]
    bN = sph_modal_coeffs(order_array, kR, array_type="rigid") / (4 * np.pi)  # [F,Na+1]
    bN = np.nan_to_num(bN, nan=0.0, posinf=0.0, neginf=0.0)
    bN = _fix_modal_dc_limits(bN)
    n_sh = (order_sht + 1) ** 2
    h_f = np.zeros((n_sh, n_mics, freqs.size), dtype=np.complex128)
    alpha = 10 ** (amp_threshold_db / 20.0)
    beta = 1.0 / (2 * alpha)
    eye_m = np.eye(n_mics, dtype=np.complex128)
    for k in range(freqs.size):
        # replicate per order but preserve complex dtype
        b_rep = np.concatenate([np.full(2 * n + 1, bN[k, n], dtype=np.complex128) for n in range(order_array + 1)])
        h_array = y_array * b_rep[None, :]  # [M,Ca]
        h_trunc = h_array[:, :n_sh]
        gram = h_array @ h_array.conj().T + (beta**2) * eye_m
        h_f[:, :, k] = h_trunc.conj().T @ np.linalg.inv(gram)
    h_half = h_f.copy()
    h_half[:, :, -1] = np.abs(h_half[:, :, -1])
    h_t = np.fft.fftshift(np.fft.ifft(np.concatenate([h_half, np.conj(h_half[:, :, -2:0:-1])], axis=2), axis=2).real, axes=2)
    return h_f, h_t


def arraySHTfiltersMeas_regLS(
    H_array: ArrayLike,
    order_sht: int,
    grid_dirs_rad: ArrayLike,
    w_grid: ArrayLike | None,
    nFFT: int,
    amp_threshold_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _arraySHTfiltersMeas_regLS(H_array, order_sht, grid_dirs_rad, w_grid, nFFT, amp_threshold_db)


def arraySHTfiltersMeas_regLSHD(
    H_array: ArrayLike,
    order_sht: int,
    grid_dirs_rad: ArrayLike,
    w_grid: ArrayLike | None,
    nFFT: int,
    amp_threshold_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _arraySHTfiltersMeas_regLSHD(H_array, order_sht, grid_dirs_rad, w_grid, nFFT, amp_threshold_db)


def evaluateSHTfilters(
    M_mic2sh: ArrayLike,
    H_array: ArrayLike,
    fs: float,
    Y_grid: ArrayLike,
    w_grid: ArrayLike | None = None,
    plot: bool = False,
):
    M = np.asarray(M_mic2sh, dtype=np.complex128)  # [nSH, nMics, nBins]
    H = np.asarray(H_array, dtype=np.complex128)  # [nBins, nMics, nGrid]
    Yg = np.asarray(Y_grid, dtype=np.complex128)  # [nGrid, nSH]
    n_grid = H.shape[2]
    if w_grid is None:
        w = np.full(n_grid, 1.0 / n_grid, dtype=float)
    else:
        w = np.asarray(w_grid, dtype=float).reshape(-1)
    W = w[:, None]
    n_bins = M.shape[2]
    n_fft = 2 * (n_bins - 1)
    _f = np.arange(n_bins) * fs / max(n_fft, 1)
    order_sht = int(round(np.sqrt(Yg.shape[1]) - 1))
    cSH = np.zeros((n_bins, order_sht + 1), dtype=np.complex128)
    lSH = np.zeros((n_bins, order_sht + 1), dtype=float)
    WNG = np.zeros((n_bins, 1), dtype=float)
    for k in range(n_bins):
        Hk = H[k, :, :]  # [M,G]
        y_recon = M[:, :, k] @ Hk  # [nSH,G]
        for n in range(order_sht + 1):
            csum = 0.0 + 0.0j
            lsum = 0.0
            for m in range(-n, n + 1):
                q = n * n + n + m
                y_recon_nm = y_recon[q, :][:, None]
                y_ideal_nm = Yg[:, q][:, None]
                num = np.vdot((y_recon_nm * W).ravel(), y_ideal_nm.ravel())
                den = np.sqrt(np.sum((y_recon_nm * W).ravel() * np.conj(y_recon_nm.ravel())))
                csum += 0.0 if abs(den) < 1e-12 else num / den
                lsum += float(np.real(np.sum((y_recon_nm * W).ravel() * np.conj(y_recon_nm.ravel()))))
            cSH[k, n] = csum / (2 * n + 1)
            lSH[k, n] = lsum / (2 * n + 1)
        eigM = np.linalg.eigvals(M[:, :, k].conj().T @ M[:, :, k])
        WNG[k, 0] = float(np.max(np.real(eigM)))
    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].semilogx(_f, np.abs(cSH))
        axs[1].semilogx(_f, 10 * np.log10(np.maximum(lSH, 1e-12)))
        axs[2].semilogx(_f, 10 * np.log10(np.maximum(WNG[:, 0], 1e-12)))
        axs[0].set_title("Spatial correlation")
        axs[1].set_title("Level difference")
        axs[2].set_title("Maximum amplification")
        axs[2].set_xlabel("Frequency (Hz)")
        for ax in axs:
            ax.grid(True)
    return cSH, lSH, WNG


def arraySHTfilters_diffEQ(M_mic2sh: ArrayLike, M_dfc: ArrayLike, f_alias: ArrayLike, fs: float) -> np.ndarray:
    M = np.asarray(M_mic2sh, dtype=np.complex128)  # [nSH, nMics, nBins]
    D = np.asarray(M_dfc, dtype=np.complex128)  # [nMics, nMics, nBins]
    f_alias = np.asarray(f_alias, dtype=float).reshape(-1)
    n_bins = M.shape[2]
    f = np.arange(n_bins, dtype=float) * fs / (2 * max(n_bins - 1, 1))
    idx_alias = int(np.argmin(np.abs(f - f_alias[0])))
    out = M.copy()
    if idx_alias >= n_bins - 1:
        return out
    E_fal = M[:, :, idx_alias]
    L_diff_fal = np.real(np.diag(E_fal @ D[:, :, idx_alias] @ E_fal.conj().T / (4 * np.pi)))
    for nf in range(idx_alias + 1, n_bins):
        E = M[:, :, nf]
        L_diff_f = np.real(np.diag(E @ D[:, :, nf] @ E.conj().T / (4 * np.pi)))
        gain = np.sqrt(np.maximum(L_diff_fal, 1e-20) / np.maximum(L_diff_f, 1e-20))
        out[:, :, nf] = np.diag(gain) @ E
    return out
