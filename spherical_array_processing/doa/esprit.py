"""Spherical ESPRIT direction-of-arrival estimation.

Closed-form subspace-invariance DOA estimator for spherical microphone
arrays, based on the Jo & Choi (2019) three-recurrence formulation.

Unlike the grid-scanning :func:`~spherical_array_processing.doa.pwd_spectrum`
and :func:`~spherical_array_processing.doa.music_spectrum`, ESPRIT
recovers DOAs analytically from the signal-subspace eigenvectors of a
Hermitian SH covariance, so it is both faster (no scan grid, no peak
picking) and resolution-limited only by SNR rather than grid density —
at the cost of being harder to interpret when the assumed source count
is wrong.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eig as _generalized_eig

from ..types import SpatialSpectrumResult, SphericalGrid


Convention = Literal["az_el", "az_colat"]


# ---------------------------------------------------------------------------
# Core spherical-ESPRIT subroutines (Politis 2016 MATLAB translation).
# Kept private to this module — the public entry point is esprit_doa below.
# ---------------------------------------------------------------------------

def _muni2q(order: int, ni: int, mu: int) -> tuple[np.ndarray, np.ndarray]:
    nm = []
    for n in range(order):
        nm.extend([(n, m) for m in range(-n, n + 1)])
    nm = np.asarray(nm, dtype=int)
    nimu = np.column_stack([nm[:, 0] + ni, nm[:, 1] + mu])
    qnm = nm[:, 0] ** 2 + nm[:, 0] + nm[:, 1]
    qnimu = nimu[:, 0] ** 2 + nimu[:, 0] + nimu[:, 1]
    valid = np.where(np.abs(nimu[:, 1]) <= nimu[:, 0])[0]
    return qnm[valid].astype(int), qnimu[valid].astype(int)


def _getYnimu(Ynm: np.ndarray, ni: int, mu: int) -> np.ndarray:
    N = int(round(np.sqrt(Ynm.shape[1]) - 1))
    idx_nimu, idx_nm = _muni2q(N, ni, mu)
    Ynimu = np.zeros((Ynm.shape[0], N**2), dtype=np.complex128)
    Ynimu[:, idx_nimu] = Ynm[:, idx_nm]
    return Ynimu


def _getWnimu(order: int, mm: int, ni: int, mu: int) -> np.ndarray:
    nm = []
    for n in range(order):
        nm.extend([(n, m) for m in range(-n, n + 1)])
    nm = np.asarray(nm, dtype=int)
    if mm == 1:
        nimu = np.column_stack([nm[:, 0] + ni, nm[:, 1] + mu])
    else:
        nimu = np.column_stack([nm[:, 0] + ni, -nm[:, 1] + mu])
    num = (nimu[:, 0] - nimu[:, 1] - 1) * (nimu[:, 0] - nimu[:, 1])
    den = (2 * nimu[:, 0] - 1) * (2 * nimu[:, 0] + 1)
    w = np.sqrt(np.maximum(num / np.maximum(den, 1e-20), 0))
    return np.diag(w.astype(np.complex128))


def _getVnimu(order: int, ni: int, mu: int) -> np.ndarray:
    nm = []
    for n in range(order):
        nm.extend([(n, m) for m in range(-n, n + 1)])
    nm = np.asarray(nm, dtype=int)
    nimu = np.column_stack([nm[:, 0] + ni, nm[:, 1] + mu])
    num = (nimu[:, 0] - nimu[:, 1]) * (nimu[:, 0] + nimu[:, 1])
    den = (2 * nimu[:, 0] - 1) * (2 * nimu[:, 0] + 1)
    v = np.sqrt(np.maximum(num / np.maximum(den, 1e-20), 0))
    return np.diag(v.astype(np.complex128))


def _getLambda(Us: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = int(round(np.sqrt(Us.shape[0]) - 1))
    LambdaXYp = (
        _getWnimu(order, 1, 1, -1) @ _getYnimu(Us.T, 1, -1).T
        - _getWnimu(order, -1, 0, 0) @ _getYnimu(Us.T, -1, -1).T
    )
    LambdaXYm = (
        -_getWnimu(order, -1, 1, -1) @ _getYnimu(Us.T, 1, 1).T
        + _getWnimu(order, 1, 0, 0) @ _getYnimu(Us.T, -1, 1).T
    )
    LambdaZ = (
        _getVnimu(order, 0, 0) @ _getYnimu(Us.T, -1, 0).T
        + _getVnimu(order, 1, 0) @ _getYnimu(Us.T, 1, 0).T
    )
    return LambdaXYp, LambdaXYm, LambdaZ


def _getPsi(
    Us: np.ndarray,
    LambdaXYp: np.ndarray,
    LambdaXYm: np.ndarray,
    LambdaZ: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pinvUs = np.linalg.pinv(_getYnimu(Us.T, 0, 0).T)
    return pinvUs @ LambdaXYp, pinvUs @ LambdaXYm, pinvUs @ LambdaZ


def _sph_esprit_core(Us: np.ndarray) -> np.ndarray:
    """Direct port of Politis' `sphESPRIT`.  Returns an (n_src, 2) array of
    ``(azimuth, elevation)`` pairs in radians, elevation measured from
    the horizon."""
    Us = np.asarray(Us, dtype=np.complex128)
    LambdaXYp, LambdaXYm, LambdaZ = _getLambda(Us)
    PsiXYp, PsiXYm, PsiZ = _getPsi(Us, LambdaXYp, LambdaXYm, LambdaZ)
    _, V = _generalized_eig(PsiXYp, PsiZ)
    Vinv = np.linalg.pinv(V)
    PhiXYp = Vinv @ (PsiXYp @ V)
    PhiXYm = Vinv @ (PsiXYm @ V)
    PhiZ = Vinv @ (PsiZ @ V)
    phiX = np.real(np.diag(PhiXYp + PhiXYm) / 2)
    phiY = np.real(np.diag(PhiXYp - PhiXYm) / (2j))
    phiZ = np.real(np.diag(PhiZ))
    azim = np.arctan2(phiY, phiX)
    elev = np.arctan2(phiZ, np.sqrt(phiX**2 + phiY**2))
    return np.stack([azim, elev], axis=1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _signal_subspace(cov: NDArray[np.complex128], n_sources: int) -> NDArray[np.complex128]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals.real)[::-1]
    return eigvecs[:, order[:n_sources]]


def esprit_doa(
    cov: ArrayLike,
    n_sources: int,
    *,
    convention: Convention = "az_el",
) -> SpatialSpectrumResult:
    """Spherical-ESPRIT DOA estimation from an SH covariance matrix.

    Uses the Jo & Choi (2019) three-recurrence form of Eigenbeam-ESPRIT
    to extract source directions from the signal-subspace eigenvectors
    of a Hermitian SH covariance.  The input covariance is expected to
    follow the same convention as the one consumed by
    :func:`pwd_spectrum`, i.e. built from the physical SHT output
    ``R = E[c cᴴ]`` with ``c_{nm} ∝ Y_n^m*(k̂_src)``.

    .. important::
       The three-recurrence relations are defined for the **complex**
       ACN/orthonormal SH basis.  If your pipeline builds the
       covariance in the real (tesseral) basis, convert the
       coefficients first with
       :func:`spherical_array_processing.sh.real_to_complex_coeffs`
       before forming ``R``.  Feeding a real-basis covariance directly
       will produce biased DOAs (typically off by an azimuthal rotation).

    Parameters
    ----------
    cov : array_like, shape (Q, Q)
        Hermitian SH covariance matrix in ACN ordering; ``Q = (N+1)²``.
        ``N`` must be at least ``1``; practically ``N ≥ 3`` is
        recommended for robust estimates.
    n_sources : int
        Number of sources to estimate.  Must satisfy
        ``1 ≤ n_sources ≤ N²``.
    convention : {"az_el", "az_colat"}, optional
        Output direction convention.  ``"az_el"`` (default) reports
        elevation from the horizon; ``"az_colat"`` reports colatitude
        from the +z axis.

    Returns
    -------
    SpatialSpectrumResult
        Container with the estimated DOAs.  ``grid`` has size
        ``n_sources`` and carries one entry per source;
        ``peak_dirs_rad`` stores ``(azimuth, elevation)`` pairs in
        radians (elevation from horizon, independent of *convention*);
        ``peak_indices`` enumerates ``0 … n_sources − 1``; the synthetic
        ``spectrum`` is uniformly ``1`` at each reported source.

    References
    ----------
    .. [1] B. Jo and J.-W. Choi, "Spherical harmonic smoothing for
       localizing coherent sound sources", *J. Acoust. Soc. Am.*, 145(1),
       Jan 2019.
    .. [2] A. Politis, *Spherical Array Processing Toolkit*, 2016.
    """
    r = np.asarray(cov, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be a square matrix")
    q = r.shape[0]
    order = int(round(np.sqrt(q) - 1))
    if (order + 1) ** 2 != q:
        raise ValueError(
            f"cov size {q} is not (N+1)² for any integer N"
        )
    if order < 1:
        raise ValueError("ESPRIT requires SH max order N ≥ 1")
    if n_sources < 1 or n_sources > order ** 2:
        raise ValueError(
            f"n_sources must be in [1, N²] = [1, {order**2}], "
            f"got {n_sources}"
        )

    # Empirically verified against synthetic rank-1 covariances: for
    # the public SHT/PWD convention ``c_{nm} ∝ Y_n^m*(k̂_src)``, the
    # Politis recurrence recovers the correct ``k̂_src`` directly from
    # the leading eigenvectors of ``R = E[c cᴴ]``.  Feeding ``conj(R)``
    # instead yields the mirror image ``(θ_src, -φ_src)``.
    us = _signal_subspace(r, int(n_sources))
    dirs = _sph_esprit_core(us)  # (n_sources, 2)
    azimuth = dirs[:, 0] % (2.0 * np.pi)
    elevation = dirs[:, 1]

    if convention == "az_el":
        angle2 = elevation
    elif convention == "az_colat":
        angle2 = np.pi / 2.0 - elevation
    else:
        raise ValueError(
            f"convention must be 'az_el' or 'az_colat', got {convention!r}"
        )

    grid = SphericalGrid(
        azimuth=azimuth,
        angle2=angle2,
        weights=np.full(int(n_sources), 4.0 * np.pi / int(n_sources)),
        convention=convention,
    )
    spectrum = np.ones(int(n_sources), dtype=float)
    peak_indices = np.arange(int(n_sources), dtype=np.int64)
    peak_dirs_rad = np.stack([azimuth, elevation], axis=1)
    return SpatialSpectrumResult(
        spectrum=spectrum,
        grid=grid,
        peak_indices=peak_indices,
        peak_dirs_rad=peak_dirs_rad,
        metadata={"method": "esprit"},
    )


__all__ = ["esprit_doa"]
