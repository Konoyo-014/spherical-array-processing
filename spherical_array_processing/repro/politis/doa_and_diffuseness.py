from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eig as generalized_eig

from ...beamforming.adaptive import lcmv_weights as _lcmv, mvdr_weights as _mvdr
from ...diffuseness.estimators import diffuseness_cmd, diffuseness_ie, diffuseness_sv, diffuseness_tv
from ...doa.spectra import music_spectrum
from ...coords import cart_to_sph, unit_sph_to_cart
from ...array.sampling import get_tdesign_fallback
from ...sh import matrix as sh_matrix
from ...types import SHBasisSpec, SphericalGrid


def _basis_from_cov(cov: np.ndarray) -> SHBasisSpec:
    n_coeffs = cov.shape[0]
    order = int(round(np.sqrt(n_coeffs) - 1))
    if (order + 1) ** 2 != n_coeffs:
        raise ValueError("covariance size is not a valid SH channel count")
    return SHBasisSpec(max_order=order, basis="real")


def sphPWDmap(sph_cov: ArrayLike, grid_dirs_rad: ArrayLike, n_src: int = 1):
    """Politis MATLAB-compatible PWD steered-response map."""
    cov = np.asarray(sph_cov, dtype=np.complex128)
    grid_dirs = np.asarray(grid_dirs_rad, dtype=float)
    n_sh = cov.shape[0]
    order = int(round(np.sqrt(n_sh) - 1))
    if (order + 1) ** 2 != n_sh:
        raise ValueError("sph_cov size is not a valid SH channel count")

    grid_dirs2 = np.column_stack([grid_dirs[:, 0], (np.pi / 2.0) - grid_dirs[:, 1]])  # [azi, incl]
    grid_colat = SphericalGrid(azimuth=grid_dirs2[:, 0], angle2=grid_dirs2[:, 1], convention="az_colat")
    y_grid = np.asarray(sh_matrix(SHBasisSpec(max_order=order, basis="real"), grid_colat))  # [G,C]

    scale = 4.0 * np.pi / float((order + 1) ** 2)
    p_pwd = np.zeros(grid_dirs.shape[0], dtype=float)
    for ng in range(grid_dirs.shape[0]):
        st = scale * y_grid[ng, :].reshape(-1, 1)  # [C,1]
        p_pwd[ng] = float(np.real((st.conj().T @ cov @ st).squeeze()))

    est_dirs = _von_mises_peaks(p_pwd, grid_dirs, nSrc=n_src)
    return p_pwd, est_dirs


def sphMUSIC(sph_cov: ArrayLike, grid_dirs_rad: ArrayLike, n_src: int = 1):
    cov = np.asarray(sph_cov, dtype=np.complex128)
    grid_dirs = np.asarray(grid_dirs_rad, dtype=float)
    grid = SphericalGrid(azimuth=grid_dirs[:, 0], angle2=grid_dirs[:, 1], convention="az_el")
    res = music_spectrum(cov, grid, _basis_from_cov(cov), n_sources=n_src, n_peaks=n_src)
    return res.spectrum, res.peak_dirs_rad


def sphMVDR(sph_cov: ArrayLike, beam_dirs_rad: ArrayLike) -> np.ndarray:
    cov = np.asarray(sph_cov, dtype=np.complex128)
    dirs = np.asarray(beam_dirs_rad, dtype=float)
    if dirs.ndim == 1:
        dirs = dirs[None, :]
    grid = SphericalGrid(azimuth=dirs[:, 0], angle2=dirs[:, 1], convention="az_el")
    y = np.asarray(sh_matrix(_basis_from_cov(cov), grid))  # [B,C]
    w = np.stack([_mvdr(cov, y_i) for y_i in y], axis=1)
    return w[:, 0] if w.shape[1] == 1 else w


def sphMVDRmap(sph_cov: ArrayLike, grid_dirs_rad: ArrayLike, n_src: int = 1):
    cov = np.asarray(sph_cov, dtype=np.complex128)
    dirs = np.asarray(grid_dirs_rad, dtype=float)
    grid = SphericalGrid(azimuth=dirs[:, 0], angle2=dirs[:, 1], convention="az_el")
    y = np.asarray(sh_matrix(_basis_from_cov(cov), grid))  # [G,C]
    p = np.zeros(grid.size, dtype=float)
    for i in range(grid.size):
        w = _mvdr(cov, y[i])
        p[i] = np.real(np.vdot(w, cov @ w))
    idx = np.argpartition(p, -n_src)[-n_src:]
    idx = idx[np.argsort(p[idx])[::-1]]
    est_dirs = np.stack([grid.azimuth[idx], grid.elevation[idx]], axis=1)
    return p, est_dirs


def sphLCMV(sph_cov: ArrayLike, constraint_dirs_rad: ArrayLike, constraints: ArrayLike) -> np.ndarray:
    cov = np.asarray(sph_cov, dtype=np.complex128)
    dirs = np.asarray(constraint_dirs_rad, dtype=float)
    grid = SphericalGrid(azimuth=dirs[:, 0], angle2=dirs[:, 1], convention="az_el")
    cmat = np.asarray(sh_matrix(_basis_from_cov(cov), grid)).T
    return _lcmv(cov, cmat, constraints)


def getDiffuseness_IE(pv_cov: ArrayLike) -> float:
    return diffuseness_ie(pv_cov)


def getDiffuseness_TV(i_vecs: ArrayLike) -> float:
    return diffuseness_tv(i_vecs)


def getDiffuseness_SV(i_vecs: ArrayLike) -> float:
    return diffuseness_sv(i_vecs)


def getDiffuseness_CMD(sh_cov: ArrayLike):
    return diffuseness_cmd(sh_cov)


def getDiffuseness_DPV(sh_cov: ArrayLike) -> float:
    cov = np.asarray(sh_cov, dtype=np.complex128)
    basis = _basis_from_cov(cov)
    order = basis.max_order
    grid = get_tdesign_fallback(order=4 * max(order, 1), n_points=max(64, 4 * (order + 1) ** 2))
    # Politis code uses az/elev grid; our fallback grid is az/colat
    y_grid = np.asarray(sh_matrix(SHBasisSpec(max_order=order, basis="real"), grid))  # [G, C]
    a_grid = ((4 * np.pi) / (order + 1) ** 2) * y_grid
    p_grid = np.real(np.einsum("gi,ij,gj->g", a_grid, cov, a_grid))
    p_mean = np.mean(p_grid)
    if p_mean <= 1e-12:
        return 1.0
    p_dev = (1.0 / p_mean) * np.sum(np.abs(p_grid - p_mean))
    y0 = np.asarray(sh_matrix(SHBasisSpec(max_order=order, basis="real"), SphericalGrid(np.array([0.0]), np.array([0.0]), convention="az_el")))[0]
    pw_df = (order + 1) ** 2
    pw_grid = np.abs(a_grid @ y0) ** 2
    pw_dev = pw_df * np.sum(np.abs(pw_grid - 1.0 / pw_df))
    if pw_dev <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - p_dev / pw_dev, 0.0, 1.0))


def sparse_solver_irls(
    p: float,
    A: ArrayLike,
    Y: ArrayLike,
    beta: float,
    termination_value: float,
    max_iterations: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IRLS sparse solver in the style of Politis `sparse_solver_irls`."""
    A = np.asarray(A, dtype=np.complex128)
    Y = np.asarray(Y, dtype=np.complex128)
    n_chan, n_dict = A.shape
    n_snap = Y.shape[1]
    W = np.eye(n_dict, dtype=np.complex128)
    epsilon = 1.0
    min_epsilon = termination_value
    n_iter = 0

    if n_snap > n_chan:
        U, s, _ = np.linalg.svd(Y, full_matrices=False)
        L_trunc = np.diag(s[:n_chan])
        Y2 = U[:, :n_chan] @ L_trunc
    else:
        Y2 = Y
        U = None
        L_trunc = None
    X_prev, *_ = np.linalg.lstsq(A, Y2, rcond=None)

    D = np.zeros((n_dict, n_chan), dtype=np.complex128)
    e = np.zeros((n_dict,), dtype=float)
    while epsilon > min_epsilon:
        gamma = (beta / max(1 - beta, 1e-12)) * np.trace(A @ W @ A.conj().T).real / n_chan
        D = (W @ A.conj().T) @ np.linalg.inv(A @ W @ A.conj().T + gamma * np.eye(n_chan))
        X_current = D @ Y2
        e = np.sum(np.abs(X_current) ** 2, axis=1)
        e_max = float(np.max(e)) if e.size else 0.0
        epsilon = e_max / max(n_dict, 1)
        w = (e + epsilon) ** (1 - p / 2)
        W = np.diag(w.astype(np.complex128))
        X_prev = X_current
        n_iter += 1
        if max_iterations and n_iter > max_iterations:
            break

    if n_snap > n_chan and U is not None and L_trunc is not None:
        D = X_current @ np.linalg.inv(U[:, :n_chan] @ L_trunc)
        X = D @ Y
    else:
        X = X_current
    return X, D, e


def sphSRmap(
    shsig: ArrayLike,
    p: float,
    A_grid: ArrayLike,
    regValue: float,
    stopValue: float,
    maxIter: int,
    grid_dirs_rad: ArrayLike,
    nSrc: int = 1,
):
    Y = np.asarray(shsig, dtype=np.complex128)
    A = np.asarray(A_grid, dtype=np.complex128)
    grid_dirs = np.asarray(grid_dirs_rad, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    _, _, P_sr = sparse_solver_irls(p, A, Y, regValue, stopValue, maxIter)
    est_dirs = _von_mises_peaks(P_sr, grid_dirs, nSrc=nSrc)
    return P_sr, est_dirs


def sphIntensityHist(i_xyz: ArrayLike, grid_dirs_rad: ArrayLike, nSrc: int = 1):
    i_xyz = np.asarray(i_xyz, dtype=float)
    grid_dirs = np.asarray(grid_dirs_rad, dtype=float)
    grid_xyz = unit_sph_to_cart(grid_dirs[:, 0], grid_dirs[:, 1], convention="az_el")
    # nearest grid point for each intensity direction
    d2 = np.sum((i_xyz[:, None, :] - grid_xyz[None, :, :]) ** 2, axis=-1)
    idx = np.argmin(d2, axis=1)
    i_mag = np.sum(i_xyz**2, axis=1)
    hist = np.bincount(idx, weights=i_mag, minlength=grid_xyz.shape[0]).astype(float)
    est_dirs = _von_mises_peaks(hist, grid_dirs, nSrc=nSrc)
    return hist, est_dirs


def sphiPMMW(Phi_x: ArrayLike, Phi_n: ArrayLike, src_dirs_rad: ArrayLike):
    """Pythonic implementation of Politis SHD iPMMW beamformer weights."""
    Phi_x = np.asarray(Phi_x, dtype=np.complex128)
    Phi_n = np.asarray(Phi_n, dtype=np.complex128)
    src_dirs = np.asarray(src_dirs_rad, dtype=float)
    n_sh = Phi_x.shape[0]
    order = int(round(np.sqrt(n_sh) - 1))
    n_src = src_dirs.shape[0]
    basis = SHBasisSpec(max_order=order, basis="real")
    grid = SphericalGrid(azimuth=src_dirs[:, 0], angle2=src_dirs[:, 1], convention="az_el")
    Y_src = np.asarray(sh_matrix(basis, grid))  # [K,nSH]
    A = Y_src.T  # [nSH,K]

    AA = A @ A.conj().T
    evals_aa, evecs_aa = np.linalg.eigh(AA)
    perm = np.argsort(evals_aa)[::-1]
    evecs_aa = evecs_aa[:, perm]
    U_AA = evecs_aa[:, n_src:] if n_src < n_sh else np.zeros((n_sh, 0), dtype=np.complex128)
    if U_AA.shape[1] == 0:
        Pd_est = 0.0
    else:
        Dm = U_AA.conj().T @ Phi_n @ U_AA
        Em = U_AA.conj().T @ U_AA
        vals, vecs = generalized_eig(Dm, -Em)
        idx = int(np.argmax(np.real(vals)))
        c_d = vecs[:, idx]
        w_d = U_AA @ c_d
        denom = np.vdot(w_d, w_d)
        Pd_est = float(np.real(np.vdot(w_d, Phi_x @ w_d) / denom)) if abs(denom) > 1e-12 else 0.0

    D_src = np.column_stack([(A[:, i : i + 1] @ A[:, i : i + 1].conj().T).reshape(-1) for i in range(n_src)])
    Phi_u = Phi_n + Pd_est * np.eye(n_sh) / (4 * np.pi)
    Phi_v = Phi_x - Phi_u
    Ps_est, *_ = np.linalg.lstsq(D_src, Phi_v.reshape(-1), rcond=None)
    Ps_est = np.real(Ps_est)
    Phi_s = np.diag(np.maximum(Ps_est, 1e-12))

    a_1, a_2, a_3 = 3.0, 0.5, -28.0
    sig = lambda ksi: a_1 / 2 * (1 + np.tanh(a_2 * (a_3 - ksi) / 2))
    phi_u = Pd_est + np.real(np.trace(Phi_n)) / n_sh
    ksi_l = 10 * np.log10(np.maximum(Ps_est, 1e-12) / max(phi_u, 1e-12))
    beta_l = sig(ksi_l)
    B = np.diag(beta_l)

    Phi_u_inv = np.linalg.pinv(Phi_u)
    mid = B @ np.linalg.pinv(Phi_s) + A.conj().T @ Phi_u_inv @ A
    W = Phi_u_inv @ A @ np.linalg.pinv(mid)
    return W, Pd_est, Ps_est


def sphESPRIT(Us: ArrayLike) -> np.ndarray:
    """Spherical ESPRIT DoA estimation (translated from Politis MATLAB implementation)."""
    Us = np.asarray(Us, dtype=np.complex128)
    LambdaXYp, LambdaXYm, LambdaZ = _esprit_getLambda(Us)
    PsiXYp, PsiXYm, PsiZ = _esprit_getPsi(Us, LambdaXYp, LambdaXYm, LambdaZ)
    _, V = generalized_eig(PsiXYp, PsiZ)
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


def _esprit_getYnimu(Ynm: np.ndarray, ni: int, mu: int) -> np.ndarray:
    N = int(round(np.sqrt(Ynm.shape[1]) - 1))
    idx_nimu, idx_nm = _esprit_muni2q(N, ni, mu)
    Ynimu = np.zeros((Ynm.shape[0], N**2), dtype=np.complex128)
    Ynimu[:, idx_nimu] = Ynm[:, idx_nm]
    return Ynimu


def _esprit_muni2q(order: int, ni: int, mu: int) -> tuple[np.ndarray, np.ndarray]:
    nm = []
    for n in range(order):
        nm.extend([(n, m) for m in range(-n, n + 1)])
    nm = np.asarray(nm, dtype=int)
    nimu = np.column_stack([nm[:, 0] + ni, nm[:, 1] + mu])
    qnm = nm[:, 0] ** 2 + nm[:, 0] + nm[:, 1]
    qnimu = nimu[:, 0] ** 2 + nimu[:, 0] + nimu[:, 1]
    valid = np.where(np.abs(nimu[:, 1]) <= nimu[:, 0])[0]
    idx_nm = qnimu[valid].astype(int)
    idx_nimu = qnm[valid].astype(int)
    return idx_nimu, idx_nm


def _esprit_getWnimu(order: int, mm: int, ni: int, mu: int) -> np.ndarray:
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


def _esprit_getVnimu(order: int, ni: int, mu: int) -> np.ndarray:
    nm = []
    for n in range(order):
        nm.extend([(n, m) for m in range(-n, n + 1)])
    nm = np.asarray(nm, dtype=int)
    nimu = np.column_stack([nm[:, 0] + ni, nm[:, 1] + mu])
    num = (nimu[:, 0] - nimu[:, 1]) * (nimu[:, 0] + nimu[:, 1])
    den = (2 * nimu[:, 0] - 1) * (2 * nimu[:, 0] + 1)
    v = np.sqrt(np.maximum(num / np.maximum(den, 1e-20), 0))
    return np.diag(v.astype(np.complex128))


def _esprit_getPsi(Us: np.ndarray, LambdaXYp: np.ndarray, LambdaXYm: np.ndarray, LambdaZ: np.ndarray):
    pinvUs = np.linalg.pinv(_esprit_getYnimu(Us.T, 0, 0).T)
    PsiXYp = pinvUs @ LambdaXYp
    PsiXYm = pinvUs @ LambdaXYm
    PsiZ = pinvUs @ LambdaZ
    return PsiXYp, PsiXYm, PsiZ


def _esprit_getLambda(Us: np.ndarray):
    order = int(round(np.sqrt(Us.shape[0]) - 1))
    LambdaXYp = (
        _esprit_getWnimu(order, 1, 1, -1) @ _esprit_getYnimu(Us.T, 1, -1).T
        - _esprit_getWnimu(order, -1, 0, 0) @ _esprit_getYnimu(Us.T, -1, -1).T
    )
    LambdaXYm = (
        -_esprit_getWnimu(order, -1, 1, -1) @ _esprit_getYnimu(Us.T, 1, 1).T
        + _esprit_getWnimu(order, 1, 0, 0) @ _esprit_getYnimu(Us.T, -1, 1).T
    )
    LambdaZ = (
        _esprit_getVnimu(order, 0, 0) @ _esprit_getYnimu(Us.T, -1, 0).T
        + _esprit_getVnimu(order, 1, 0) @ _esprit_getYnimu(Us.T, 1, 0).T
    )
    return LambdaXYp, LambdaXYm, LambdaZ


def _von_mises_peaks(power: np.ndarray, grid_dirs: np.ndarray, nSrc: int) -> np.ndarray:
    kappa = 20.0
    grid_xyz = unit_sph_to_cart(grid_dirs[:, 0], grid_dirs[:, 1], convention="az_el")
    p_minus = power.astype(float).copy()
    est = np.zeros((nSrc, 2), dtype=float)
    for k in range(nSrc):
        peak_idx = int(np.argmax(p_minus))
        est[k, :] = grid_dirs[peak_idx, :]
        vm_mean = grid_xyz[peak_idx, :]
        vm = kappa / (2 * np.pi * np.exp(kappa) - np.exp(-kappa)) * np.exp(kappa * (grid_xyz @ vm_mean))
        vm_mask = 1.0 / (1e-5 + vm)
        p_minus = p_minus * vm_mask
    return est
