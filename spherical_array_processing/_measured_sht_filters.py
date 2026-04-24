from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .sh import matrix as sh_matrix
from .types import SHBasisSpec, SphericalGrid


def _onesided_from_full_or_half(H_array: np.ndarray, nFFT: int) -> np.ndarray:
    n_bins = nFFT // 2 + 1
    if H_array.shape[0] == n_bins:
        return H_array
    if H_array.shape[0] == nFFT:
        return H_array[:n_bins, :, :]
    raise ValueError("H_array first axis must be nFFT or nFFT/2+1")


def arraySHTfiltersMeas_regLS(
    H_array: ArrayLike,
    order_sht: int,
    grid_dirs_rad: ArrayLike,
    w_grid: ArrayLike | None,
    nFFT: int,
    amp_threshold_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H_array, dtype=np.complex128)
    H = _onesided_from_full_or_half(H, nFFT)
    n_bins, n_mics, n_grid = H.shape
    order_sht = min(order_sht, int(np.floor(np.sqrt(n_mics) - 1)))
    w = np.ones(n_grid) if w_grid is None else np.asarray(w_grid, dtype=float).reshape(-1)
    if w.size != n_grid:
        raise ValueError("w_grid length mismatch")
    dirs = np.asarray(grid_dirs_rad, dtype=float)
    grid = SphericalGrid(dirs[:, 0], dirs[:, 1], convention="az_el")
    Y = np.asarray(sh_matrix(SHBasisSpec(max_order=order_sht, basis="real"), grid)).T * np.sqrt(4 * np.pi)
    Wg = np.diag(w)
    alpha = 10 ** (amp_threshold_db / 20.0)
    beta = 1 / (2 * alpha)
    n_sh = (order_sht + 1) ** 2
    H_filt = np.zeros((n_sh, n_mics, n_bins), dtype=np.complex128)
    for k in range(n_bins):
        tempH = H[k, :, :]
        gram = tempH @ Wg @ tempH.conj().T + beta**2 * np.eye(n_mics)
        H_filt[:, :, k] = Y @ Wg @ tempH.conj().T @ np.linalg.inv(gram)
    h_filt = np.fft.fftshift(np.fft.ifft(np.concatenate([H_filt, np.conj(H_filt[:, :, -2:0:-1])], axis=2), axis=2).real, axes=2)
    return H_filt, h_filt


def arraySHTfiltersMeas_regLSHD(
    H_array: ArrayLike,
    order_sht: int,
    grid_dirs_rad: ArrayLike,
    w_grid: ArrayLike | None,
    nFFT: int,
    amp_threshold_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H_array, dtype=np.complex128)
    H = _onesided_from_full_or_half(H, nFFT)
    n_bins, n_mics, n_grid = H.shape
    order_sht = min(order_sht, int(np.floor(np.sqrt(n_mics) - 1)))
    w = np.ones(n_grid) if w_grid is None else np.asarray(w_grid, dtype=float).reshape(-1)
    if w.size != n_grid:
        raise ValueError("w_grid length mismatch")
    order_array = max(0, int(np.floor(np.sqrt(n_grid) / 2 - 1)))
    dirs = np.asarray(grid_dirs_rad, dtype=float)
    grid = SphericalGrid(dirs[:, 0], dirs[:, 1], convention="az_el")
    Yg = np.asarray(sh_matrix(SHBasisSpec(max_order=order_array, basis="real"), grid)).T * np.sqrt(4 * np.pi)
    Wg = np.diag(w)
    alpha = 10 ** (amp_threshold_db / 20.0)
    beta = 1 / (2 * alpha)
    n_sh = (order_sht + 1) ** 2
    H_nm = np.zeros((n_bins, n_mics, (order_array + 1) ** 2), dtype=np.complex128)
    yg_gram_inv = np.linalg.pinv(Yg @ Wg @ Yg.conj().T)
    for k in range(n_bins):
        tempH = H[k, :, :]
        H_nm[k, :, :] = tempH @ Wg @ Yg.conj().T @ yg_gram_inv
    H_filt = np.zeros((n_sh, n_mics, n_bins), dtype=np.complex128)
    for k in range(n_bins):
        temp = H_nm[k, :, :]
        temp_trunc = temp[:, :n_sh]
        gram = temp @ temp.conj().T + beta**2 * np.eye(n_mics)
        H_filt[:, :, k] = temp_trunc.conj().T @ np.linalg.inv(gram)
    h_filt = np.fft.fftshift(np.fft.ifft(np.concatenate([H_filt, np.conj(H_filt[:, :, -2:0:-1])], axis=2), axis=2).real, axes=2)
    return H_filt, h_filt
