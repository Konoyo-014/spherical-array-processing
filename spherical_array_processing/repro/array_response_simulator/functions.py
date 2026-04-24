from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import jv, yv, hankel1, hankel2

from ...acoustics.radial import (
    besselhs,
    besselhsd,
    besseljs,
    besseljsd,
    sph_modal_coeffs,
)
from ...coords import unit_sph_to_cart


def sph_besselj(n: int, x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.sqrt(np.pi / (2 * np.where(x == 0, 1.0, x))) * jv(n + 0.5, x)
    out = out.astype(np.complex128)
    if n == 0:
        out[x == 0] = 1.0
    else:
        out[x == 0] = 0.0
    return out


def sph_bessely(n: int, x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.pi / (2 * x)) * yv(n + 0.5, x)


def sph_hankel1(n: int, x: ArrayLike) -> np.ndarray:
    return sph_besselj(n, x) + 1j * sph_bessely(n, x)


def sph_hankel2(n: int, x: ArrayLike) -> np.ndarray:
    return sph_besselj(n, x) - 1j * sph_bessely(n, x)


def dsph_besselj(n: int, x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (n * sph_besselj(n - 1, x) - (n + 1) * sph_besselj(n + 1, x)) / (2 * n + 1)


def dsph_bessely(n: int, x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (n * sph_bessely(n - 1, x) - (n + 1) * sph_bessely(n + 1, x)) / (2 * n + 1)


def dsph_hankel1(n: int, x: ArrayLike) -> np.ndarray:
    return dsph_besselj(n, x) + 1j * dsph_bessely(n, x)


def dsph_hankel2(n: int, x: ArrayLike) -> np.ndarray:
    return dsph_besselj(n, x) - 1j * dsph_bessely(n, x)


def dbesselj(n: int, x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if n == 0:
        return -jv(1, x)
    return 0.5 * (jv(n - 1, x) - jv(n + 1, x))


def dbessely(n: int, x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if n == 0:
        return -yv(1, x)
    return 0.5 * (yv(n - 1, x) - yv(n + 1, x))


def dhankel1(n: int, x: ArrayLike) -> np.ndarray:
    return dbesselj(n, x) + 1j * dbessely(n, x)


def dhankel2(n: int, x: ArrayLike) -> np.ndarray:
    return dbesselj(n, x) - 1j * dbessely(n, x)


def sph_function(n: int, x: ArrayLike, funcName: str) -> np.ndarray:
    fn = str(funcName)
    if fn == "besselj":
        return sph_besselj(n, x)
    if fn == "bessely":
        return sph_bessely(n, x)
    if fn == "hankel1":
        return sph_hankel1(n, x)
    if fn == "hankel2":
        return sph_hankel2(n, x)
    raise ValueError("funcName must be one of besselj/bessely/hankel1/hankel2")


def dsph_function(n: int, x: ArrayLike, funcName: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    fn = str(funcName)
    if fn == "besselj":
        return (n / np.where(x == 0, 1.0, x)) * sph_besselj(n, x) - sph_besselj(n + 1, x)
    if fn == "bessely":
        return (n / x) * sph_bessely(n, x) - sph_bessely(n + 1, x)
    if fn == "hankel1":
        return (n / x) * sph_hankel1(n, x) - sph_hankel1(n + 1, x)
    if fn == "hankel2":
        return (n / x) * sph_hankel2(n, x) - sph_hankel2(n + 1, x)
    raise ValueError("funcName must be one of besselj/bessely/hankel1/hankel2")


def sphModalCoeffs(N: int, kr: ArrayLike, arrayType: str, dirCoeff: float | None = None) -> np.ndarray:
    kr_arr = np.asarray(kr, dtype=float).reshape(-1)
    if arrayType in ("open", "rigid"):
        return sph_modal_coeffs(int(N), kr_arr, array_type=arrayType)
    if arrayType == "directional":
        coeff = 0.5 if dirCoeff is None else float(dirCoeff)
        out = np.zeros((kr_arr.size, int(N) + 1), dtype=np.complex128)
        for n in range(int(N) + 1):
            jn = sph_besselj(n, kr_arr)
            jnd = dsph_besselj(n, kr_arr)
            out[:, n] = 4 * np.pi * (1j**n) * (coeff * jn - 1j * (1 - coeff) * jnd)
        out[np.isnan(out)] = 0.0
        return out
    raise ValueError("Wrong array type")


def cylModalCoeffs(N: int, kr: ArrayLike, arrayType: str) -> np.ndarray:
    kr_arr = np.asarray(kr, dtype=float).reshape(-1)
    out = np.zeros((kr_arr.size, int(N) + 1), dtype=np.complex128)
    for n in range(int(N) + 1):
        if arrayType == "open":
            out[:, n] = (1j**n) * jv(n, kr_arr)
        elif arrayType == "rigid":
            jn = jv(n, kr_arr)
            jnd = dbesselj(n, kr_arr)
            hn = hankel2(n, kr_arr)
            hnd = dhankel2(n, kr_arr)
            temp = (1j**n) * (jn - (jnd / hnd) * hn)
            temp[kr_arr == 0] = 1.0 if n == 0 else 0.0
            out[:, n] = temp
        else:
            raise ValueError("Wrong array type")
    out[np.isnan(out)] = 0.0
    return out


def simulateSphArray(
    N_filt: int,
    mic_dirs_rad: ArrayLike,
    src_dirs_rad: ArrayLike,
    arrayType: str,
    R: float,
    N_order: int,
    fs: float,
    dirCoeff: float | None = None,
):
    f = np.arange(N_filt // 2 + 1, dtype=float) * fs / N_filt
    c = 343.0
    kR = 2 * np.pi * f * R / c
    b_n = sphModalCoeffs(N_order, kR, arrayType, dirCoeff)
    temp = b_n.copy()
    temp[-1, :] = np.real(temp[-1, :])
    b_nt = np.fft.fftshift(np.fft.ifft(np.vstack([temp, np.conj(temp[-2:0:-1, :])]), axis=0), axes=0)

    mic = np.asarray(mic_dirs_rad, dtype=float)
    src = np.asarray(src_dirs_rad, dtype=float)
    n_doa = src.shape[0]
    n_mic = mic.shape[0]

    u_mic = unit_sph_to_cart(mic[:, 0], mic[:, 1], convention="az_el")
    u_doa = unit_sph_to_cart(src[:, 0], src[:, 1], convention="az_el")

    h_mic = np.zeros((N_filt, n_mic, n_doa), dtype=np.complex128)
    h_mic_f = np.zeros((N_filt // 2 + 1, n_mic, n_doa), dtype=np.complex128)
    for i in range(n_doa):
        cosangle = u_mic @ u_doa[i]
        p = np.zeros((N_order + 1, n_mic), dtype=float)
        for n in range(N_order + 1):
            from scipy.special import eval_legendre

            p[n, :] = ((2 * n + 1) / (4 * np.pi)) * eval_legendre(n, cosangle)
        h_mic[:, :, i] = b_nt @ p
        h_mic_f[:, :, i] = b_n @ p
    return h_mic, h_mic_f


def sphericalScatterer(mic_dirs_rad: ArrayLike, src_dirs_rad: ArrayLike, R: float, N_order: int, N_filt: int, fs: float):
    f = np.arange(N_filt // 2 + 1, dtype=float) * fs / N_filt
    c = 343.0
    kR = 2 * np.pi * f * R / c

    mic = np.asarray(mic_dirs_rad, dtype=float)
    src = np.asarray(src_dirs_rad, dtype=float)
    n_mic = mic.shape[0]
    n_pw = src.shape[0]
    if np.any(mic[:, 2] < R):
        raise ValueError("measurement distance cannot be less than radius")

    all_same_rad = np.allclose(mic[:, 2], mic[0, 2])
    if all_same_rad:
        b_n = np.zeros((N_filt // 2 + 1, N_order + 1), dtype=np.complex128)
        r = mic[0, 2]
        kr = 2 * np.pi * f * r / c
        for n in range(N_order + 1):
            jn = sph_besselj(n, kr)
            jnp = dsph_besselj(n, kR)
            hn = sph_hankel2(n, kr)
            hnp = dsph_hankel2(n, kR)
            b_n[:, n] = (2 * n + 1) * (1j**n) * (jn - (jnp / hnp) * hn)
    else:
        b_n = np.zeros((N_filt // 2 + 1, N_order + 1, n_mic), dtype=np.complex128)
        for nm in range(n_mic):
            r = mic[nm, 2]
            kr = 2 * np.pi * f * r / c
            for n in range(N_order + 1):
                jn = sph_besselj(n, kr)
                jnp = dsph_besselj(n, kR)
                hn = sph_hankel2(n, kr)
                hnp = dsph_hankel2(n, kR)
                b_n[:, n, nm] = (2 * n + 1) * (1j**n) * (jn - (jnp / hnp) * hn)
    b_n[np.isnan(b_n)] = 0.0

    h_mic_f = np.zeros((N_filt // 2 + 1, n_mic, n_pw), dtype=np.complex128)
    for npw in range(n_pw):
        azi0, elev0 = src[npw, 0], src[npw, 1]
        azi, elev = mic[:, 0], mic[:, 1]
        cos_alpha = np.sin(elev) * np.sin(elev0) + np.cos(elev) * np.cos(elev0) * np.cos(azi - azi0)
        p_n = np.zeros((N_order + 1, n_mic), dtype=float)
        for n in range(N_order + 1):
            from scipy.special import eval_legendre

            p_n[n, :] = eval_legendre(n, cos_alpha)
        if all_same_rad:
            h_mic_f[:, :, npw] = b_n @ p_n
        else:
            for nm in range(n_mic):
                h_mic_f[:, nm, npw] = b_n[:, :, nm] @ p_n[:, nm]

    temp = h_mic_f.copy()
    temp[-1, :, :] = np.abs(temp[-1, :, :])
    temp = np.vstack([temp, np.conj(temp[-2:0:-1, :, :])])
    h_mic = np.fft.fftshift(np.fft.ifft(temp, axis=0), axes=0)
    return h_mic, h_mic_f


def getArrayResponse(
    U_doa: ArrayLike,
    R_mic: ArrayLike,
    U_orient: ArrayLike | None,
    fDir_handle: Any,
    Lfilt: int,
    fs: float = 48000.0,
):
    u_doa = np.asarray(U_doa, dtype=float)
    r_mic = np.asarray(R_mic, dtype=float)
    if u_doa.shape[1] != 3 or r_mic.shape[1] != 3:
        raise ValueError("U_doa and R_mic must be Nx3")
    n_mics = r_mic.shape[0]
    n_doa = u_doa.shape[0]

    if fDir_handle is None:
        fDir_handle = [lambda ang: np.ones_like(ang)] * n_mics
    elif callable(fDir_handle):
        fDir_handle = [fDir_handle] * n_mics
    elif isinstance(fDir_handle, (list, tuple)):
        if len(fDir_handle) == 1:
            fDir_handle = list(fDir_handle) * n_mics
        elif len(fDir_handle) != n_mics:
            raise ValueError("fDir_handle size should be 1xNmics")
    else:
        raise ValueError("fDir_handle should be callable or list/tuple of callables")

    u_mic = r_mic / np.maximum(np.linalg.norm(r_mic, axis=1, keepdims=True), 1e-12)
    if U_orient is None or (np.asarray(U_orient).size == 0):
        u_orient = u_mic
    else:
        u_orient = np.asarray(U_orient, dtype=float)
        if u_orient.ndim == 1:
            u_orient = np.tile(u_orient.reshape(1, 3), (n_mics, 1))
        if u_orient.shape != (n_mics, 3):
            raise ValueError("U_orient shape mismatch")

    nfft = int(Lfilt)
    f = np.arange(nfft // 2 + 1, dtype=float) * fs / nfft
    k = (2 * np.pi / 343.0) * f

    u_eval = np.repeat(u_doa[:, None, :], n_mics, axis=1)
    temp_r = np.repeat(r_mic[None, :, :], n_doa, axis=0)
    temp_o = np.repeat(u_orient[None, :, :], n_doa, axis=0)

    cos_angle_u = np.sum(u_eval * temp_o, axis=2)
    dcos_angle_u = np.sum(u_eval * temp_r, axis=2)

    b = np.zeros((n_doa, n_mics), dtype=float)
    for nm in range(n_mics):
        b[:, nm] = fDir_handle[nm](np.arccos(np.clip(cos_angle_u[:, nm], -1.0, 1.0)))

    mic_tfs = np.zeros((nfft // 2 + 1, n_mics, n_doa), dtype=np.complex128)
    for kk, kv in enumerate(k):
        temp_tf = b * np.exp(1j * kv * dcos_angle_u)
        mic_tfs[kk, :, :] = temp_tf.T

    mic_irs = np.zeros((nfft, n_mics, n_doa), dtype=np.complex128)
    for nd in range(n_doa):
        temp_tf = mic_tfs[:, :, nd].copy()
        temp_tf[-1, :] = np.abs(temp_tf[-1, :])
        temp_full = np.vstack([temp_tf, np.conj(temp_tf[-2:0:-1, :])])
        mic_irs[:, :, nd] = np.fft.fftshift(np.fft.ifft(temp_full, axis=0), axes=0)
    return mic_irs, mic_tfs


def simulateCylArray(N_filt: int, mic_dirs_rad: ArrayLike, src_dirs_rad: ArrayLike, arrayType: str, R: float, N_order: int, fs: float):
    f = np.arange(N_filt // 2 + 1, dtype=float) * fs / N_filt
    c = 343.0
    kR = 2 * np.pi * f * R / c
    b_n = cylModalCoeffs(N_order, kR, arrayType)
    temp = b_n.copy()
    temp[-1, :] = np.real(temp[-1, :])
    b_nt = np.fft.fftshift(np.fft.ifft(np.vstack([temp, np.conj(temp[-2:0:-1, :])]), axis=0), axes=0)

    mic = np.asarray(mic_dirs_rad, dtype=float).reshape(-1)
    src = np.asarray(src_dirs_rad, dtype=float).reshape(-1)
    n_mic = mic.size
    n_doa = src.size

    h_mic = np.zeros((N_filt, n_mic, n_doa), dtype=np.complex128)
    h_mic_f = np.zeros((N_filt // 2 + 1, n_mic, n_doa), dtype=np.complex128)
    for i in range(n_doa):
        angle = mic - src[i]
        cmat = np.zeros((N_order + 1, n_mic), dtype=float)
        for n in range(N_order + 1):
            cmat[n, :] = 1.0 if n == 0 else 2.0 * np.cos(n * angle)
        h_mic[:, :, i] = b_nt @ cmat
        h_mic_f[:, :, i] = b_n @ cmat
    return h_mic, h_mic_f


def cylindricalScatterer(mic_dirs_rad: ArrayLike, src_azis_rad: ArrayLike, R: float, N_order: int, N_filt: int, fs: float):
    f = np.arange(N_filt // 2 + 1, dtype=float) * fs / N_filt
    c = 343.0
    kR = 2 * np.pi * f * R / c
    mic = np.asarray(mic_dirs_rad, dtype=float)
    src = np.asarray(src_azis_rad, dtype=float).reshape(-1)
    n_mic = mic.shape[0]
    n_pw = src.size

    if np.any(mic[:, 1] < R):
        raise ValueError("measurement distance cannot be less than radius")
    all_same_rad = np.allclose(mic[:, 1], mic[0, 1])

    if all_same_rad:
        b_n = np.zeros((N_filt // 2 + 1, N_order + 1), dtype=np.complex128)
        r = mic[0, 1]
        kr = 2 * np.pi * f * r / c
        for n in range(N_order + 1):
            jn = jv(n, kr)
            jnp = dbesselj(n, kR)
            hn = hankel2(n, kr)
            hnp = dhankel2(n, kR)
            b_n[:, n] = (1j**n) * (jn - (jnp / hnp) * hn)
    else:
        b_n = np.zeros((N_filt // 2 + 1, N_order + 1, n_mic), dtype=np.complex128)
        for nm in range(n_mic):
            r = mic[nm, 1]
            kr = 2 * np.pi * f * r / c
            for n in range(N_order + 1):
                jn = jv(n, kr)
                jnp = dbesselj(n, kR)
                hn = hankel2(n, kr)
                hnp = dhankel2(n, kR)
                b_n[:, n, nm] = (1j**n) * (jn - (jnp / hnp) * hn)
    b_n[np.isnan(b_n)] = 0.0

    h_mic_f = np.zeros((N_filt // 2 + 1, n_mic, n_pw), dtype=np.complex128)
    for npw in range(n_pw):
        angle = mic[:, 0] - src[npw]
        cmat = np.zeros((N_order + 1, n_mic), dtype=float)
        for n in range(N_order + 1):
            cmat[n, :] = 1.0 if n == 0 else 2.0 * np.cos(n * angle)
        if all_same_rad:
            h_mic_f[:, :, npw] = b_n @ cmat
        else:
            for nm in range(n_mic):
                h_mic_f[:, nm, npw] = b_n[:, :, nm] @ cmat[:, nm]

    temp = h_mic_f.copy()
    temp[-1, :, :] = np.abs(temp[-1, :, :])
    temp = np.vstack([temp, np.conj(temp[-2:0:-1, :, :])])
    h_mic = np.fft.fftshift(np.fft.ifft(temp, axis=0), axes=0)
    return h_mic, h_mic_f


__all__ = [
    "cylModalCoeffs",
    "cylindricalScatterer",
    "dbesselj",
    "dbessely",
    "dhankel1",
    "dhankel2",
    "dsph_besselj",
    "dsph_bessely",
    "dsph_function",
    "dsph_hankel1",
    "dsph_hankel2",
    "getArrayResponse",
    "simulateCylArray",
    "simulateSphArray",
    "sphModalCoeffs",
    "sph_besselj",
    "sph_bessely",
    "sph_function",
    "sph_hankel1",
    "sph_hankel2",
    "sphericalScatterer",
]
