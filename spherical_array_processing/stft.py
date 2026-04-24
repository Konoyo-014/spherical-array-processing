"""Short-Time Fourier Transform helpers.

Thin wrappers around :mod:`scipy.signal` that reshape the output into
the ``(F, M, T)`` frequency-channel-frame layout consumed throughout
this package (see :func:`spherical_array_processing.doa.srp_map`,
:func:`spherical_array_processing.encoding.apply_radial_equalizer`,
etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import stft as _scipy_stft, istft as _scipy_istft


def stft(
    signal: ArrayLike,
    fs: float,
    *,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: Any = "hann",
    nfft: int | None = None,
    return_onesided: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    """Compute the STFT of a possibly multichannel signal.

    Parameters
    ----------
    signal : array_like
        Time-domain signal.  ``shape == (n_samples,)`` for mono or
        ``shape == (n_channels, n_samples)`` for multichannel.
    fs : float
        Sampling rate in Hz.
    nperseg : int, optional
        Window length in samples.  Default ``1024``.
    noverlap : int or None, optional
        Overlap in samples.  Default ``nperseg // 2`` (50 %).
    window : str | tuple | array_like, optional
        Window specification passed through to :func:`scipy.signal.stft`.
    nfft : int or None, optional
        FFT length.  Defaults to *nperseg*.
    return_onesided : bool, optional
        Whether to return only non-negative frequencies.  Default
        ``True`` (real-to-complex output).

    Returns
    -------
    freqs : ndarray, shape (F,)
        Frequency bin centres in Hz.
    times : ndarray, shape (T,)
        Frame-centre timestamps in seconds.
    Z : ndarray
        Complex STFT with layout ``(F, T)`` for mono input and
        ``(F, M, T)`` for multichannel input — the same layout expected
        by :func:`spherical_array_processing.doa.srp_map`.
    """
    x = np.asarray(signal)
    if x.ndim == 1:
        freqs, times, z = _scipy_stft(
            x,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=return_onesided,
            boundary="zeros",
            padded=True,
        )
        return freqs, times, z
    if x.ndim != 2:
        raise ValueError(
            f"signal must be 1-D or 2-D (n_channels, n_samples); got shape {x.shape}"
        )
    # scipy's stft with axis=-1 treats the last axis as time and
    # prepends frequency/time axes — result shape (M, F, T).  We
    # transpose into (F, M, T) to match the srp_map / encoding APIs.
    freqs, times, z = _scipy_stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=return_onesided,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    # z shape: (M, F, T)
    z = np.moveaxis(z, 0, 1)  # -> (F, M, T)
    return freqs, times, z


def istft(
    Z: ArrayLike,
    fs: float,
    *,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: Any = "hann",
    nfft: int | None = None,
    input_onesided: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse STFT, matching the ``(F, M, T)`` layout of :func:`stft`.

    Parameters
    ----------
    Z : array_like
        Complex STFT with shape ``(F, T)`` (mono) or ``(F, M, T)``
        (multichannel).
    fs : float
        Sampling rate in Hz.
    nperseg, noverlap, window, nfft, input_onesided : optional
        Passed through to :func:`scipy.signal.istft`; use the same
        values as the forward :func:`stft` call.

    Returns
    -------
    times : ndarray, shape (n_samples,)
        Reconstructed time axis in seconds.
    signal : ndarray
        Reconstructed time-domain signal — shape ``(n_samples,)`` for
        mono input or ``(n_channels, n_samples)`` for multichannel.
    """
    z = np.asarray(Z)
    if z.ndim == 2:
        times, x = _scipy_istft(
            z,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            input_onesided=input_onesided,
            boundary=True,
        )
        return times, x
    if z.ndim != 3:
        raise ValueError(
            f"Z must be 2-D (F, T) or 3-D (F, M, T); got shape {z.shape}"
        )
    # (F, M, T) -> scipy expects axis ordering with F as the 1st axis
    # and M treated as a leading batch dim.  Reorder to (M, F, T).
    z_scipy = np.moveaxis(z, 1, 0)  # (M, F, T)
    times, x = _scipy_istft(
        z_scipy,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=input_onesided,
        boundary=True,
    )
    return times, x


__all__ = ["stft", "istft"]
