"""Encoding filters driven by measured spherical-array steering matrices.

The theoretical radial equalizers in :mod:`.radial_filters` assume an
idealised open / rigid / cardioid array described analytically by
``B_n(kr)``.  Real spherical microphone arrays deviate from the model
due to transducer mismatches, housing diffraction, and sensor-position
errors.  When a per-direction steering matrix has been measured on an
anechoic grid, it can be inverted directly to obtain a regularized
encoding filter that includes those deviations.

This module exposes two such designs originally implemented by Politis
in MATLAB and now available natively in Python:

* ``method="regLS"`` — regularized pseudo-inverse of the measurement
  matrix, solved bin-by-bin with a white-noise-gain floor.
* ``method="regLSHD"`` — project the measurement matrix onto a
  higher-order SH *array* basis first, then pseudo-invert; equivalent
  to the ``_regLSHD`` variant in the Politis toolbox.

The returned frequency-domain filter has shape ``(F, Q, M)`` where
``F`` is the one-sided bin count, ``Q = (N+1)²`` the SH channel count,
and ``M`` the number of microphones.  Apply it with
:func:`apply_measured_equalizer` to a multichannel mic STFT of shape
``(F, M, T)``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._measured_sht_filters import (
    arraySHTfiltersMeas_regLS,
    arraySHTfiltersMeas_regLSHD,
)
from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SphericalGrid


MeasuredEqualizerMethod = Literal["regLS", "regLSHD"]


def measured_array_equalizer(
    H_array: ArrayLike,
    grid_dirs_az_el_rad: ArrayLike,
    *,
    max_order: int,
    n_fft: int,
    w_grid: ArrayLike | None = None,
    amp_threshold_db: float = 15.0,
    method: MeasuredEqualizerMethod = "regLS",
    return_fir: bool = False,
) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Encoding filter from a measured array steering matrix.

    Parameters
    ----------
    H_array : array_like, shape ``(F, M, G)``
        Measured complex frequency response of the array.  ``F`` may
        be the one-sided bin count ``n_fft // 2 + 1`` or the full
        ``n_fft`` — the full-spectrum form is truncated internally.
    grid_dirs_az_el_rad : array_like, shape ``(G, 2)``
        Grid directions ``(azimuth, elevation)`` in radians.
    max_order : int
        Target encoding SH order ``N``.  Capped internally at
        ``floor(sqrt(M) - 1)`` (the Nyquist SH order for the array).
    n_fft : int
        FFT length that produced *H_array*.  Used to derive the
        one-sided-spectrum width and to fold the filter back to a
        real-valued time-domain FIR when *return_fir* is true.
    w_grid : array_like or None, optional
        Per-direction integration weights.  ``None`` uses uniform
        weights.
    amp_threshold_db : float, optional
        White-noise-gain ceiling in dB.  Larger values → sharper
        inversion / more noise; smaller values → gentler inversion.
        Default ``15``.
    method : {"regLS", "regLSHD"}, optional
        Regularization strategy.  ``"regLS"`` is the direct
        regularized pseudo-inverse; ``"regLSHD"`` first projects the
        measurement matrix onto an SH array basis of order
        ``floor(sqrt(G)/2 − 1)`` before inverting.  ``"regLSHD"`` is
        preferable for dense measurement grids because it decouples
        the SH decomposition from the microphone inversion.
    return_fir : bool, optional
        If true, also return a real-valued FIR filter of length
        *n_fft* (first axis of the second return) for offline
        convolutional encoding.

    Returns
    -------
    H_filt : ndarray, shape ``(F, Q, M)``, complex
        Frequency-domain encoding filter.  ``Q = (N+1)²``.
    h_filt : ndarray, shape ``(n_fft, Q, M)``, real
        Only returned when *return_fir* is true.  Linear-phase FIR
        obtained by zero-phase inverse FFT + fftshift; apply with
        :func:`scipy.signal.oaconvolve` along the time axis.

    Notes
    -----
    The sign convention matches the rest of this package:
    ``sh_stft[f] = H_filt[f] @ mic_stft[f]`` reconstructs the SH
    plane-wave-steering signal from the microphone STFT, i.e.
    ``H_filt[f]`` takes the role of the pseudo-inverse of the array
    steering matrix at frequency ``f``.
    """
    if method == "regLS":
        impl = arraySHTfiltersMeas_regLS
    elif method == "regLSHD":
        impl = arraySHTfiltersMeas_regLSHD
    else:
        raise ValueError(
            f"method must be 'regLS' or 'regLSHD', got {method!r}"
        )

    n_fft_int = int(n_fft)
    if n_fft_int <= 0:
        raise ValueError("n_fft must be positive")

    H_f, h_t = impl(
        H_array,
        int(max_order),
        grid_dirs_az_el_rad,
        w_grid,
        n_fft_int,
        float(amp_threshold_db),
    )
    # Politis convention: (Q, M, F) / (Q, M, n_fft).  Reorder to the
    # package-wide (F, Q, M) / (n_fft, Q, M) layout.
    H_public = np.ascontiguousarray(np.moveaxis(H_f, -1, 0))
    if not return_fir:
        return H_public
    h_public = np.ascontiguousarray(np.moveaxis(h_t, -1, 0))
    return H_public, h_public


def apply_measured_equalizer(
    mic_stft: ArrayLike,
    equalizer: ArrayLike,
    *,
    freq_axis: int = 0,
    mic_axis: int = 1,
) -> NDArray[np.complex128]:
    """Apply a measured-array encoding filter to a multichannel STFT.

    Computes ``sh_stft[f, :, t] = equalizer[f] @ mic_stft[f, :, t]``
    for every frequency bin, returning an array with the microphone
    axis replaced by the SH channel axis.

    Parameters
    ----------
    mic_stft : array_like
        Multichannel microphone STFT.  Default layout ``(F, M, T)``.
    equalizer : array_like, shape ``(F, Q, M)``
        Output of :func:`measured_array_equalizer`.
    freq_axis, mic_axis : int, optional
        Override if *mic_stft* uses a different layout.  The remaining
        axes (if any) are preserved in their relative order.

    Returns
    -------
    ndarray
        SH-domain STFT with the same layout as *mic_stft* except that
        the microphone axis has length ``Q`` instead of ``M``.
    """
    eq = np.asarray(equalizer, dtype=np.complex128)
    if eq.ndim != 3:
        raise ValueError(
            f"equalizer must be 3-D (F, Q, M); got shape {eq.shape}"
        )
    x = np.asarray(mic_stft)
    n = x.ndim
    f_ax = freq_axis % n
    m_ax = mic_axis % n
    if f_ax == m_ax:
        raise ValueError("freq_axis and mic_axis must be different")
    # Permute to (F, M, ...rest) and invert afterwards.
    rest = [a for a in range(n) if a not in (f_ax, m_ax)]
    x_perm = np.transpose(x, (f_ax, m_ax, *rest))
    if x_perm.shape[0] != eq.shape[0]:
        raise ValueError(
            f"frequency-axis mismatch: mic_stft has {x_perm.shape[0]} "
            f"bins, equalizer has {eq.shape[0]}"
        )
    if x_perm.shape[1] != eq.shape[2]:
        raise ValueError(
            f"microphone-axis mismatch: mic_stft has {x_perm.shape[1]} "
            f"channels, equalizer has {eq.shape[2]}"
        )
    out = np.einsum("fqm,fm...->fq...", eq, x_perm)
    # Move Q into the old mic_axis slot.
    target = [0] + list(range(2, n)) + [1]  # current: F, Q, rest — want to map back
    inverse = list(range(n))
    # Build inverse permutation: (f_ax, m_ax, *rest) → original order
    perm = [f_ax, m_ax, *rest]
    for dst, src in enumerate(perm):
        inverse[src] = dst
    return np.transpose(out, inverse)


def measured_array_diagnostics(
    H_array: ArrayLike,
    equalizer: ArrayLike,
    grid_dirs_az_el_rad: ArrayLike,
    *,
    max_order: int,
) -> dict:
    """Report numerical health metrics for a measured-array encoder.

    Parameters
    ----------
    H_array : array_like, shape ``(F, M, G)``
        Measured array steering matrix.
    equalizer : array_like, shape ``(F, Q, M)``
        Encoding matrix, typically from :func:`measured_array_equalizer`.
    grid_dirs_az_el_rad : array_like, shape ``(G, 2)``
        Measurement directions as azimuth/elevation in radians.
    max_order : int
        Target SH order ``N``.

    Returns
    -------
    dict
        Machine-readable diagnostics: steering-matrix condition
        numbers, per-bin reconstruction error against the target SH
        basis, white-noise-gain range, and maximum filter gain.
    """
    H = np.asarray(H_array, dtype=np.complex128)
    E = np.asarray(equalizer, dtype=np.complex128)
    dirs = np.asarray(grid_dirs_az_el_rad, dtype=float)
    if H.ndim != 3:
        raise ValueError("H_array must have shape (F, M, G)")
    if E.ndim != 3:
        raise ValueError("equalizer must have shape (F, Q, M)")
    if dirs.ndim != 2 or dirs.shape[1] != 2:
        raise ValueError("grid_dirs_az_el_rad must have shape (G, 2)")
    if H.shape[0] != E.shape[0]:
        raise ValueError("H_array and equalizer must have the same frequency count")
    if H.shape[1] != E.shape[2]:
        raise ValueError("microphone count mismatch between H_array and equalizer")
    expected_q = (int(max_order) + 1) ** 2
    if E.shape[1] != expected_q:
        raise ValueError(
            f"equalizer has {E.shape[1]} SH channels; expected {expected_q}"
        )
    if H.shape[2] != dirs.shape[0]:
        raise ValueError("direction count mismatch between H_array and grid_dirs")

    grid = SphericalGrid(
        azimuth=dirs[:, 0],
        angle2=dirs[:, 1],
        convention="az_el",
    )
    Y = np.asarray(
        sh_matrix(SHBasisSpec(max_order=int(max_order), basis="real"), grid)
    ).T  # (Q, G)
    reconstructed = np.einsum("fqm,fmg->fqg", E, H)
    residual = reconstructed - Y[None, :, :]
    target_norm = np.maximum(np.linalg.norm(Y), 1e-30)
    reconstruction_error = np.linalg.norm(residual, axis=(1, 2)) / target_norm

    condition_numbers = np.empty(H.shape[0], dtype=float)
    for f_idx in range(H.shape[0]):
        condition_numbers[f_idx] = float(np.linalg.cond(H[f_idx]))
    wng = 1.0 / np.maximum(np.sum(np.abs(E) ** 2, axis=2), 1e-30)
    wng_db = 10.0 * np.log10(wng)
    max_filter_gain_db = 20.0 * np.log10(
        np.maximum(np.max(np.abs(E), axis=(1, 2)), 1e-30)
    )
    return {
        "n_freqs": int(H.shape[0]),
        "n_mics": int(H.shape[1]),
        "n_directions": int(H.shape[2]),
        "n_coeffs": int(E.shape[1]),
        "max_order": int(max_order),
        "condition_numbers": condition_numbers,
        "condition_number_max": float(np.max(condition_numbers)),
        "condition_number_median": float(np.median(condition_numbers)),
        "reconstruction_error": reconstruction_error,
        "reconstruction_error_mean": float(np.mean(reconstruction_error)),
        "reconstruction_error_max": float(np.max(reconstruction_error)),
        "white_noise_gain_db": wng_db,
        "white_noise_gain_db_min": float(np.min(wng_db)),
        "white_noise_gain_db_max": float(np.max(wng_db)),
        "max_filter_gain_db": max_filter_gain_db,
        "max_filter_gain_db_max": float(np.max(max_filter_gain_db)),
    }


__all__ = [
    "apply_measured_equalizer",
    "measured_array_diagnostics",
    "measured_array_equalizer",
]
