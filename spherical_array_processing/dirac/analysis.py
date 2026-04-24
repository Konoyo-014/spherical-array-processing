"""DirAC analysis: per-bin direction-of-arrival + diffuseness estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..ambi.intensity import _canonical_foa_pv
from ..types import NormalizationKind


@dataclass(frozen=True)
class DirACParameters:
    """Per-bin DirAC parameters extracted by :func:`dirac_analysis`.

    Attributes
    ----------
    direction_xyz : ndarray, shape (F, T, 3)
        Unit-norm direction-of-arrival in Cartesian coordinates
        corresponding to the **source** direction (the direction
        *from which* the wave arrives).  Components are ``[x, y, z]``.
    diffuseness : ndarray, shape (F, T)
        Bounded to ``[0, 1]``.  Zero means a single plane wave, one
        means a fully isotropic field.
    pressure : ndarray, shape (F, T), complex
        The W (omni) channel of the input STFT, retained so that
        synthesis can reuse it without re-encoding.
    freqs_hz : ndarray, shape (F,)
        Frequency axis shared with the STFT input.
    """

    direction_xyz: NDArray[np.float64]
    diffuseness: NDArray[np.float64]
    pressure: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64]


def _time_smooth_iir(x: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Single-pole IIR low-pass applied along the last axis.

    ``y[f, t] = α · x[f, t] + (1 - α) · y[f, t-1]``

    with ``y[..., 0] = x[..., 0]``.  Uses :func:`scipy.signal.lfilter`
    for O(T) runtime per frequency bin, avoiding a Python per-frame
    loop over potentially thousands of STFT frames.
    """
    from scipy.signal import lfilter

    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {a}")
    # y[t] = α · x[t] + (1 - α) · y[t-1]  ⇔  b = [α], a = [1, α - 1].
    b = np.array([a], dtype=float)
    a_coeffs = np.array([1.0, a - 1.0], dtype=float)
    # Initial condition so that y[..., 0] = x[..., 0] (rather than zero).
    zi = (1.0 - a) * x[..., :1]
    out, _ = lfilter(b, a_coeffs, x, axis=-1, zi=zi)
    return np.asarray(out, dtype=x.dtype)


def dirac_analysis(
    ambi_stft: ArrayLike,
    freqs_hz: ArrayLike,
    *,
    smoothing_alpha: float = 0.1,
    coeff_axis: int = -2,
    normalization: NormalizationKind = "orthonormal",
) -> DirACParameters:
    """Compute DirAC direction and diffuseness per time-frequency bin.

    Parameters
    ----------
    ambi_stft : array_like
        Ambisonic STFT with at least 4 SH channels along *coeff_axis*
        (W, Y, Z, X in ACN order — higher orders are allowed but
        ignored).  The expected layout is ``(F, Q, T)`` (the
        package-wide convention; matches the output of
        :func:`spherical_array_processing.stft.stft`) or ``(F, T, Q)``
        if you pass ``coeff_axis=-1``.  The diffuseness calibration
        is performed on the package's canonical FOA coefficient
        convention, so pass *normalization* accordingly — the function
        handles the conversion internally.
    freqs_hz : array_like, shape (F,)
        Frequency axis.
    smoothing_alpha : float, optional
        Single-pole IIR pole for time-smoothing the intensity and
        energy envelopes.  Default ``0.1`` corresponds to a ≈ 10-frame
        effective integration window.  Larger values track transient
        direction changes more quickly; smaller values are more
        robust for stationary diffuse components.
    coeff_axis : int, optional
        Axis that indexes SH coefficients.  Default ``-2``, matching
        the ``(F, Q, T)`` layout produced by
        :func:`spherical_array_processing.stft.stft` and consumed by
        :func:`spherical_array_processing.ambi.intensity_vector`.
    normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation of *ambi_stft*.  Default ``"orthonormal"``
        (the package's internal convention).  The function converts
        internally so the diffuseness estimate stays calibrated
        regardless of the declared input convention.  The direction
        estimate is invariant to normalisation anyway.

    Returns
    -------
    DirACParameters
        Direction unit vectors, diffuseness, omni-pressure signal, and
        frequency axis.

    Notes
    -----
    .. versionchanged:: 0.4.0b12
       Default ``coeff_axis`` changed from ``-1`` to ``-2`` so that the
       function works out of the box on STFT output in the
       ``(F, Q, T)`` layout — this is now consistent with the rest of
       the package.  Callers who previously relied on ``(F, T, Q)``
       input must now pass ``coeff_axis=-1`` explicitly.

    .. versionchanged:: 0.4.0b13
       New ``normalization`` parameter.  Previously the function
       implicitly assumed a specific FOA scaling and silently biased
       ``ψ`` away from its textbook values when given orthonormal (the
       package default) or N3D / SN3D input.  The default is now
       ``"orthonormal"`` — matching the rest of the package — and
       internal canonicalisation keeps ``ψ`` calibrated across all
       three conventions.
    """
    x = np.asarray(ambi_stft, dtype=np.complex128)
    if x.ndim != 3:
        raise ValueError(
            "ambi_stft must be 3-D ``(F, ?, ?)``; got shape " f"{x.shape}"
        )
    # Canonicalise to (F, Q, T).
    if coeff_axis in (-1, 2):
        x = np.moveaxis(x, -1, 1)  # (F, Q, T)
    elif coeff_axis in (0, -3):
        raise ValueError("coeff_axis cannot index the frequency axis (0)")
    n_bins, n_coeffs, n_frames = x.shape
    if n_coeffs < 4:
        raise ValueError(
            "ambi_stft must have at least 4 SH channels (W, Y, Z, X)"
        )
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if freqs.size != n_bins:
        raise ValueError("freqs_hz length must equal the frequency axis length")

    # DirAC's textbook ψ is written in pressure / velocity variables:
    # ``ψ = 1 − ||I|| / E`` with ``I = Re{p* v}`` and
    # ``E = 0.5·(|p|² + |v|²)``.  The canonicalisation that turns the
    # first four ambisonic channels into (p, v_x, v_y, v_z) — convert
    # to orthonormal, then divide the Cartesian dipoles by ``√3`` — is
    # shared with :func:`spherical_array_processing.ambi.intensity_vector`
    # (via the ``physical_units=True`` path) so both entry points
    # agree bit-for-bit on pressure and velocity semantics regardless
    # of the declared input normalisation.
    w, vx, vy, vz = _canonical_foa_pv(
        x[:, :4, :], normalization=normalization, coeff_axis=1,
    )

    # Instantaneous intensity vector I = Re(W* · [X, Y, Z]) and energy
    # E = 0.5 · (|W|² + |X|² + |Y|² + |Z|²).
    w_conj = np.conj(w)
    ix = np.real(w_conj * vx)
    iy = np.real(w_conj * vy)
    iz = np.real(w_conj * vz)
    intensity = np.stack([ix, iy, iz], axis=0)  # (3, F, T)
    energy = 0.5 * (
        np.abs(w) ** 2
        + np.abs(vx) ** 2
        + np.abs(vy) ** 2
        + np.abs(vz) ** 2
    )

    # Time-smoothing via a single-pole IIR on both intensity and
    # energy.  Operates on the last axis (time).
    i_smoothed = np.stack(
        [_time_smooth_iir(intensity[k], smoothing_alpha) for k in range(3)],
        axis=0,
    )
    e_smoothed = _time_smooth_iir(energy, smoothing_alpha)

    i_mag = np.linalg.norm(i_smoothed, axis=0)  # (F, T)
    eps = np.finfo(float).tiny
    direction = np.where(
        i_mag[None, :, :] > eps,
        i_smoothed / np.maximum(i_mag[None, :, :], eps),
        0.0,
    )  # (3, F, T) — degenerate bins get zero direction.
    diffuseness = np.clip(
        1.0 - i_mag / np.maximum(e_smoothed, eps), 0.0, 1.0
    )

    return DirACParameters(
        direction_xyz=np.moveaxis(direction, 0, -1),  # (F, T, 3)
        diffuseness=diffuseness,
        pressure=w,
        freqs_hz=freqs,
    )
