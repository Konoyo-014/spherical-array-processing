"""Bilateral-Ambisonics MagLS (BiMagLS) binaural rendering.

BiMagLS (Engel, Goodwin, Alon 2021) improves low-order MagLS by
**time-aligning** each ear's HRTF before applying the magnitude
least-squares fit: the bulk of the inter-aural time difference (ITD)
is factored out from the phase, so the remaining residual that must be
captured in ``(N+1)²`` SH coefficients is small and well-behaved even
at low orders.  The returned filter pair is *delay-aligned*; the
per-direction ear delay is itself returned as SH coefficients so the
user can reattach it at render time.

Reference
---------
.. [1] I. Engel, D. Goodwin, D. Alon, "Improving Binaural Rendering
   with Bilateral Ambisonics and MagLS", *Proc. AES Immersive and
   Interactive Audio Conference*, 2021.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..coords import unit_sph_to_cart
from ..sh import direct_sht, matrix as sh_matrix
from ..types import SHBasisSpec, SphericalGrid
from .magls import magls_binaural_filters


BasisKind = Literal["real", "complex"]


def _per_direction_ear_delays(
    hrtf_grid: SphericalGrid,
    ear_positions_xyz: NDArray[np.float64],
    c: float,
) -> NDArray[np.float64]:
    """Geometric arrival-time delay between the acoustic centre and
    each ear under a free-field plane-wave assumption.

    For a plane wave from direction ``k̂`` and an ear at position
    ``r_ear`` (both unit-norm in the same Cartesian frame), the wave
    hits the ear earlier than the origin by
    ``τ_ear(k̂) = (k̂ · r_ear) / c``.  The sign convention is such that
    applying ``exp(+j 2π f τ)`` to the HRTF pushes the ear's response
    back to the origin timebase.
    """
    directions = unit_sph_to_cart(
        hrtf_grid.azimuth,
        hrtf_grid.angle2,
        convention=hrtf_grid.convention,
    )  # (G, 3)
    delays = directions @ ear_positions_xyz.T / c  # (G, 2)
    return delays.astype(float, copy=False)


def bimagls_binaural_filters(
    hrtfs: ArrayLike,
    freqs_hz: ArrayLike,
    hrtf_grid: SphericalGrid,
    max_order: int,
    ear_positions_m: ArrayLike,
    *,
    basis: BasisKind = "real",
    c: float = 343.0,
    f_cut_hz: float = 1500.0,
    n_iterations: int = 10,
    phase_continuation: bool = True,
    rcond: float = 1e-8,
) -> tuple[
    NDArray[np.complex128], NDArray[np.complex128] | NDArray[np.float64]
]:
    """Bilateral-Ambisonics MagLS SH-to-binaural rendering filters.

    The pipeline per ear is:

    1. Compute the geometric ear-to-origin delay table
       ``τ_ear(k̂) = (k̂ · r_ear) / c`` for every direction in
       *hrtf_grid*.
    2. Remove that delay from the HRTF phase
       ``H̃(f, k̂) = H(f, k̂) · exp(+j 2π f · τ_ear(k̂))``.
    3. Fit a MagLS filter to the delay-aligned HRTFs via
       :func:`magls_binaural_filters`.
    4. SH-encode the delay table so the caller can re-apply it at
       render time.

    Parameters
    ----------
    hrtfs : array_like, shape (F, G, 2)
        Complex HRTFs at F frequency bins, G measurement directions,
        and two ears (0 = left, 1 = right).
    freqs_hz : array_like, shape (F,)
        Frequency axis in Hz.
    hrtf_grid : SphericalGrid
        Direction grid with ``G`` measurement points.
    max_order : int
        Ambisonic order ``N``.
    ear_positions_m : array_like, shape (2, 3)
        Cartesian ear positions relative to the acoustic centre, in
        metres.  Typical anthropometric value: ``(±0.09, 0, 0)``
        metres with the inter-aural axis aligned with ``x``.
    basis : {"real", "complex"}, optional
        SH basis to target.
    c : float, optional
        Speed of sound in m/s.  Default ``343.0``.
    f_cut_hz : float, optional
        MagLS cutoff frequency (complex LS below, magnitude LS above).
    n_iterations : int, optional
        Number of MagLS iterations per high-frequency bin.
    phase_continuation : bool, optional
        Whether to seed MagLS with the previous bin's phase
        (recommended; see :func:`magls_binaural_filters`).
    rcond : float, optional
        ``numpy.linalg.pinv`` cut-off.

    Returns
    -------
    filters : ndarray, shape (F, (N+1)², 2), complex128
        Delay-aligned SH rendering filters.  Applying them to an
        ambisonic signal yields a binaural output **with the bulk ITD
        removed**; re-attach the ITD via *delay_sh_coeffs* at render
        time.
    delay_sh_coeffs : ndarray, shape ((N+1)², 2)
        SH-domain coefficients of the per-direction ear delay.  In the
        real basis they are real-valued up to numerical precision; in
        the complex basis they are generally complex.  At a scan
        direction ``k̂`` the delay is recovered by
        ``τ_ear(k̂) = Y(k̂) @ delay_sh_coeffs[:, ear]``.  Apply the
        delay at render time either as a frequency-domain phase shift
        ``exp(-j 2π f τ)`` or as a per-direction fractional delay in
        the time domain.
    """
    h = np.asarray(hrtfs, dtype=np.complex128)
    if h.ndim != 3 or h.shape[-1] != 2:
        raise ValueError("hrtfs must have shape (F, G, 2); got " f"{h.shape}")
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    ear_xyz = np.asarray(ear_positions_m, dtype=float)
    if ear_xyz.shape != (2, 3):
        raise ValueError(
            "ear_positions_m must have shape (2, 3); got " f"{ear_xyz.shape}"
        )

    delays = _per_direction_ear_delays(hrtf_grid, ear_xyz, c)  # (G, 2)
    aligned = np.empty_like(h)
    for ear in (0, 1):
        phase = np.exp(1j * 2.0 * np.pi * freqs[:, None] * delays[None, :, ear])
        aligned[:, :, ear] = h[:, :, ear] * phase

    filters = magls_binaural_filters(
        aligned,
        freqs,
        hrtf_grid,
        max_order=max_order,
        basis=basis,
        f_cut_hz=f_cut_hz,
        n_iterations=n_iterations,
        phase_continuation=phase_continuation,
        rcond=rcond,
    )

    # SH-encode the per-direction delay so the caller can re-apply it
    # at render time without re-measuring the geometry.
    spec = SHBasisSpec(
        max_order=int(max_order),
        basis=basis,
        angle_convention=hrtf_grid.convention,
    )
    y = np.asarray(sh_matrix(spec, hrtf_grid))
    delay_sh_coeffs = np.real_if_close(
        np.stack(
            [direct_sht(delays[:, ear], y, hrtf_grid) for ear in (0, 1)],
            axis=-1,
        )
    )
    return filters, delay_sh_coeffs


__all__ = ["bimagls_binaural_filters"]
