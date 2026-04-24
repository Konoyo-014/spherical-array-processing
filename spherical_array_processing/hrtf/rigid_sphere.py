"""Analytic rigid-sphere head-related transfer function.

A simple pressure-sensor on a rigid sphere gives a closed-form HRTF
via the Legendre/Bessel series

``H(kR, γ) = Σ_n b_n(kR) · Σ_{m=-n}^{n} Y_{nm}(r̂_ear) · Y_{nm}*(r̂_src)``

where ``b_n(kR)`` is the modal coefficient of the rigid sphere (see
:func:`spherical_array_processing.acoustics.radial.bn_matrix`) and
``γ`` is the angle between the source direction and the ear position
on the sphere.  This is the standard Rayleigh / Duda-Martens model
used for ambisonic binaural testbeds when measured HRTF data are not
available.

Human heads are well approximated by a sphere of radius ``0.085 m``
(≈ 8.5 cm) with ears offset by ± 90 ° in azimuth.  This module
returns an :class:`HRTFDataset` consumable by the MagLS / BiMagLS
renderers and the :func:`ambi_to_binaural_time_domain` pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..acoustics.radial import bn_matrix
from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SphericalGrid
from .dataset import HRTFDataset


def _validate_ear_positions(
    ear_positions_m: ArrayLike, head_radius_m: float, *, atol: float = 1e-3,
) -> NDArray[np.float64]:
    ears = np.asarray(ear_positions_m, dtype=float)
    if ears.shape != (2, 3):
        raise ValueError(
            f"ear_positions_m must have shape (2, 3); got {ears.shape}"
        )
    norms = np.linalg.norm(ears, axis=1)
    if np.any(np.abs(norms - head_radius_m) > atol):
        raise ValueError(
            "ear_positions_m must lie on the sphere surface (|r| = "
            f"head_radius_m); got norms {norms} for radius "
            f"{head_radius_m}"
        )
    return ears


def rigid_sphere_hrtf(
    head_radius_m: float,
    ear_positions_m: ArrayLike,
    source_grid: SphericalGrid,
    fs: float,
    n_taps: int,
    *,
    max_order: int = 30,
    c: float = 343.0,
) -> HRTFDataset:
    """Closed-form rigid-sphere HRTF on the given source grid.

    Parameters
    ----------
    head_radius_m : float
        Sphere radius in metres.  ``0.085 m`` is a typical human-head
        value (Kuhn 1977).
    ear_positions_m : array_like, shape ``(2, 3)``
        Cartesian positions of the left (index 0) and right (index 1)
        ears.  Must lie exactly on the sphere surface (to within 1 mm
        by default — use a tighter tolerance if your inputs are
        computed).
    source_grid : SphericalGrid
        Directions on the unit sphere at which to evaluate the HRTF.
    fs : float
        Sampling rate in Hz.  Used to derive the frequency axis for
        the FFT and the output HRIR length / FFT length.
    n_taps : int
        Length of the returned HRIR in samples.  The FFT length is
        ``n_taps`` (no zero-pad).  Choose ``n_taps`` large enough to
        contain the ITD plus the low-order ringing of the sphere
        response — 256 at ``fs = 16 kHz`` / 512 at ``fs = 48 kHz``
        typically suffices.
    max_order : int, optional
        SH truncation for the modal series.  ``30`` (default) is
        more than enough for human-head-size accuracy up to 20 kHz
        because ``kR`` stays under ≈ 31 there.
    c : float, optional
        Speed of sound.  Default ``343`` m/s.

    Returns
    -------
    HRTFDataset
        Time-domain HRIRs shaped ``(G, 2, n_taps)`` with the supplied
        source grid and ear positions.  The ``metadata`` dict records
        the model parameters for reproducibility.
    """
    head_r = float(head_radius_m)
    if head_r <= 0.0:
        raise ValueError("head_radius_m must be positive")
    fs_f = float(fs)
    if fs_f <= 0.0:
        raise ValueError("fs must be positive")
    n = int(n_taps)
    if n <= 0:
        raise ValueError("n_taps must be positive")
    n_bins = n // 2 + 1
    freqs = np.arange(n_bins, dtype=float) * fs_f / n
    kR = 2.0 * np.pi * freqs * head_r / float(c)

    ears = _validate_ear_positions(ear_positions_m, head_r)
    # Normalise ear positions to the unit sphere for SH evaluation.
    ear_unit = ears / head_r
    # Build a SphericalGrid for the two ears.  We need az/colat for
    # sh_matrix, which internally handles either convention.
    ear_az = np.arctan2(ear_unit[:, 1], ear_unit[:, 0]) % (2.0 * np.pi)
    ear_col = np.arccos(np.clip(ear_unit[:, 2], -1.0, 1.0))
    ear_grid = SphericalGrid(
        azimuth=ear_az, angle2=ear_col, convention="az_colat",
    )

    # SH matrices at ears (2, Q) and sources (G, Q), complex basis.
    spec = SHBasisSpec(
        max_order=int(max_order), basis="complex",
        angle_convention="az_colat",
    )
    Y_ear = np.asarray(sh_matrix(spec, ear_grid))        # (2, Q)
    Y_src = np.asarray(sh_matrix(spec, source_grid))     # (G, Q)

    bn = bn_matrix(
        int(max_order), kr=kR, sphere="rigid", repeat_per_order=True,
    )  # (F, Q)
    # Handle the DC bin: bn_matrix may produce NaN at kR=0 for n≥1.
    # Under the package's orthonormal SH convention,
    # ``Σ_m Y_nm(r̂) Y_nm*(ŝ) = (2n+1)/(4π) P_n(cos γ)``.  Therefore the
    # rigid-sphere DC limit needs ``b_0(0) = 4π`` and ``b_n(0) = 0`` for
    # n≥1 so that the omnidirectional response becomes exactly unity.
    bn = np.nan_to_num(bn, nan=0.0, posinf=0.0, neginf=0.0)
    bn[0, 0] = 4.0 * np.pi

    # HRTF frequency response: H[f, g, ear] = Σ_q bn[f,q] · Y_ear[ear,q] · Y_src*[g,q]
    H = np.einsum(
        "fq,eq,gq->fge", bn, Y_ear, np.conj(Y_src),
    )  # (F, G, 2)
    H[0] = np.real(H[0])
    if n % 2 == 0:
        H[-1] = np.real(H[-1])

    # Inverse RFFT to time-domain HRIRs.  The closed-form response
    # references phase at the sphere centre, so the impulse response
    # out of ``irfft`` sits partly at negative time (wrapped to the
    # tail of the buffer).  fftshift along the time axis re-centres
    # the main energy at sample ``n_taps // 2`` — a bulk delay of
    # ``n_taps / (2·fs)`` seconds applied identically to both ears, so
    # the physical ITD is preserved.
    hrirs_time = np.fft.irfft(H, n=n, axis=0)           # (n_taps, G, 2)
    hrirs_time = np.fft.fftshift(hrirs_time, axes=0)
    hrirs = hrirs_time.transpose(1, 2, 0)
    hrirs = np.asarray(hrirs, dtype=np.float64)

    return HRTFDataset(
        hrirs=hrirs,
        fs=fs_f,
        source_grid=source_grid,
        ear_positions_m=ears,
        metadata={
            "DatabaseName": "rigid-sphere (analytic)",
            "head_radius_m": head_r,
            "max_order": int(max_order),
            "c": float(c),
        },
    )


__all__ = ["rigid_sphere_hrtf"]
