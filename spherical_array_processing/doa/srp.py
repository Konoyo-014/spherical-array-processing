"""Steered-Response Power with Phase Transform (SRP-PHAT) DOA estimator.

Classical mic-domain SRP(-PHAT) is more robust than narrow-band PWD /
MUSIC at low SNR and for moderate-sized arrays, because it integrates
evidence across the full signal bandwidth and (optionally) whitens each
frequency so that spectral coloration of the source does not bias the
map.

Formulation
-----------
For microphone positions ``r_m`` and a scan direction ``q̂``, the
plane-wave inter-mic delay under the DOA convention used throughout
this package is

``τ_{mn}(q̂) = (r_m - r_n) · q̂ / c.``

Under the same DOA convention the simulated plane-wave response is
``X[f, m] ∝ exp(+j 2π f · û_s · r_m / c)``.  Conjugate-aligning those
phases to scan direction ``q̂`` means multiplying the microphone signal
by ``exp(-j 2π f · r_m · q̂ / c)``, so the SRP power map is

``P(q̂) = Σ_f |Σ_m W[f, m] · X[f, m] · exp(-j 2π f · r_m · q̂ / c)|²``

with ``W[f, m] = 1`` for vanilla SRP and ``W[f, m] = 1 / |X[f, m]|``
for the phase-transform (PHAT) variant.  (The covariance-based entry
point :func:`srp_map_from_covariance` writes the algebraically
equivalent Hermitian form ``sᴴ R s`` with ``s = exp(+j 2π f · r_m · q̂
/ c)``.)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..coords import unit_sph_to_cart
from ..types import ArrayGeometry, SpatialSpectrumResult, SphericalGrid
from .spectra import spatial_spectrum_from_map


PhatMode = Literal["none", "phat"]


def srp_map(
    stft_bins: ArrayLike,
    freqs_hz: ArrayLike,
    geometry: ArrayGeometry,
    scan_grid: SphericalGrid,
    *,
    weighting: PhatMode = "phat",
    c: float = 343.0,
    freq_range_hz: tuple[float, float] | None = None,
    normalize: bool = True,
    n_peaks: int = 1,
    min_separation_deg: float | None = None,
) -> SpatialSpectrumResult:
    """Compute the broadband SRP(-PHAT) spectrum on *scan_grid*.

    Parameters
    ----------
    stft_bins : array_like, shape (F, M) or (F, M, T)
        Microphone-domain STFT.  The leading axis is frequency with
        length ``F`` matching *freqs_hz*; the second is the microphone
        index.  An optional third axis indexes time frames and is
        averaged internally (power summed across frames).
    freqs_hz : array_like, shape (F,)
        Frequency axis in Hz corresponding to the first dimension of
        *stft_bins*.
    geometry : ArrayGeometry
        Uniform-radius microphone geometry.
    scan_grid : SphericalGrid
        Scan directions (interpreted as DOA).
    weighting : {"phat", "none"}, optional
        ``"phat"`` (default) normalises each frequency-microphone entry
        by its magnitude, producing the classical SRP-PHAT response;
        ``"none"`` gives the plain steered-sum power.
    c : float, optional
        Speed of sound in m/s.
    freq_range_hz : tuple[float, float] or None, optional
        Only bins within ``(f_low, f_high)`` contribute.  Useful for
        excluding DC and near-Nyquist bins that typically add only noise.
    normalize : bool, optional
        If ``True`` (default) scale the output so its maximum equals 1,
        which makes peak-picking numerically stable.
    n_peaks : int, optional
        Number of peaks to report in the :class:`SpatialSpectrumResult`.

    Returns
    -------
    SpatialSpectrumResult
        Spatial spectrum evaluated on *scan_grid*.
    """
    x = np.asarray(stft_bins, dtype=np.complex128)
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError("stft_bins must have shape (F, M) or (F, M, T)")
    if x.shape[0] != f.size:
        raise ValueError("freqs_hz length must equal the frequency axis of stft_bins")
    n_bins, n_mics, n_frames = x.shape

    mic_xyz = unit_sph_to_cart(
        geometry.sensor_grid.azimuth,
        geometry.sensor_grid.angle2,
        convention=geometry.sensor_grid.convention,
    ) * float(geometry.radius_m)  # [M, 3]
    if mic_xyz.shape[0] != n_mics:
        raise ValueError("geometry microphone count does not match stft_bins")

    scan_u = unit_sph_to_cart(
        scan_grid.azimuth, scan_grid.angle2, convention=scan_grid.convention
    )  # [G, 3]

    # PHAT weighting — whiten each bin/mic/frame pair by its magnitude.
    if weighting == "phat":
        mag = np.abs(x)
        mag = np.where(mag > 1e-20, mag, 1.0)
        x_weighted = x / mag
    elif weighting == "none":
        x_weighted = x
    else:
        raise ValueError(f"weighting must be 'phat' or 'none', got {weighting!r}")

    mask = np.ones(n_bins, dtype=bool)
    if freq_range_hz is not None:
        f_low, f_high = freq_range_hz
        mask &= (f >= float(f_low)) & (f <= float(f_high))
    mask &= f > 0.0  # exclude DC to avoid a constant offset bias

    if not np.any(mask):
        raise ValueError("no frequency bins active — check freq_range_hz")

    # Pre-compute the mic·scan dot product — shape (M, G).
    proj = mic_xyz @ scan_u.T  # (M, G)

    # Streamed accumulation — avoids a (F, M, G) tensor when G is large.
    #
    # The beamformer conjugate-aligns the plane-wave phase
    # exp(+j k q̂ · r_m) that a source from direction q̂ imprints onto
    # microphone m.  Its weight is therefore exp(-j k q̂ · r_m), i.e. the
    # negative of the steering projection.  With the DOA convention
    # anywhere else in this package (the Jacobi-Anger expansion uses
    # ``+j k · k̂_src · r``), using ``exp(+j k · proj)`` here would make
    # the output peak at the **antipodal** direction.
    p_map = np.zeros(scan_u.shape[0], dtype=float)
    for fi in np.nonzero(mask)[0]:
        k = 2.0 * np.pi * f[fi] / c
        s = np.exp(-1j * k * proj)  # (M, G)
        bf = x_weighted[fi].T @ s  # (T, G) since x_weighted[fi] has shape (M, T)
        p_map += np.sum(np.abs(bf) ** 2, axis=0)

    if normalize and np.max(p_map) > 0:
        p_map = p_map / np.max(p_map)

    method = "srp_phat" if weighting == "phat" else "srp"
    return spatial_spectrum_from_map(
        p_map,
        scan_grid,
        n_peaks=n_peaks,
        metadata={"method": method},
        min_separation_deg=min_separation_deg,
    )


def srp_map_from_covariance(
    mic_cov: ArrayLike,
    freqs_hz: ArrayLike,
    geometry: ArrayGeometry,
    scan_grid: SphericalGrid,
    *,
    weighting: PhatMode = "phat",
    c: float = 343.0,
    freq_range_hz: tuple[float, float] | None = None,
    normalize: bool = True,
    n_peaks: int = 1,
    min_separation_deg: float | None = None,
) -> SpatialSpectrumResult:
    """SRP(-PHAT) map computed from per-bin microphone covariance matrices.

    Parameters
    ----------
    mic_cov : array_like, shape (F, M, M)
        Per-bin microphone-domain covariance or cross-spectrum matrices.
    freqs_hz : array_like, shape (F,)
        Frequency axis.
    geometry : ArrayGeometry
        Microphone geometry.
    scan_grid : SphericalGrid
        Scan directions (DOA convention).
    weighting : {"phat", "none"}, optional
        With ``"phat"`` every entry ``R[m,n]`` is divided by ``|R[m,n]|``
        before steering; otherwise raw covariance entries are used.
    c : float, optional
        Speed of sound in m/s.
    freq_range_hz : tuple[float, float] or None, optional
        Restrict integration to the given band (inclusive).
    normalize : bool, optional
        Peak-normalize the output.
    n_peaks : int, optional
        Number of peaks to report.

    Returns
    -------
    SpatialSpectrumResult
        SRP(-PHAT) map.
    """
    r = np.asarray(mic_cov, dtype=np.complex128)
    if r.ndim != 3 or r.shape[1] != r.shape[2]:
        raise ValueError("mic_cov must have shape (F, M, M)")
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if f.size != r.shape[0]:
        raise ValueError("freqs_hz length mismatch with mic_cov")

    mic_xyz = unit_sph_to_cart(
        geometry.sensor_grid.azimuth,
        geometry.sensor_grid.angle2,
        convention=geometry.sensor_grid.convention,
    ) * float(geometry.radius_m)
    if mic_xyz.shape[0] != r.shape[1]:
        raise ValueError("geometry microphone count does not match mic_cov")

    scan_u = unit_sph_to_cart(
        scan_grid.azimuth, scan_grid.angle2, convention=scan_grid.convention
    )
    proj = mic_xyz @ scan_u.T  # (M, G)

    if weighting == "phat":
        mag = np.abs(r)
        mag = np.where(mag > 1e-20, mag, 1.0)
        r_w = r / mag
    elif weighting == "none":
        r_w = r
    else:
        raise ValueError(f"weighting must be 'phat' or 'none', got {weighting!r}")

    mask = np.ones(f.size, dtype=bool)
    if freq_range_hz is not None:
        mask &= (f >= float(freq_range_hz[0])) & (f <= float(freq_range_hz[1]))
    mask &= f > 0.0
    if not np.any(mask):
        raise ValueError("no frequency bins active — check freq_range_hz")

    p_map = np.zeros(scan_u.shape[0], dtype=float)
    for fi in np.nonzero(mask)[0]:
        k = 2.0 * np.pi * f[fi] / c
        s = np.exp(1j * k * proj)  # (M, G)
        # R_fi is Hermitian; the real part of s^H R s is the beamformer power.
        p_map += np.real(np.einsum("mg,mn,ng->g", np.conj(s), r_w[fi], s))

    if normalize and np.max(p_map) > 0:
        p_map = p_map / np.max(p_map)

    method = "srp_phat" if weighting == "phat" else "srp"
    return spatial_spectrum_from_map(
        p_map,
        scan_grid,
        n_peaks=n_peaks,
        metadata={"method": method},
        min_separation_deg=min_separation_deg,
    )


__all__ = ["srp_map", "srp_map_from_covariance"]
