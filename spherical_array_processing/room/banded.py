"""Frequency-dependent shoebox image-source RIR.

Extends :func:`shoebox_rir` with **per-octave-band wall absorption**:
each wall gets its own reflection magnitude per frequency band, so the
RIR's spectral shape evolves with reflection order.  This is the
standard modelling refinement on top of the scalar-reflection
Allen–Berkley method for matching a target absorption curve.

The implementation builds one short FIR filter per distinct bounce
count vector (grouping identical images together to keep the cost
manageable), convolves each image-source impulse with its FIR, and
sums the result into the output buffer at the appropriate sample
delay.  The FIR is built via :func:`scipy.signal.firwin2` from the
band edges and effective per-band gain.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import firwin2

from .shoebox import ShoeboxRoom, _image_source_grid


def _validate_reflection_bands(
    reflection_bands: ArrayLike, n_walls: int = 6,
) -> NDArray[np.float64]:
    arr = np.asarray(reflection_bands, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != n_walls:
        raise ValueError(
            f"reflection_bands must have shape ({n_walls}, n_bands); "
            f"got {arr.shape}"
        )
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(
            "reflection_bands entries must lie in [0, 1]"
        )
    return arr


def _validate_band_edges(
    band_edges_hz: ArrayLike, n_bands: int, fs: float,
) -> NDArray[np.float64]:
    arr = np.asarray(band_edges_hz, dtype=float).reshape(-1)
    if arr.size != n_bands + 1:
        raise ValueError(
            f"band_edges_hz must have n_bands+1 = {n_bands + 1} entries; "
            f"got {arr.size}"
        )
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("band_edges_hz must be strictly increasing")
    if arr[0] < 0.0:
        raise ValueError("band_edges_hz must start at 0 Hz or above")
    if arr[-1] > fs / 2.0 + 1e-9:
        raise ValueError(
            f"last band edge {arr[-1]} Hz exceeds Nyquist ({fs / 2.0})"
        )
    return arr


def _fir_from_bands(
    band_gains: NDArray[np.float64],
    band_edges_hz: NDArray[np.float64],
    fs: float,
    fir_taps: int,
) -> NDArray[np.float64]:
    """Build a linear-phase FIR with piecewise-constant band gains.

    Uses :func:`scipy.signal.firwin2` with a frequency response that
    is constant inside each band.  Interior band edges are duplicated
    so :func:`firwin2` realises the intended step discontinuities
    rather than ramping across the whole next band.
    """
    nyq = fs / 2.0
    # Build the piecewise-constant target response expected by firwin2.
    # Repeating an interior edge once with the gain on each side creates
    # a discontinuity at that edge instead of a linear ramp over the
    # following band.
    freqs = [0.0]
    gains = [float(band_gains[0])]
    for b in range(len(band_gains) - 1):
        edge = float(band_edges_hz[b + 1])
        freqs.extend([edge, edge])
        gains.extend([
            float(band_gains[b]),
            float(band_gains[b + 1]),
        ])

    last_edge = float(band_edges_hz[-1])
    if last_edge < nyq:
        freqs.append(last_edge)
        gains.append(float(band_gains[-1]))
        freqs.append(nyq)
        gains.append(float(band_gains[-1]))
    else:
        freqs.append(last_edge)
        gains.append(float(band_gains[-1]))

    freqs_arr = np.asarray(freqs, dtype=float) / nyq
    gains_arr = np.asarray(gains, dtype=float)
    # firwin2 requires numtaps odd when nyq gain ≠ 0 (the usual case).
    taps = int(fir_taps)
    if taps < 2:
        raise ValueError("fir_taps must be >= 2")
    if taps % 2 == 0:
        taps += 1
    fir = firwin2(taps, freqs_arr, gains_arr)
    return np.asarray(fir, dtype=np.float64)


def shoebox_rir_banded(
    room_dimensions_m: tuple[float, float, float],
    source_position_m: ArrayLike,
    listener_position_m: ArrayLike,
    reflection_bands: ArrayLike,
    band_edges_hz: ArrayLike,
    *,
    fs: float,
    ir_length: int,
    max_reflection_order: int = 16,
    c: float = 343.0,
    fir_taps: int = 129,
) -> NDArray[np.float64]:
    """Shoebox RIR with per-octave-band wall absorption.

    Parameters
    ----------
    room_dimensions_m : tuple
        Room extents ``(Lx, Ly, Lz)`` in metres.
    source_position_m, listener_position_m : array_like, shape (3,)
    reflection_bands : array_like, shape ``(6, B)``
        Per-wall reflection **magnitudes** (in ``[0, 1]``) for each
        of ``B`` frequency bands, in wall order
        ``(−x, +x, −y, +y, −z, +z)``.  Relate to absorption ``α`` via
        ``|β| = √(1 − α)``.
    band_edges_hz : array_like, shape ``(B + 1,)``
        Strictly-increasing band edges in Hz, starting at ``0`` and
        ending at or below Nyquist.  Typical choice: octave edges at
        ``[0, 88, 177, 355, 710, 1420, 2840, 5680, 11360]`` Hz for the
        standard 125 Hz–8 kHz centres.
    fs : float
        Sampling rate in Hz.
    ir_length : int
        Output IR length in samples.
    max_reflection_order : int, optional
        Cap on the per-axis bounce count.  Default ``16``.
    c : float, optional
        Speed of sound.  Default ``343`` m/s.
    fir_taps : int, optional
        Length of the per-image band-gain FIR.  Default ``129``
        (``≈ 2.7`` ms at ``48`` kHz — a reasonable trade-off between
        spectral resolution and pre-ringing).  The FIR is centred on
        the image delay, so each image produces symmetric ringing up to
        ``fir_taps//2`` samples before and after the nominal arrival.
        Keep ``fir_taps`` modest when onset timing of early reflections
        matters.

    Returns
    -------
    ndarray, shape ``(ir_length,)``
        Monaural banded-absorption RIR.
    """
    room_lengths = np.asarray(room_dimensions_m, dtype=float)
    if room_lengths.shape != (3,) or np.any(room_lengths <= 0.0):
        raise ValueError("room_dimensions_m must be 3 positive floats")
    src = np.asarray(source_position_m, dtype=float)
    lis = np.asarray(listener_position_m, dtype=float)
    for name, pos in (("source", src), ("listener", lis)):
        if pos.shape != (3,):
            raise ValueError(f"{name}_position_m must have shape (3,)")
        if np.any(pos <= 0.0) or np.any(pos >= room_lengths):
            raise ValueError(
                f"{name}_position_m must lie strictly inside the room"
            )

    fs_f = float(fs)
    if fs_f <= 0.0:
        raise ValueError("fs must be positive")
    n_samples = int(ir_length)
    if n_samples <= 0:
        raise ValueError("ir_length must be positive")
    c_f = float(c)
    if c_f <= 0.0:
        raise ValueError("c must be positive")
    max_order = int(max_reflection_order)
    if max_order < 0:
        raise ValueError("max_reflection_order must be non-negative")

    refl = _validate_reflection_bands(reflection_bands)
    n_bands = refl.shape[1]
    edges = _validate_band_edges(band_edges_hz, n_bands, fs_f)

    positions, bounces = _image_source_grid(src, room_lengths, max_order)
    displacement = positions - lis[None, :]
    distances = np.linalg.norm(displacement, axis=1)
    safe_dist = np.maximum(distances, 1e-9)
    delays_s = distances / c_f
    # Per-image reference amplitude (geometric spreading only — the
    # band-gain FIR handles the wall losses).
    geom_amp = 1.0 / safe_dist
    sample_idx = np.round(delays_s * fs_f).astype(np.int64)
    mask = (sample_idx >= 0) & (sample_idx < n_samples)
    kept_bounces = bounces[mask]       # (K, 6)
    kept_amp = geom_amp[mask]           # (K,)
    kept_idx = sample_idx[mask]         # (K,)

    out = np.zeros(n_samples, dtype=np.float64)
    if kept_bounces.size == 0:
        return out

    # Group images by identical bounce-count vector — FIR design is the
    # expensive step, so amortise it across images with equal walls.
    unique_bounces, inverse = np.unique(
        kept_bounces, axis=0, return_inverse=True,
    )
    half_taps = fir_taps // 2
    for g, bvec in enumerate(unique_bounces):
        # Per-band effective gain.
        band_gains = np.prod(refl ** bvec[:, None], axis=0)  # (B,)
        if np.all(band_gains == 0.0):
            continue
        fir = _fir_from_bands(band_gains, edges, fs_f, fir_taps)
        idxs = np.where(inverse == g)[0]
        for i in idxs:
            centre = int(kept_idx[i])
            amp = float(kept_amp[i])
            start = centre - half_taps
            stop = start + fir.size
            lo = max(0, start)
            hi = min(n_samples, stop)
            fir_lo = lo - start
            fir_hi = fir_lo + (hi - lo)
            if hi > lo:
                out[lo:hi] += amp * fir[fir_lo:fir_hi]
    return out


__all__ = ["shoebox_rir_banded"]
