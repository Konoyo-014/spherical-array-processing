"""Acoustic metrics from room impulse responses (ISO 3382 family).

Given a monaural room impulse response ``h[n]`` at sample rate ``fs``,
this module derives the standard acoustic descriptors:

* **Energy decay curve** (EDC): reverse-cumulative integral of
  ``|h|²``, normalised so ``EDC[0] = 0 dB``.  The raw ingredient for
  every reverberation-time metric below.
* **T20 / T30 / T60** reverberation time via linear regression of the
  EDC over the ``[-5, -25]`` / ``[-5, -35]`` / ``[-5, -65]`` dB range
  (Schroeder method, ISO 3382-1 §A.2.2).  Default is T30 because it
  averages out more decay noise while staying within the
  instantaneous SNR of most measurements.
* **Early decay time** (EDT): slope of the first 10 dB of decay,
  extrapolated to 60 dB.
* **Clarity** C50 / C80: ratio of early energy (first 50 / 80 ms) to
  late energy, in dB.  C50 is the speech-intelligibility metric; C80
  is the classical music clarity descriptor.
* **Definition** D50: early energy (0–50 ms) / total energy, a
  linear-domain definition-of-speech descriptor.

All functions work on a monaural RIR.  For a multi-channel RIR (e.g.
an ambisonic shoebox response), apply them to the ``W`` / omni
channel or to each SH channel individually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


RtMethod = Literal["T20", "T30", "T60"]


def _validate_1d(rir: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(rir, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("rir must be non-empty")
    return arr


def energy_decay_curve(rir: ArrayLike) -> NDArray[np.float64]:
    """Schroeder-integrated EDC in dB, normalised so ``EDC[0] = 0``.

    Computes ``10·log10(Σ_{m≥n} h[m]² / Σ_m h[m]²)``.  The curve is
    monotonically decreasing and captures the instantaneous
    reverberation-energy budget remaining after sample ``n``.

    Parameters
    ----------
    rir : array_like, shape ``(T,)``
        Monaural room impulse response.

    Returns
    -------
    ndarray, shape ``(T,)``
        Energy decay curve in dB.  Values past the noise floor saturate
        to ``-∞`` (represented as a large negative number).
    """
    h = _validate_1d(rir)
    energy = h * h
    # Reverse cumulative integration.
    tail_energy = np.flip(np.cumsum(np.flip(energy)))
    total = tail_energy[0]
    if total <= 0.0:
        raise ValueError("rir has zero energy; cannot compute EDC")
    with np.errstate(divide="ignore"):
        edc_db = 10.0 * np.log10(np.maximum(tail_energy / total, 1e-300))
    return edc_db


def _rt_from_edc(
    edc_db: NDArray[np.float64],
    fs: float,
    low_db: float,
    high_db: float,
) -> float:
    """Linear-regression RT60 estimate between two EDC levels."""
    # First sample where the EDC crosses low_db / high_db, from the start.
    below_low = np.where(edc_db <= low_db)[0]
    below_high = np.where(edc_db <= high_db)[0]
    if below_low.size == 0 or below_high.size == 0:
        raise ValueError(
            f"EDC does not reach the required {high_db} dB decay; the "
            "RIR may be too short or too noisy"
        )
    start = int(below_low[0])
    stop = int(below_high[0])
    if stop <= start + 1:
        raise ValueError(
            "EDC decays too fast between the regression anchors; "
            f"start={start}, stop={stop}"
        )
    t = np.arange(start, stop + 1) / float(fs)
    y = edc_db[start : stop + 1]
    slope, _intercept = np.polyfit(t, y, 1)  # slope in dB/s, negative.
    if slope >= 0:
        raise ValueError(
            "EDC regression produced a non-negative slope; the RIR "
            "may be dominated by noise"
        )
    return -60.0 / slope


def reverberation_time(
    rir: ArrayLike, fs: float, *, method: RtMethod = "T30",
) -> float:
    """Reverberation time (RT60) via Schroeder-integrated regression.

    Parameters
    ----------
    rir : array_like
        Monaural room impulse response.
    fs : float
        Sampling rate in Hz.
    method : {"T20", "T30", "T60"}
        Regression window.  ``T20`` fits ``[-5, -25]`` dB, ``T30``
        fits ``[-5, -35]`` dB, ``T60`` fits the full ``[-5, -65]`` dB
        range.  Default ``"T30"`` is ISO 3382's recommended compromise
        between noise rejection and decay-region coverage.

    Returns
    -------
    float
        Estimated RT60 in seconds.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    edc_db = energy_decay_curve(rir)
    if method == "T20":
        return _rt_from_edc(edc_db, fs, -5.0, -25.0)
    if method == "T30":
        return _rt_from_edc(edc_db, fs, -5.0, -35.0)
    if method == "T60":
        return _rt_from_edc(edc_db, fs, -5.0, -65.0)
    raise ValueError(f"method must be 'T20', 'T30', or 'T60', got {method!r}")


def early_decay_time(rir: ArrayLike, fs: float) -> float:
    """Early decay time: first 10 dB of EDC decay, extrapolated to 60 dB."""
    if fs <= 0:
        raise ValueError("fs must be positive")
    edc_db = energy_decay_curve(rir)
    # EDT fits [0, -10] dB.  Use ``-0.1`` as a tiny offset from the
    # exactly-0 dB start so numerics are well conditioned.
    return _rt_from_edc(edc_db, fs, -0.1, -10.0)


def clarity(
    rir: ArrayLike, fs: float, *, time_ms: float = 50.0,
) -> float:
    """Clarity index ``C_τ`` = 10·log10(early / late energy), in dB.

    ``time_ms = 50`` gives **C50** (speech clarity), ``time_ms = 80``
    gives **C80** (music clarity).  Defined per ISO 3382-1 §A.1.4.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    h = _validate_1d(rir)
    split = int(round(float(time_ms) * 1e-3 * float(fs)))
    if split <= 0 or split >= h.size:
        raise ValueError(
            f"split index {split} (time={time_ms} ms) falls outside "
            f"the RIR of length {h.size}"
        )
    early = float(np.sum(h[:split] ** 2))
    late = float(np.sum(h[split:] ** 2))
    if late <= 0.0:
        return float("inf")
    if early <= 0.0:
        return -float("inf")
    return 10.0 * np.log10(early / late)


def definition(
    rir: ArrayLike, fs: float, *, time_ms: float = 50.0,
) -> float:
    """Definition ``D_τ`` = early / total energy (linear, in [0, 1]).

    ``time_ms = 50`` gives the standard D50 descriptor.  Unlike
    :func:`clarity`, the ratio is in **linear scale**, not dB.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    h = _validate_1d(rir)
    split = int(round(float(time_ms) * 1e-3 * float(fs)))
    if split <= 0 or split >= h.size:
        raise ValueError(
            f"split index {split} (time={time_ms} ms) falls outside "
            f"the RIR of length {h.size}"
        )
    early = float(np.sum(h[:split] ** 2))
    total = float(np.sum(h ** 2))
    if total <= 0.0:
        raise ValueError("rir has zero energy")
    return early / total


def early_late_energy(
    rir: ArrayLike,
    fs: float,
    *,
    time_ms: float = 80.0,
) -> tuple[float, float]:
    """Return early and late RIR energies split at ``time_ms``."""

    if fs <= 0:
        raise ValueError("fs must be positive")
    h = _validate_1d(rir)
    split = int(round(float(time_ms) * 1e-3 * float(fs)))
    if split <= 0 or split >= h.size:
        raise ValueError(
            f"split index {split} (time={time_ms} ms) falls outside "
            f"the RIR of length {h.size}"
        )
    return float(np.sum(h[:split] ** 2)), float(np.sum(h[split:] ** 2))


def total_energy(rir: ArrayLike) -> float:
    """Total squared RIR energy."""

    h = _validate_1d(rir)
    return float(np.sum(h * h))


def direct_to_reverberant_ratio(
    rir: ArrayLike,
    fs: float,
    *,
    direct_window_ms: float = 2.5,
) -> float:
    """Direct-to-reverberant energy ratio in dB.

    The direct component is approximated by a short window starting at
    the largest absolute sample.  This is a practical diagnostic, not a
    replacement for a full source/receiver-calibrated measurement.
    """

    if fs <= 0:
        raise ValueError("fs must be positive")
    h = _validate_1d(rir)
    peak = int(np.argmax(np.abs(h)))
    n_win = max(1, int(round(float(direct_window_ms) * 1e-3 * float(fs))))
    stop = min(h.size, peak + n_win)
    direct = float(np.sum(h[peak:stop] ** 2))
    reverberant = float(np.sum(h ** 2) - direct)
    if reverberant <= 0.0:
        return float("inf")
    if direct <= 0.0:
        return -float("inf")
    return 10.0 * np.log10(direct / reverberant)


def center_time(rir: ArrayLike, fs: float) -> float:
    """Centre time ``Ts`` in seconds: energy-weighted mean arrival time."""

    if fs <= 0:
        raise ValueError("fs must be positive")
    h = _validate_1d(rir)
    energy = h * h
    total = float(np.sum(energy))
    if total <= 0.0:
        raise ValueError("rir has zero energy")
    t = np.arange(h.size, dtype=float) / float(fs)
    return float(np.sum(t * energy) / total)


def strength(
    rir: ArrayLike,
    *,
    reference_rir: ArrayLike | None = None,
    reference_energy: float | None = None,
) -> float:
    """Sound strength ``G`` in dB from RIR energy and a reference energy."""

    e = total_energy(rir)
    if reference_energy is not None:
        ref = float(reference_energy)
    elif reference_rir is not None:
        ref = total_energy(reference_rir)
    else:
        ref = 1.0
    if ref <= 0.0:
        raise ValueError("reference energy must be positive")
    if e <= 0.0:
        return -float("inf")
    return 10.0 * np.log10(e / ref)


def reverberation_times(
    rir: ArrayLike,
    fs: float,
    *,
    methods: tuple[RtMethod, ...] = ("T20", "T30"),
) -> dict[str, float]:
    """Compute multiple reverberation-time estimates from one RIR."""

    return {method: reverberation_time(rir, fs, method=method) for method in methods}


def banded_rir_metrics(
    rirs: ArrayLike,
    fs: float,
    *,
    rt_method: RtMethod = "T30",
    axis: int = -1,
) -> list[RIRMetrics]:
    """Compute :func:`rir_metrics` over all non-time axes of a RIR stack."""

    arr = np.asarray(rirs, dtype=float)
    time_axis = int(axis) % arr.ndim
    moved = np.moveaxis(arr, time_axis, -1)
    flat = moved.reshape((-1, moved.shape[-1]))
    return [rir_metrics(row, fs, rt_method=rt_method) for row in flat]


def interaural_cross_correlation(
    left_rir: ArrayLike,
    right_rir: ArrayLike,
    fs: float,
    *,
    window_ms: float = 80.0,
    max_lag_ms: float = 1.0,
) -> float:
    """Maximum absolute interaural cross-correlation over a lag window."""

    if fs <= 0:
        raise ValueError("fs must be positive")
    left = _validate_1d(left_rir)
    right = _validate_1d(right_rir)
    if left.size != right.size:
        raise ValueError("left_rir and right_rir must have the same length")
    n_win = min(left.size, max(1, int(round(float(window_ms) * 1e-3 * float(fs)))))
    max_lag = int(round(float(max_lag_ms) * 1e-3 * float(fs)))
    if max_lag < 0:
        raise ValueError("max_lag_ms must be non-negative")
    l = left[:n_win]
    r = right[:n_win]
    denom = np.sqrt(float(np.sum(l * l) * np.sum(r * r)))
    if denom <= 0.0:
        raise ValueError("IACC window has zero energy")
    values = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            values.append(float(np.sum(l[-lag:] * r[: n_win + lag]) / denom))
        elif lag > 0:
            values.append(float(np.sum(l[: n_win - lag] * r[lag:]) / denom))
        else:
            values.append(float(np.sum(l * r) / denom))
    return float(np.max(np.abs(values)))


def lateral_energy_fraction(
    lateral_rir: ArrayLike,
    omni_rir: ArrayLike,
    fs: float,
    *,
    early_start_ms: float = 5.0,
    early_end_ms: float = 80.0,
) -> float:
    """Early lateral energy fraction using lateral and omni RIR channels."""

    if fs <= 0:
        raise ValueError("fs must be positive")
    lat = _validate_1d(lateral_rir)
    omni = _validate_1d(omni_rir)
    if lat.size != omni.size:
        raise ValueError("lateral_rir and omni_rir must have the same length")
    start = int(round(float(early_start_ms) * 1e-3 * float(fs)))
    stop = int(round(float(early_end_ms) * 1e-3 * float(fs)))
    if start < 0 or stop <= start or stop > lat.size:
        raise ValueError("early window falls outside the RIR")
    numerator = float(np.sum(lat[start:stop] ** 2))
    denominator = float(np.sum(omni[:stop] ** 2))
    if denominator <= 0.0:
        raise ValueError("omni_rir early energy is zero")
    return numerator / denominator


@dataclass(frozen=True)
class RIRMetrics:
    """Standard acoustic metrics bundled in one object.

    Attributes
    ----------
    edc_db : ndarray
        Energy decay curve in dB (Schroeder-integrated).
    rt60_s : float
        Reverberation time via T30 regression (seconds).
    edt_s : float
        Early decay time (seconds).
    c50_db : float
        Clarity C50 (dB).
    c80_db : float
        Clarity C80 (dB).
    d50 : float
        Definition D50 (linear, in [0, 1]).
    ts_s : float
        Centre time in seconds.
    g_db : float
        Strength in dB relative to unit reference energy.
    drr_db : float
        Direct-to-reverberant ratio in dB using a short peak-aligned
        direct window.
    """

    edc_db: NDArray[np.float64]
    rt60_s: float
    edt_s: float
    c50_db: float
    c80_db: float
    d50: float
    ts_s: float
    g_db: float
    drr_db: float


def rir_metrics(
    rir: ArrayLike, fs: float, *, rt_method: RtMethod = "T30",
) -> RIRMetrics:
    """Compute a bundle of standard acoustic metrics from a RIR.

    Calls :func:`energy_decay_curve`, :func:`reverberation_time`,
    :func:`early_decay_time`, :func:`clarity` (C50, C80), and
    :func:`definition` in one shot.
    """
    edc = energy_decay_curve(rir)
    return RIRMetrics(
        edc_db=edc,
        rt60_s=reverberation_time(rir, fs, method=rt_method),
        edt_s=early_decay_time(rir, fs),
        c50_db=clarity(rir, fs, time_ms=50.0),
        c80_db=clarity(rir, fs, time_ms=80.0),
        d50=definition(rir, fs, time_ms=50.0),
        ts_s=center_time(rir, fs),
        g_db=strength(rir),
        drr_db=direct_to_reverberant_ratio(rir, fs),
    )


__all__ = [
    "RIRMetrics",
    "banded_rir_metrics",
    "center_time",
    "clarity",
    "definition",
    "direct_to_reverberant_ratio",
    "early_decay_time",
    "early_late_energy",
    "energy_decay_curve",
    "interaural_cross_correlation",
    "lateral_energy_fraction",
    "reverberation_time",
    "reverberation_times",
    "rir_metrics",
    "strength",
    "total_energy",
]
