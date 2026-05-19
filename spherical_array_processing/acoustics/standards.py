"""Classical acoustics utilities for levels, bands, weighting, and media.

This module collects small, stable formulas that appear throughout
room-acoustics, electroacoustics, and spatial-audio measurements.  The
functions deliberately avoid claiming instrument compliance: IEC/ISO
standards define complete measurement procedures, tolerances, filters,
and reporting rules.  The helpers here implement the mathematical
building blocks that users need before and after those measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FrequencyWeighting = Literal["A", "C", "Z", "a", "c", "z"]

REFERENCE_PRESSURE_PA = 20e-6
REFERENCE_SOUND_INTENSITY_W_M2 = 1e-12
REFERENCE_SOUND_POWER_W = 1e-12


@dataclass(frozen=True)
class FractionalOctaveBand:
    """Fractional-octave band metadata.

    Attributes
    ----------
    center_hz
        Exact centre frequency in hertz.
    lower_hz, upper_hz
        Exact lower and upper band-edge frequencies in hertz.
    fraction
        Number of bands per octave.  ``1`` means octave bands and ``3``
        means one-third-octave bands.
    nominal_hz
        Rounded display value.  It is intentionally separate from the
        exact centre frequency used for computation.
    """

    center_hz: float
    lower_hz: float
    upper_hz: float
    fraction: int
    nominal_hz: float


def _arr(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=float)


def _maybe_scalar(x: ArrayLike) -> float | NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


def _positive(value: float, name: str) -> float:
    val = float(value)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be positive")
    return val


def amplitude_to_db(
    amplitude: ArrayLike,
    *,
    reference: float = 1.0,
    floor: float | None = None,
) -> float | NDArray[np.float64]:
    """Convert an amplitude ratio to decibels with ``20 log10``."""

    ref = _positive(reference, "reference")
    amp = np.abs(_arr(amplitude))
    if floor is not None:
        amp = np.maximum(amp, float(floor))
    with np.errstate(divide="ignore"):
        out = 20.0 * np.log10(amp / ref)
    return _maybe_scalar(out)


def db_to_amplitude(level_db: ArrayLike, *, reference: float = 1.0) -> float | NDArray[np.float64]:
    """Convert decibels back to a linear amplitude."""

    ref = _positive(reference, "reference")
    return _maybe_scalar(ref * np.power(10.0, _arr(level_db) / 20.0))


def power_to_db(
    power: ArrayLike,
    *,
    reference: float = 1.0,
    floor: float | None = None,
) -> float | NDArray[np.float64]:
    """Convert a power or energy ratio to decibels with ``10 log10``."""

    ref = _positive(reference, "reference")
    p = _arr(power)
    if np.any(p < 0.0):
        raise ValueError("power must be non-negative")
    if floor is not None:
        p = np.maximum(p, float(floor))
    with np.errstate(divide="ignore"):
        out = 10.0 * np.log10(p / ref)
    return _maybe_scalar(out)


def db_to_power(level_db: ArrayLike, *, reference: float = 1.0) -> float | NDArray[np.float64]:
    """Convert decibels back to linear power or energy."""

    ref = _positive(reference, "reference")
    return _maybe_scalar(ref * np.power(10.0, _arr(level_db) / 10.0))


def db_from_amplitude_ratio(ratio: ArrayLike, *, floor: float | None = None) -> float | NDArray[np.float64]:
    """Decibels from a linear amplitude ratio."""

    return amplitude_to_db(ratio, reference=1.0, floor=floor)


def amplitude_ratio_from_db(level_db: ArrayLike) -> float | NDArray[np.float64]:
    """Linear amplitude ratio from decibels."""

    return db_to_amplitude(level_db, reference=1.0)


def db_from_power_ratio(ratio: ArrayLike, *, floor: float | None = None) -> float | NDArray[np.float64]:
    """Decibels from a linear power ratio."""

    return power_to_db(ratio, reference=1.0, floor=floor)


def power_ratio_from_db(level_db: ArrayLike) -> float | NDArray[np.float64]:
    """Linear power ratio from decibels."""

    return db_to_power(level_db, reference=1.0)


def level_difference_db(reference_level_db: ArrayLike, comparison_level_db: ArrayLike) -> float | NDArray[np.float64]:
    """Difference ``comparison - reference`` between two level values."""

    return _maybe_scalar(_arr(comparison_level_db) - _arr(reference_level_db))


def pressure_ratio_from_spl_difference(delta_db: ArrayLike) -> float | NDArray[np.float64]:
    """Pressure ratio implied by an SPL difference."""

    return amplitude_ratio_from_db(delta_db)


def intensity_ratio_from_level_difference(delta_db: ArrayLike) -> float | NDArray[np.float64]:
    """Intensity or power ratio implied by a level difference."""

    return power_ratio_from_db(delta_db)


def level_change_for_distance_ratio(distance_ratio: ArrayLike) -> float | NDArray[np.float64]:
    """Free-field SPL change for a source-observer distance ratio."""

    ratio = _arr(distance_ratio)
    if np.any(ratio <= 0.0):
        raise ValueError("distance_ratio must be positive")
    return _maybe_scalar(-20.0 * np.log10(ratio))


def distance_ratio_for_level_change(delta_db: ArrayLike) -> float | NDArray[np.float64]:
    """Distance ratio that produces a free-field SPL change."""

    return _maybe_scalar(np.power(10.0, -_arr(delta_db) / 20.0))


def intensity_to_sound_intensity_level(
    intensity_w_m2: ArrayLike,
    *,
    reference_w_m2: float = REFERENCE_SOUND_INTENSITY_W_M2,
    floor_w_m2: float | None = None,
) -> float | NDArray[np.float64]:
    """Convert sound intensity to dB re 1 pW/m²."""

    return power_to_db(intensity_w_m2, reference=reference_w_m2, floor=floor_w_m2)


def sound_intensity_level_to_intensity(
    level_db: ArrayLike,
    *,
    reference_w_m2: float = REFERENCE_SOUND_INTENSITY_W_M2,
) -> float | NDArray[np.float64]:
    """Convert sound-intensity level to W/m²."""

    return db_to_power(level_db, reference=reference_w_m2)


def sound_power_to_sound_power_level(
    power_w: ArrayLike,
    *,
    reference_w: float = REFERENCE_SOUND_POWER_W,
    floor_w: float | None = None,
) -> float | NDArray[np.float64]:
    """Convert acoustic power to sound-power level in dB."""

    return power_to_db(power_w, reference=reference_w, floor=floor_w)


def sound_power_level_to_power(
    level_db: ArrayLike,
    *,
    reference_w: float = REFERENCE_SOUND_POWER_W,
) -> float | NDArray[np.float64]:
    """Convert sound-power level to acoustic power in watts."""

    return db_to_power(level_db, reference=reference_w)


def plane_wave_intensity_from_pressure(
    pressure_rms_pa: ArrayLike,
    *,
    impedance_pa_s_m: float = 413.0,
) -> float | NDArray[np.float64]:
    """Plane-wave active intensity from RMS pressure."""

    z0 = _positive(impedance_pa_s_m, "impedance_pa_s_m")
    p = _arr(pressure_rms_pa)
    return _maybe_scalar((p * p) / z0)


def plane_wave_pressure_from_intensity(
    intensity_w_m2: ArrayLike,
    *,
    impedance_pa_s_m: float = 413.0,
) -> float | NDArray[np.float64]:
    """RMS pressure of a plane wave from active intensity."""

    z0 = _positive(impedance_pa_s_m, "impedance_pa_s_m")
    i = _arr(intensity_w_m2)
    if np.any(i < 0.0):
        raise ValueError("intensity_w_m2 must be non-negative")
    return _maybe_scalar(np.sqrt(i * z0))


def particle_velocity_from_pressure(
    pressure_pa: ArrayLike,
    *,
    impedance_pa_s_m: float = 413.0,
) -> float | NDArray[np.float64]:
    """Plane-wave particle velocity from pressure."""

    return _maybe_scalar(_arr(pressure_pa) / _positive(impedance_pa_s_m, "impedance_pa_s_m"))


def pressure_from_particle_velocity(
    velocity_m_s: ArrayLike,
    *,
    impedance_pa_s_m: float = 413.0,
) -> float | NDArray[np.float64]:
    """Plane-wave pressure from particle velocity."""

    return _maybe_scalar(_arr(velocity_m_s) * _positive(impedance_pa_s_m, "impedance_pa_s_m"))


def wavelength(frequency_hz: ArrayLike, *, c: float = 343.0) -> float | NDArray[np.float64]:
    """Acoustic wavelength for a frequency."""

    f = _arr(frequency_hz)
    if np.any(f <= 0.0):
        raise ValueError("frequency_hz must be positive")
    return _maybe_scalar(_positive(c, "c") / f)


def frequency_from_wavelength(wavelength_m: ArrayLike, *, c: float = 343.0) -> float | NDArray[np.float64]:
    """Frequency from acoustic wavelength."""

    lam = _arr(wavelength_m)
    if np.any(lam <= 0.0):
        raise ValueError("wavelength_m must be positive")
    return _maybe_scalar(_positive(c, "c") / lam)


def period(frequency_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Signal period in seconds."""

    f = _arr(frequency_hz)
    if np.any(f <= 0.0):
        raise ValueError("frequency_hz must be positive")
    return _maybe_scalar(1.0 / f)


def frequency_from_period(period_s: ArrayLike) -> float | NDArray[np.float64]:
    """Frequency from signal period."""

    t = _arr(period_s)
    if np.any(t <= 0.0):
        raise ValueError("period_s must be positive")
    return _maybe_scalar(1.0 / t)


def angular_frequency(frequency_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Angular frequency in rad/s."""

    f = _arr(frequency_hz)
    if np.any(f < 0.0):
        raise ValueError("frequency_hz must be non-negative")
    return _maybe_scalar(2.0 * np.pi * f)


def frequency_from_angular_frequency(omega_rad_s: ArrayLike) -> float | NDArray[np.float64]:
    """Frequency in hertz from angular frequency."""

    omega = _arr(omega_rad_s)
    if np.any(omega < 0.0):
        raise ValueError("omega_rad_s must be non-negative")
    return _maybe_scalar(omega / (2.0 * np.pi))


def phase_shift_for_delay(frequency_hz: ArrayLike, delay_s: ArrayLike) -> float | NDArray[np.float64]:
    """Phase shift in radians for a time delay."""

    return _maybe_scalar(-2.0 * np.pi * _arr(frequency_hz) * _arr(delay_s))


def phase_degrees_to_radians(phase_deg: ArrayLike) -> float | NDArray[np.float64]:
    """Phase angle in degrees to radians."""

    return _maybe_scalar(np.deg2rad(_arr(phase_deg)))


def phase_radians_to_degrees(phase_rad: ArrayLike) -> float | NDArray[np.float64]:
    """Phase angle in radians to degrees."""

    return _maybe_scalar(np.rad2deg(_arr(phase_rad)))


def wrap_phase_rad(phase_rad: ArrayLike) -> float | NDArray[np.float64]:
    """Wrap phase to ``[-π, π)``."""

    return _maybe_scalar((_arr(phase_rad) + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_phase_deg(phase_deg: ArrayLike) -> float | NDArray[np.float64]:
    """Wrap phase to ``[-180, 180)`` degrees."""

    return _maybe_scalar((_arr(phase_deg) + 180.0) % 360.0 - 180.0)


def time_delay_for_phase_shift(frequency_hz: ArrayLike, phase_rad: ArrayLike) -> float | NDArray[np.float64]:
    """Time delay represented by a phase shift."""

    f = _arr(frequency_hz)
    if np.any(f <= 0.0):
        raise ValueError("frequency_hz must be positive")
    return _maybe_scalar(-_arr(phase_rad) / (2.0 * np.pi * f))


def distance_delay(distance_m: ArrayLike, *, c: float = 343.0) -> float | NDArray[np.float64]:
    """Propagation delay for an acoustic path length."""

    d = _arr(distance_m)
    if np.any(d < 0.0):
        raise ValueError("distance_m must be non-negative")
    return _maybe_scalar(d / _positive(c, "c"))


def sample_delay_from_distance(
    distance_m: ArrayLike,
    fs: float,
    *,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Propagation delay in samples for an acoustic path length."""

    return _maybe_scalar(_arr(distance_delay(distance_m, c=c)) * _positive(fs, "fs"))


def distance_from_sample_delay(
    delay_samples: ArrayLike,
    fs: float,
    *,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Acoustic path length from a sample delay."""

    samples = _arr(delay_samples)
    if np.any(samples < 0.0):
        raise ValueError("delay_samples must be non-negative")
    return _maybe_scalar(samples * _positive(c, "c") / _positive(fs, "fs"))


def rms_to_peak_sine(rms: ArrayLike) -> float | NDArray[np.float64]:
    """Peak amplitude of a sinusoid from RMS amplitude."""

    return _maybe_scalar(_arr(rms) * np.sqrt(2.0))


def peak_to_rms_sine(peak: ArrayLike) -> float | NDArray[np.float64]:
    """RMS amplitude of a sinusoid from peak amplitude."""

    return _maybe_scalar(_arr(peak) / np.sqrt(2.0))


def rms_to_peak_to_peak_sine(rms: ArrayLike) -> float | NDArray[np.float64]:
    """Peak-to-peak amplitude of a sinusoid from RMS amplitude."""

    return _maybe_scalar(2.0 * _arr(rms_to_peak_sine(rms)))


def peak_to_peak_to_rms_sine(peak_to_peak: ArrayLike) -> float | NDArray[np.float64]:
    """RMS amplitude of a sinusoid from peak-to-peak amplitude."""

    return _maybe_scalar(_arr(peak_to_peak) / (2.0 * np.sqrt(2.0)))


def crest_factor(signal: ArrayLike, *, axis: int | None = None, eps: float = 1e-30) -> float | NDArray[np.float64]:
    """Peak-to-RMS crest factor."""

    x = np.asarray(signal)
    peak = np.max(np.abs(x), axis=axis)
    rms = np.sqrt(np.mean(np.abs(x) ** 2, axis=axis))
    return _maybe_scalar(peak / np.maximum(rms, float(eps)))


def crest_factor_db(signal: ArrayLike, *, axis: int | None = None, eps: float = 1e-30) -> float | NDArray[np.float64]:
    """Peak-to-RMS crest factor in decibels."""

    return amplitude_to_db(crest_factor(signal, axis=axis, eps=eps))


def pressure_rms(pressure_pa: ArrayLike, axis: int | None = None) -> float | NDArray[np.float64]:
    """Root-mean-square sound pressure in pascals."""

    p = _arr(pressure_pa)
    return _maybe_scalar(np.sqrt(np.mean(p * p, axis=axis)))


def pressure_to_spl(
    pressure_pa: ArrayLike,
    *,
    reference_pa: float = REFERENCE_PRESSURE_PA,
    floor_pa: float | None = None,
) -> float | NDArray[np.float64]:
    """Convert RMS pressure to sound-pressure level in dB SPL."""

    return amplitude_to_db(pressure_pa, reference=reference_pa, floor=floor_pa)


def spl_to_pressure(
    spl_db: ArrayLike,
    *,
    reference_pa: float = REFERENCE_PRESSURE_PA,
) -> float | NDArray[np.float64]:
    """Convert dB SPL to RMS pressure in pascals."""

    return db_to_amplitude(spl_db, reference=reference_pa)


def leq(
    pressure_pa: ArrayLike,
    *,
    reference_pa: float = REFERENCE_PRESSURE_PA,
    axis: int | None = None,
) -> float | NDArray[np.float64]:
    """Equivalent continuous sound-pressure level from pressure samples."""

    ref = _positive(reference_pa, "reference_pa")
    p = _arr(pressure_pa)
    mean_square = np.mean(p * p, axis=axis)
    return power_to_db(mean_square, reference=ref * ref)


def sound_exposure(
    pressure_pa: ArrayLike,
    fs: float,
    *,
    axis: int = -1,
) -> float | NDArray[np.float64]:
    """Sound exposure ``E = integral p(t)^2 dt`` in Pa^2 s."""

    sample_rate = _positive(fs, "fs")
    p = _arr(pressure_pa)
    return _maybe_scalar(np.sum(p * p, axis=axis) / sample_rate)


def sound_exposure_level(
    pressure_pa: ArrayLike,
    fs: float,
    *,
    reference_pa: float = REFERENCE_PRESSURE_PA,
    axis: int = -1,
) -> float | NDArray[np.float64]:
    """Sound exposure level in dB re ``p0^2 s``."""

    ref = _positive(reference_pa, "reference_pa")
    exposure = np.asarray(sound_exposure(pressure_pa, fs, axis=axis), dtype=float)
    return power_to_db(exposure, reference=ref * ref)


def energetic_sum_db(levels_db: ArrayLike, axis: int | None = None) -> float | NDArray[np.float64]:
    """Energetic sum of independent levels in decibels."""

    levels = _arr(levels_db)
    energy = np.sum(np.power(10.0, levels / 10.0), axis=axis)
    return power_to_db(energy)


def energetic_mean_db(
    levels_db: ArrayLike,
    *,
    axis: int | None = None,
    weights: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    """Energetic mean of decibel levels."""

    levels = _arr(levels_db)
    lin = np.power(10.0, levels / 10.0)
    if weights is None:
        mean = np.mean(lin, axis=axis)
    else:
        w = _arr(weights)
        mean = np.average(lin, axis=axis, weights=w)
    return power_to_db(mean)


def equivalent_continuous_level(
    levels_db: ArrayLike,
    durations_s: ArrayLike | None = None,
) -> float:
    """Time-weighted equivalent level from segment levels."""

    levels = _arr(levels_db).reshape(-1)
    if levels.size == 0:
        raise ValueError("levels_db must be non-empty")
    if durations_s is None:
        weights = np.ones_like(levels)
    else:
        weights = _arr(durations_s).reshape(-1)
        if weights.shape != levels.shape:
            raise ValueError("durations_s must match levels_db")
        if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("durations_s must be non-negative with positive sum")
    lin = np.power(10.0, levels / 10.0)
    return float(power_to_db(np.sum(weights * lin) / np.sum(weights)))


def fractional_octave_center_frequencies(
    fraction: int,
    *,
    f_min_hz: float = 20.0,
    f_max_hz: float = 20000.0,
    reference_hz: float = 1000.0,
) -> NDArray[np.float64]:
    """Exact base-10 fractional-octave centre frequencies."""

    b = int(fraction)
    if b <= 0:
        raise ValueError("fraction must be positive")
    f_min = _positive(f_min_hz, "f_min_hz")
    f_max = _positive(f_max_hz, "f_max_hz")
    ref = _positive(reference_hz, "reference_hz")
    if f_max < f_min:
        raise ValueError("f_max_hz must be >= f_min_hz")
    k_min = int(np.ceil((10.0 * b / 3.0) * np.log10(f_min / ref)))
    k_max = int(np.floor((10.0 * b / 3.0) * np.log10(f_max / ref)))
    k = np.arange(k_min, k_max + 1)
    return ref * np.power(10.0, 3.0 * k / (10.0 * b))


def fractional_octave_edges(
    center_hz: ArrayLike,
    fraction: int,
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Lower and upper exact band edges for fractional-octave centres."""

    b = int(fraction)
    if b <= 0:
        raise ValueError("fraction must be positive")
    center = _arr(center_hz)
    if np.any(center <= 0.0):
        raise ValueError("center_hz must be positive")
    ratio = np.power(10.0, 3.0 / (20.0 * b))
    return _maybe_scalar(center / ratio), _maybe_scalar(center * ratio)


def nominal_fractional_octave_frequency(center_hz: float) -> float:
    """Round a centre frequency to a readable nominal band label."""

    f = _positive(center_hz, "center_hz")
    if f < 100.0:
        return float(np.round(f, 1))
    if f < 1000.0:
        return float(np.round(f))
    return float(np.round(f / 10.0) * 10.0)


def fractional_octave_bands(
    fraction: int,
    *,
    f_min_hz: float = 20.0,
    f_max_hz: float = 20000.0,
    reference_hz: float = 1000.0,
) -> tuple[FractionalOctaveBand, ...]:
    """Return exact fractional-octave bands over a frequency range."""

    centers = fractional_octave_center_frequencies(
        fraction,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        reference_hz=reference_hz,
    )
    bands = []
    for center in centers:
        lo, hi = fractional_octave_edges(float(center), fraction)
        bands.append(
            FractionalOctaveBand(
                center_hz=float(center),
                lower_hz=float(lo),
                upper_hz=float(hi),
                fraction=int(fraction),
                nominal_hz=nominal_fractional_octave_frequency(float(center)),
            )
        )
    return tuple(bands)


def octave_band_centers(**kwargs: float) -> NDArray[np.float64]:
    """Convenience wrapper for octave-band centre frequencies."""

    return fractional_octave_center_frequencies(1, **kwargs)


def third_octave_band_centers(**kwargs: float) -> NDArray[np.float64]:
    """Convenience wrapper for one-third-octave centre frequencies."""

    return fractional_octave_center_frequencies(3, **kwargs)


def nearest_fractional_octave_band(
    frequency_hz: float,
    fraction: int,
    *,
    reference_hz: float = 1000.0,
) -> FractionalOctaveBand:
    """Return the fractional-octave band whose centre is nearest to a frequency."""

    f = _positive(frequency_hz, "frequency_hz")
    b = int(fraction)
    ref = _positive(reference_hz, "reference_hz")
    k = int(np.round((10.0 * b / 3.0) * np.log10(f / ref)))
    center = ref * np.power(10.0, 3.0 * k / (10.0 * b))
    lo, hi = fractional_octave_edges(float(center), b)
    return FractionalOctaveBand(
        center_hz=float(center),
        lower_hz=float(lo),
        upper_hz=float(hi),
        fraction=b,
        nominal_hz=nominal_fractional_octave_frequency(float(center)),
    )


def a_weighting_db(frequencies_hz: ArrayLike) -> float | NDArray[np.float64]:
    """IEC-style A-frequency-weighting curve in dB."""

    f = _arr(frequencies_hz)
    if np.any(f <= 0.0):
        raise ValueError("frequencies_hz must be positive")
    f2 = f * f
    ra = (
        (12200.0**2 * f2 * f2)
        / (
            (f2 + 20.6**2)
            * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
            * (f2 + 12200.0**2)
        )
    )
    return _maybe_scalar(20.0 * np.log10(ra) + 2.0)


def c_weighting_db(frequencies_hz: ArrayLike) -> float | NDArray[np.float64]:
    """IEC-style C-frequency-weighting curve in dB."""

    f = _arr(frequencies_hz)
    if np.any(f <= 0.0):
        raise ValueError("frequencies_hz must be positive")
    f2 = f * f
    rc = (12200.0**2 * f2) / ((f2 + 20.6**2) * (f2 + 12200.0**2))
    return _maybe_scalar(20.0 * np.log10(rc) + 0.06)


def z_weighting_db(frequencies_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Z-weighting, i.e. no frequency weighting."""

    f = _arr(frequencies_hz)
    if np.any(f <= 0.0):
        raise ValueError("frequencies_hz must be positive")
    return _maybe_scalar(np.zeros_like(f, dtype=float))


def frequency_weighting_db(
    frequencies_hz: ArrayLike,
    weighting: FrequencyWeighting = "A",
) -> float | NDArray[np.float64]:
    """Return A/C/Z frequency-weighting offsets in dB."""

    key = str(weighting).upper()
    if key == "A":
        return a_weighting_db(frequencies_hz)
    if key == "C":
        return c_weighting_db(frequencies_hz)
    if key == "Z":
        return z_weighting_db(frequencies_hz)
    raise ValueError("weighting must be 'A', 'C', or 'Z'")


def apply_frequency_weighting(
    levels_db: ArrayLike,
    frequencies_hz: ArrayLike,
    weighting: FrequencyWeighting = "A",
) -> NDArray[np.float64]:
    """Add frequency-weighting offsets to band or spectral levels."""

    return _arr(levels_db) + _arr(frequency_weighting_db(frequencies_hz, weighting))


def speed_of_sound(
    temperature_c: float = 20.0,
    *,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> float:
    """Ideal-gas speed of sound in dry air."""

    temp_k = float(temperature_c) + 273.15
    if temp_k <= 0.0:
        raise ValueError("temperature must be above absolute zero")
    return float(np.sqrt(float(gamma) * float(gas_constant) * temp_k))


def temperature_from_speed_of_sound(
    speed_m_s: float,
    *,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> float:
    """Dry-air temperature in Celsius implied by sound speed."""

    c0 = _positive(speed_m_s, "speed_m_s")
    return float((c0 * c0) / (float(gamma) * float(gas_constant)) - 273.15)


def mach_number(speed_m_s: ArrayLike, *, c: float = 343.0) -> float | NDArray[np.float64]:
    """Mach number relative to a sound speed."""

    return _maybe_scalar(_arr(speed_m_s) / _positive(c, "c"))


def speed_from_mach(mach: ArrayLike, *, c: float = 343.0) -> float | NDArray[np.float64]:
    """Speed in m/s from Mach number."""

    return _maybe_scalar(_arr(mach) * _positive(c, "c"))


def air_density(
    temperature_c: float = 20.0,
    pressure_kpa: float = 101.325,
    *,
    gas_constant: float = 287.05,
) -> float:
    """Dry-air density from the ideal gas law in kg/m^3."""

    temp_k = float(temperature_c) + 273.15
    if temp_k <= 0.0:
        raise ValueError("temperature must be above absolute zero")
    pressure_pa = _positive(pressure_kpa, "pressure_kpa") * 1000.0
    return float(pressure_pa / (float(gas_constant) * temp_k))


def characteristic_impedance(
    temperature_c: float = 20.0,
    pressure_kpa: float = 101.325,
) -> float:
    """Characteristic impedance ``rho c`` of dry air in Pa s / m."""

    return air_density(temperature_c, pressure_kpa) * speed_of_sound(temperature_c)


def directivity_factor_to_index(directivity_factor: ArrayLike) -> float | NDArray[np.float64]:
    """Convert directivity factor ``Q`` to directivity index in dB."""

    q = _arr(directivity_factor)
    if np.any(q <= 0.0):
        raise ValueError("directivity_factor must be positive")
    return power_to_db(q)


def directivity_index_to_factor(directivity_index_db: ArrayLike) -> float | NDArray[np.float64]:
    """Convert directivity index in dB to directivity factor ``Q``."""

    return db_to_power(directivity_index_db)


def spherical_spreading_loss_db(
    distance_m: ArrayLike,
    *,
    reference_distance_m: float = 1.0,
) -> float | NDArray[np.float64]:
    """Free-field spherical spreading loss relative to a reference distance."""

    d = _arr(distance_m)
    ref = _positive(reference_distance_m, "reference_distance_m")
    if np.any(d <= 0.0):
        raise ValueError("distance_m must be positive")
    return amplitude_to_db(d, reference=ref)


def distance_attenuated_spl(
    source_spl_db: ArrayLike,
    distance_m: ArrayLike,
    *,
    reference_distance_m: float = 1.0,
) -> float | NDArray[np.float64]:
    """Apply free-field distance attenuation to an SPL value."""

    return _maybe_scalar(_arr(source_spl_db) - _arr(spherical_spreading_loss_db(distance_m, reference_distance_m=reference_distance_m)))


def absorption_to_reflection(absorption: ArrayLike) -> float | NDArray[np.float64]:
    """Energy absorption coefficient to pressure reflection magnitude."""

    alpha = _arr(absorption)
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("absorption must lie in [0, 1]")
    return _maybe_scalar(np.sqrt(np.maximum(0.0, 1.0 - alpha)))


def reflection_to_absorption(reflection: ArrayLike) -> float | NDArray[np.float64]:
    """Pressure reflection magnitude to energy absorption coefficient."""

    r = _arr(reflection)
    if np.any(r < 0.0) or np.any(r > 1.0):
        raise ValueError("reflection must lie in [0, 1]")
    return _maybe_scalar(1.0 - r * r)


def bark_rate(frequencies_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Zwicker-style Bark-rate approximation."""

    f = _arr(frequencies_hz)
    if np.any(f < 0.0):
        raise ValueError("frequencies_hz must be non-negative")
    z = 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
    return _maybe_scalar(z)


def erb_rate(frequencies_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Glasberg-Moore ERB-rate approximation."""

    f = _arr(frequencies_hz)
    if np.any(f < 0.0):
        raise ValueError("frequencies_hz must be non-negative")
    return _maybe_scalar(21.4 * np.log10(1.0 + 0.00437 * f))


def mel_rate(frequencies_hz: ArrayLike) -> float | NDArray[np.float64]:
    """O'Shaughnessy-style mel-rate approximation."""

    f = _arr(frequencies_hz)
    if np.any(f < 0.0):
        raise ValueError("frequencies_hz must be non-negative")
    return _maybe_scalar(2595.0 * np.log10(1.0 + f / 700.0))


def inverse_mel_rate(mel: ArrayLike) -> float | NDArray[np.float64]:
    """Invert :func:`mel_rate`."""

    m = _arr(mel)
    if np.any(m < 0.0):
        raise ValueError("mel must be non-negative")
    return _maybe_scalar(700.0 * (np.power(10.0, m / 2595.0) - 1.0))


def inverse_erb_rate(erb: ArrayLike) -> float | NDArray[np.float64]:
    """Invert :func:`erb_rate`."""

    e = _arr(erb)
    if np.any(e < 0.0):
        raise ValueError("erb must be non-negative")
    return _maybe_scalar((np.power(10.0, e / 21.4) - 1.0) / 0.00437)


def erb_bandwidth(frequencies_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Equivalent rectangular bandwidth in hertz."""

    f = _arr(frequencies_hz)
    if np.any(f < 0.0):
        raise ValueError("frequencies_hz must be non-negative")
    return _maybe_scalar(24.7 * (4.37e-3 * f + 1.0))


def critical_bandwidth(frequencies_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Zwicker critical-band bandwidth approximation in hertz."""

    f = _arr(frequencies_hz)
    if np.any(f < 0.0):
        raise ValueError("frequencies_hz must be non-negative")
    bw = 25.0 + 75.0 * np.power(1.0 + 1.4 * (f / 1000.0) ** 2, 0.69)
    return _maybe_scalar(bw)


def semitones_between(frequency_a_hz: ArrayLike, frequency_b_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Pitch interval from ``frequency_a`` to ``frequency_b`` in semitones."""

    fa = _arr(frequency_a_hz)
    fb = _arr(frequency_b_hz)
    if np.any(fa <= 0.0) or np.any(fb <= 0.0):
        raise ValueError("frequencies must be positive")
    return _maybe_scalar(12.0 * np.log2(fb / fa))


def frequency_ratio_from_semitones(semitones: ArrayLike) -> float | NDArray[np.float64]:
    """Frequency ratio for a semitone interval."""

    return _maybe_scalar(np.power(2.0, _arr(semitones) / 12.0))


def cents_between(frequency_a_hz: ArrayLike, frequency_b_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Pitch interval from ``frequency_a`` to ``frequency_b`` in cents."""

    return _maybe_scalar(100.0 * _arr(semitones_between(frequency_a_hz, frequency_b_hz)))


def frequency_ratio_from_cents(cents: ArrayLike) -> float | NDArray[np.float64]:
    """Frequency ratio for a cents interval."""

    return _maybe_scalar(np.power(2.0, _arr(cents) / 1200.0))


def phon_to_sone(phon: ArrayLike) -> float | NDArray[np.float64]:
    """Approximate loudness in sones from loudness level in phons."""

    return _maybe_scalar(np.power(2.0, (_arr(phon) - 40.0) / 10.0))


def sone_to_phon(sone: ArrayLike) -> float | NDArray[np.float64]:
    """Approximate loudness level in phons from sones."""

    s = _arr(sone)
    if np.any(s <= 0.0):
        raise ValueError("sone must be positive")
    return _maybe_scalar(40.0 + 10.0 * np.log2(s))


def octave_ratio(octaves: ArrayLike = 1.0) -> float | NDArray[np.float64]:
    """Frequency ratio for an octave interval."""

    return _maybe_scalar(np.power(2.0, _arr(octaves)))


def fractional_octave_ratio(fraction: int) -> float:
    """Exact centre-to-edge ratio for a fractional-octave band."""

    b = int(fraction)
    if b <= 0:
        raise ValueError("fraction must be positive")
    return float(np.power(10.0, 3.0 / (20.0 * b)))


def bandwidth_from_edges(lower_hz: ArrayLike, upper_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Bandwidth from lower and upper band edges."""

    lo = _arr(lower_hz)
    hi = _arr(upper_hz)
    if np.any(lo <= 0.0) or np.any(hi <= lo):
        raise ValueError("band edges must satisfy 0 < lower_hz < upper_hz")
    return _maybe_scalar(hi - lo)


def center_frequency_from_edges(lower_hz: ArrayLike, upper_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Geometric centre frequency from band edges."""

    lo = _arr(lower_hz)
    hi = _arr(upper_hz)
    if np.any(lo <= 0.0) or np.any(hi <= lo):
        raise ValueError("band edges must satisfy 0 < lower_hz < upper_hz")
    return _maybe_scalar(np.sqrt(lo * hi))


def q_factor_from_band_edges(lower_hz: ArrayLike, upper_hz: ArrayLike) -> float | NDArray[np.float64]:
    """Filter Q from geometric centre and edge bandwidth."""

    return _maybe_scalar(_arr(center_frequency_from_edges(lower_hz, upper_hz)) / _arr(bandwidth_from_edges(lower_hz, upper_hz)))


def band_edges_from_center_q(center_hz: ArrayLike, q_factor: ArrayLike) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Band edges from geometric centre frequency and Q."""

    fc = _arr(center_hz)
    q = _arr(q_factor)
    if np.any(fc <= 0.0) or np.any(q <= 0.0):
        raise ValueError("center_hz and q_factor must be positive")
    inv_q = 1.0 / q
    ratio = 0.5 * (inv_q + np.sqrt(inv_q * inv_q + 4.0))
    return _maybe_scalar(fc / ratio), _maybe_scalar(fc * ratio)


def absorption_loss_db(absorption: ArrayLike, *, floor: float = 1e-30) -> float | NDArray[np.float64]:
    """Positive reflected-energy loss implied by an absorption coefficient."""

    alpha = _arr(absorption)
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("absorption must lie in [0, 1]")
    return _maybe_scalar(-10.0 * np.log10(np.maximum(1.0 - alpha, float(floor))))


def reflection_loss_db(reflection: ArrayLike, *, floor: float = 1e-30) -> float | NDArray[np.float64]:
    """Positive pressure-amplitude loss implied by a reflection magnitude."""

    r = _arr(reflection)
    if np.any(r < 0.0) or np.any(r > 1.0):
        raise ValueError("reflection must lie in [0, 1]")
    return _maybe_scalar(-20.0 * np.log10(np.maximum(r, float(floor))))


def doppler_shift_moving_source(
    source_frequency_hz: ArrayLike,
    source_velocity_m_s: float,
    *,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Observed frequency for a source moving toward the observer."""

    speed = _positive(c, "c")
    denom = speed - float(source_velocity_m_s)
    if denom <= 0.0:
        raise ValueError("source velocity must be lower than sound speed")
    return _maybe_scalar(_arr(source_frequency_hz) * speed / denom)


def doppler_shift_moving_observer(
    source_frequency_hz: ArrayLike,
    observer_velocity_m_s: float,
    *,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Observed frequency for an observer moving toward the source."""

    speed = _positive(c, "c")
    return _maybe_scalar(_arr(source_frequency_hz) * (speed + float(observer_velocity_m_s)) / speed)


def rpm_to_hz(rpm: ArrayLike) -> float | NDArray[np.float64]:
    """Rotational speed in revolutions per minute to hertz."""

    return _maybe_scalar(_arr(rpm) / 60.0)


def hz_to_rpm(hz: ArrayLike) -> float | NDArray[np.float64]:
    """Rotational speed in hertz to revolutions per minute."""

    return _maybe_scalar(60.0 * _arr(hz))


__all__ = [
    "FractionalOctaveBand",
    "REFERENCE_PRESSURE_PA",
    "REFERENCE_SOUND_INTENSITY_W_M2",
    "REFERENCE_SOUND_POWER_W",
    "a_weighting_db",
    "absorption_to_reflection",
    "absorption_loss_db",
    "air_density",
    "amplitude_to_db",
    "amplitude_ratio_from_db",
    "angular_frequency",
    "apply_frequency_weighting",
    "bark_rate",
    "band_edges_from_center_q",
    "bandwidth_from_edges",
    "c_weighting_db",
    "center_frequency_from_edges",
    "characteristic_impedance",
    "cents_between",
    "crest_factor",
    "crest_factor_db",
    "critical_bandwidth",
    "db_from_amplitude_ratio",
    "db_from_power_ratio",
    "db_to_amplitude",
    "db_to_power",
    "directivity_factor_to_index",
    "directivity_index_to_factor",
    "distance_delay",
    "distance_attenuated_spl",
    "distance_ratio_for_level_change",
    "distance_from_sample_delay",
    "doppler_shift_moving_observer",
    "doppler_shift_moving_source",
    "energetic_mean_db",
    "energetic_sum_db",
    "equivalent_continuous_level",
    "erb_bandwidth",
    "erb_rate",
    "fractional_octave_bands",
    "fractional_octave_center_frequencies",
    "fractional_octave_edges",
    "fractional_octave_ratio",
    "frequency_from_angular_frequency",
    "frequency_from_period",
    "frequency_from_wavelength",
    "frequency_ratio_from_cents",
    "frequency_ratio_from_semitones",
    "frequency_weighting_db",
    "hz_to_rpm",
    "intensity_ratio_from_level_difference",
    "intensity_to_sound_intensity_level",
    "inverse_erb_rate",
    "inverse_mel_rate",
    "leq",
    "level_change_for_distance_ratio",
    "level_difference_db",
    "mach_number",
    "mel_rate",
    "nearest_fractional_octave_band",
    "nominal_fractional_octave_frequency",
    "octave_ratio",
    "octave_band_centers",
    "particle_velocity_from_pressure",
    "peak_to_peak_to_rms_sine",
    "peak_to_rms_sine",
    "period",
    "phase_degrees_to_radians",
    "phase_radians_to_degrees",
    "phase_shift_for_delay",
    "phon_to_sone",
    "plane_wave_intensity_from_pressure",
    "plane_wave_pressure_from_intensity",
    "power_to_db",
    "power_ratio_from_db",
    "pressure_from_particle_velocity",
    "pressure_ratio_from_spl_difference",
    "pressure_rms",
    "pressure_to_spl",
    "q_factor_from_band_edges",
    "reflection_to_absorption",
    "reflection_loss_db",
    "rms_to_peak_sine",
    "rms_to_peak_to_peak_sine",
    "rpm_to_hz",
    "sample_delay_from_distance",
    "semitones_between",
    "sone_to_phon",
    "sound_exposure",
    "sound_exposure_level",
    "sound_intensity_level_to_intensity",
    "sound_power_level_to_power",
    "sound_power_to_sound_power_level",
    "speed_of_sound",
    "speed_from_mach",
    "spherical_spreading_loss_db",
    "spl_to_pressure",
    "third_octave_band_centers",
    "temperature_from_speed_of_sound",
    "time_delay_for_phase_shift",
    "wavelength",
    "wrap_phase_deg",
    "wrap_phase_rad",
    "z_weighting_db",
]
