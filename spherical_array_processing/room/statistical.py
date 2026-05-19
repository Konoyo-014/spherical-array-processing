"""Statistical room-acoustics formulas for early-stage design.

The functions in this module implement diffuse-field estimates and
rectangular-room modal formulas that are commonly used before a full
geometric or wave-based simulation is warranted.  They are deliberately
kept separate from the measured-RIR metrics in :mod:`.metrics`: Sabine,
Eyring, Millington-Sette, and Arau-Puchades predict reverberation from
geometry and absorption, while ISO 3382-style metrics estimate it from
an actual impulse response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


_REFERENCE_PRESSURE_KPA = 101.325
_REFERENCE_TEMPERATURE_K = 293.15
_TRIPLE_POINT_TEMPERATURE_K = 273.16


@dataclass(frozen=True)
class ShoeboxAcousticStats:
    """Bundle of statistical acoustic quantities for a rectangular room."""

    dimensions_m: NDArray[np.float64]
    volume_m3: float
    surface_area_m2: float
    surface_areas_m2: NDArray[np.float64]
    equivalent_absorption_area_m2: Any
    mean_absorption: Any
    room_constant_m2: Any
    sabine_rt60_s: Any
    eyring_rt60_s: Any
    millington_sette_rt60_s: Any
    arau_puchades_rt60_s: Any
    schroeder_frequency_hz: Any
    critical_distance_m: Any


def _maybe_scalar(x: NDArray[np.float64]) -> float | NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


def _positive_scalar(value: float, name: str) -> float:
    val = float(value)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be positive")
    return val


def _dimensions(dimensions_m: ArrayLike) -> NDArray[np.float64]:
    dims = np.asarray(dimensions_m, dtype=float).reshape(-1)
    if dims.size != 3:
        raise ValueError("dimensions_m must contain exactly three lengths")
    if np.any(~np.isfinite(dims)) or np.any(dims <= 0.0):
        raise ValueError("room dimensions must be finite and positive")
    return dims


def _surface_vector(surface_areas_m2: ArrayLike) -> NDArray[np.float64]:
    areas = np.asarray(surface_areas_m2, dtype=float).reshape(-1)
    if areas.size == 0:
        raise ValueError("surface_areas_m2 must be non-empty")
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("surface areas must be finite and positive")
    return areas


def _absorption_matrix(
    absorption: ArrayLike,
    n_surfaces: int,
) -> tuple[NDArray[np.float64], bool]:
    alpha = np.asarray(absorption, dtype=float)
    scalar_output = False
    if alpha.ndim == 0:
        alpha = np.full((n_surfaces, 1), float(alpha), dtype=float)
        scalar_output = True
    elif alpha.ndim == 1:
        if alpha.size != n_surfaces:
            raise ValueError(
                f"absorption must be scalar or have {n_surfaces} surface values"
            )
        alpha = alpha.reshape(n_surfaces, 1)
        scalar_output = True
    elif alpha.ndim == 2:
        if alpha.shape[0] == n_surfaces:
            alpha = alpha.astype(float, copy=False)
        elif alpha.shape[1] == n_surfaces:
            alpha = alpha.T.astype(float, copy=False)
        else:
            raise ValueError(
                f"absorption must have {n_surfaces} rows or columns for surfaces"
            )
    else:
        raise ValueError("absorption must be scalar, 1-D, or 2-D")
    if np.any(~np.isfinite(alpha)) or np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("absorption coefficients must lie in [0, 1]")
    return alpha, scalar_output


def _rt_constant(c: float) -> float:
    return 24.0 * float(np.log(10.0)) / _positive_scalar(c, "c")


def shoebox_surface_areas(dimensions_m: ArrayLike) -> NDArray[np.float64]:
    """Return the six wall areas of a shoebox room.

    The wall order is ``[-x, +x, -y, +y, -z, +z]``.  This matches the
    common per-wall absorption order used by image-source shoebox
    simulators.
    """
    lx, ly, lz = _dimensions(dimensions_m)
    return np.array(
        [ly * lz, ly * lz, lx * lz, lx * lz, lx * ly, lx * ly],
        dtype=float,
    )


def shoebox_axis_surface_areas(dimensions_m: ArrayLike) -> NDArray[np.float64]:
    """Return paired wall areas perpendicular to the x, y, and z axes."""
    areas = shoebox_surface_areas(dimensions_m)
    return np.array([areas[0] + areas[1], areas[2] + areas[3], areas[4] + areas[5]])


def shoebox_volume(dimensions_m: ArrayLike) -> float:
    """Return rectangular-room volume in cubic metres."""
    return float(np.prod(_dimensions(dimensions_m)))


def equivalent_absorption_area(
    surface_areas_m2: ArrayLike,
    absorption: ArrayLike,
) -> float | NDArray[np.float64]:
    """Equivalent absorption area ``A = sum(S_i alpha_i)`` in square metres."""
    surfaces = _surface_vector(surface_areas_m2)
    alpha, scalar = _absorption_matrix(absorption, surfaces.size)
    area = surfaces @ alpha
    return _maybe_scalar(area) if scalar else area


def mean_absorption(
    surface_areas_m2: ArrayLike,
    absorption: ArrayLike,
) -> float | NDArray[np.float64]:
    """Surface-area-weighted mean absorption coefficient."""
    surfaces = _surface_vector(surface_areas_m2)
    alpha, scalar = _absorption_matrix(absorption, surfaces.size)
    mean = (surfaces @ alpha) / float(np.sum(surfaces))
    return _maybe_scalar(mean) if scalar else mean


def room_constant(
    surface_areas_m2: ArrayLike,
    absorption: ArrayLike,
) -> float | NDArray[np.float64]:
    """Diffuse-field room constant ``R = S alpha_bar / (1 - alpha_bar)``."""
    surfaces = _surface_vector(surface_areas_m2)
    alpha_bar = np.asarray(mean_absorption(surfaces, absorption), dtype=float)
    total_surface = float(np.sum(surfaces))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = total_surface * alpha_bar / (1.0 - alpha_bar)
    return _maybe_scalar(np.asarray(r, dtype=float))


def sabine_rt60(
    volume_m3: float,
    surface_areas_m2: ArrayLike,
    absorption: ArrayLike,
    *,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Sabine diffuse-field reverberation time estimate in seconds."""
    volume = _positive_scalar(volume_m3, "volume_m3")
    surfaces = _surface_vector(surface_areas_m2)
    alpha, scalar = _absorption_matrix(absorption, surfaces.size)
    denom = surfaces @ alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = _rt_constant(c) * volume / denom
    return _maybe_scalar(rt) if scalar else rt


def eyring_rt60(
    volume_m3: float,
    surface_areas_m2: ArrayLike,
    absorption: ArrayLike,
    *,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Eyring reverberation time using the mean absorption coefficient."""
    volume = _positive_scalar(volume_m3, "volume_m3")
    surfaces = _surface_vector(surface_areas_m2)
    alpha_bar = np.asarray(mean_absorption(surfaces, absorption), dtype=float)
    total_surface = float(np.sum(surfaces))
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = -total_surface * np.log1p(-alpha_bar)
        rt = _rt_constant(c) * volume / denom
    return _maybe_scalar(np.asarray(rt, dtype=float))


def millington_sette_rt60(
    volume_m3: float,
    surface_areas_m2: ArrayLike,
    absorption: ArrayLike,
    *,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Millington-Sette RT60 with surface-specific logarithmic losses."""
    volume = _positive_scalar(volume_m3, "volume_m3")
    surfaces = _surface_vector(surface_areas_m2)
    alpha, scalar = _absorption_matrix(absorption, surfaces.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = surfaces @ (-np.log1p(-alpha))
        rt = _rt_constant(c) * volume / denom
    return _maybe_scalar(rt) if scalar else rt


def arau_puchades_rt60(
    dimensions_m: ArrayLike,
    absorption: ArrayLike,
    *,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Arau-Puchades RT60 for asymmetric absorption in shoebox rooms.

    The estimate forms directional Eyring times from the mean absorption
    of each opposite wall pair, then combines them as a geometric
    average weighted by the paired wall areas.  Uniform absorption
    therefore reduces exactly to the Eyring estimate.
    """
    dims = _dimensions(dimensions_m)
    volume = float(np.prod(dims))
    surfaces = shoebox_surface_areas(dims)
    axis_surfaces = shoebox_axis_surface_areas(dims)
    alpha, scalar = _absorption_matrix(absorption, 6)
    alpha_axis = np.vstack(
        [
            (surfaces[0] * alpha[0] + surfaces[1] * alpha[1]) / axis_surfaces[0],
            (surfaces[2] * alpha[2] + surfaces[3] * alpha[3]) / axis_surfaces[1],
            (surfaces[4] * alpha[4] + surfaces[5] * alpha[5]) / axis_surfaces[2],
        ]
    )
    total_surface = float(np.sum(surfaces))
    weights = axis_surfaces / total_surface
    with np.errstate(divide="ignore", invalid="ignore"):
        directional = _rt_constant(c) * volume / (
            -total_surface * np.log1p(-alpha_axis)
        )
        log_rt = weights @ np.log(directional)
        rt = np.exp(log_rt)
    return _maybe_scalar(rt) if scalar else rt


def schroeder_frequency(
    rt60_s: ArrayLike,
    volume_m3: float,
) -> float | NDArray[np.float64]:
    """Schroeder transition frequency ``f_s = 2000 sqrt(T60 / V)``."""
    volume = _positive_scalar(volume_m3, "volume_m3")
    rt = np.asarray(rt60_s, dtype=float)
    if np.any(~np.isfinite(rt)) or np.any(rt < 0.0):
        raise ValueError("rt60_s must be finite and non-negative")
    return _maybe_scalar(2000.0 * np.sqrt(rt / volume))


def critical_distance_from_rt60(
    volume_m3: float,
    rt60_s: ArrayLike,
    *,
    directivity_factor: float = 1.0,
    c: float = 343.0,
) -> float | NDArray[np.float64]:
    """Diffuse-field critical distance from volume and RT60.

    This uses the Sabine equivalent absorption area implied by RT60:
    ``A = (24 ln 10 / c) V / T60``, then
    ``d_c = sqrt(Q A / (16 pi))``.
    """
    volume = _positive_scalar(volume_m3, "volume_m3")
    q = _positive_scalar(directivity_factor, "directivity_factor")
    rt = np.asarray(rt60_s, dtype=float)
    if np.any(~np.isfinite(rt)) or np.any(rt <= 0.0):
        raise ValueError("rt60_s must be finite and positive")
    a = _rt_constant(c) * volume / rt
    return _maybe_scalar(np.sqrt(q * a / (16.0 * np.pi)))


def critical_distance(
    surface_areas_m2: ArrayLike,
    absorption: ArrayLike,
    *,
    directivity_factor: float = 1.0,
) -> float | NDArray[np.float64]:
    """Diffuse-field critical distance from room constant and directivity."""
    q = _positive_scalar(directivity_factor, "directivity_factor")
    r = np.asarray(room_constant(surface_areas_m2, absorption), dtype=float)
    with np.errstate(invalid="ignore"):
        dc = np.sqrt(q * r / (16.0 * np.pi))
    return _maybe_scalar(dc)


def rectangular_room_modes(
    dimensions_m: ArrayLike,
    max_frequency_hz: float,
    *,
    c: float = 343.0,
) -> NDArray[np.float64]:
    """Modal frequencies of a rigid rectangular room up to a cutoff.

    Returns an array with columns ``nx, ny, nz, frequency_hz`` sorted by
    frequency.  The all-zero mode is excluded.
    """
    dims = _dimensions(dimensions_m)
    fmax = _positive_scalar(max_frequency_hz, "max_frequency_hz")
    speed = _positive_scalar(c, "c")
    max_indices = np.ceil(2.0 * fmax * dims / speed).astype(int)
    rows: list[tuple[int, int, int, float]] = []
    for nx in range(max_indices[0] + 1):
        for ny in range(max_indices[1] + 1):
            for nz in range(max_indices[2] + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                freq = 0.5 * speed * np.sqrt(
                    (nx / dims[0]) ** 2
                    + (ny / dims[1]) ** 2
                    + (nz / dims[2]) ** 2
                )
                if freq <= fmax + 1e-12:
                    rows.append((nx, ny, nz, float(freq)))
    if not rows:
        return np.empty((0, 4), dtype=float)
    out = np.asarray(rows, dtype=float)
    order = np.argsort(out[:, 3], kind="mergesort")
    return out[order]


def classify_room_modes(mode_indices: ArrayLike) -> NDArray[np.str_]:
    """Classify rectangular-room mode indices as axial/tangential/oblique."""
    idx = np.asarray(mode_indices, dtype=int)
    if idx.shape[-1] != 3:
        raise ValueError("mode_indices must have a final dimension of length 3")
    nonzero = np.count_nonzero(idx, axis=-1)
    labels = np.empty(nonzero.shape, dtype="<U10")
    labels[nonzero == 1] = "axial"
    labels[nonzero == 2] = "tangential"
    labels[nonzero == 3] = "oblique"
    labels[nonzero == 0] = "zero"
    return labels


def air_absorption_coefficient_iso9613(
    frequencies_hz: ArrayLike,
    *,
    temperature_c: float = 20.0,
    relative_humidity: float = 0.5,
    pressure_kpa: float = _REFERENCE_PRESSURE_KPA,
) -> NDArray[np.float64]:
    """Atmospheric absorption coefficient from ISO 9613-1, in dB/m.

    ``relative_humidity`` is a fraction in ``[0, 1]``.  The formula is
    valid for ordinary audible-frequency atmospheric propagation; it is
    a loss model, not a replacement for full outdoor sound propagation
    standards.
    """
    f = np.asarray(frequencies_hz, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("frequencies_hz must be finite and non-negative")
    temperature_k = float(temperature_c) + 273.15
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_c gives a non-positive Kelvin temperature")
    rh = float(relative_humidity)
    if not np.isfinite(rh) or rh < 0.0 or rh > 1.0:
        raise ValueError("relative_humidity must lie in [0, 1]")
    pressure = _positive_scalar(pressure_kpa, "pressure_kpa")

    # Saturation vapour pressure ratio from ISO 9613-1 Annex-style
    # formulation.  h is the molar concentration of water vapour in %.
    c_sat = -6.8346 * (_TRIPLE_POINT_TEMPERATURE_K / temperature_k) ** 1.261 + 4.6151
    p_sat_over_pr = 10.0 ** c_sat
    h = rh * 100.0 * p_sat_over_pr * (_REFERENCE_PRESSURE_KPA / pressure)
    tr = temperature_k / _REFERENCE_TEMPERATURE_K
    pr_over_pa = _REFERENCE_PRESSURE_KPA / pressure
    pa_over_pr = pressure / _REFERENCE_PRESSURE_KPA
    fr_o = pa_over_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pa_over_pr * tr ** (-0.5) * (
        9.0 + 280.0 * h * np.exp(-4.170 * (tr ** (-1.0 / 3.0) - 1.0))
    )
    classical = 1.84e-11 * pr_over_pa * np.sqrt(tr)
    oxygen = 0.01275 * np.exp(-2239.1 / temperature_k) / (fr_o + f * f / fr_o)
    nitrogen = 0.1068 * np.exp(-3352.0 / temperature_k) / (fr_n + f * f / fr_n)
    relaxation = tr ** (-2.5) * (oxygen + nitrogen)
    alpha = 8.686 * f * f * (classical + relaxation)
    return np.asarray(alpha, dtype=float)


def air_absorption_attenuation_iso9613(
    frequencies_hz: ArrayLike,
    distance_m: float,
    *,
    temperature_c: float = 20.0,
    relative_humidity: float = 0.5,
    pressure_kpa: float = _REFERENCE_PRESSURE_KPA,
) -> NDArray[np.float64]:
    """Atmospheric absorption loss over distance, in dB."""
    distance = _positive_scalar(distance_m, "distance_m")
    return distance * air_absorption_coefficient_iso9613(
        frequencies_hz,
        temperature_c=temperature_c,
        relative_humidity=relative_humidity,
        pressure_kpa=pressure_kpa,
    )


def shoebox_acoustic_stats(
    dimensions_m: ArrayLike,
    absorption: ArrayLike,
    *,
    c: float = 343.0,
    directivity_factor: float = 1.0,
) -> ShoeboxAcousticStats:
    """Compute a statistical room-acoustics summary for a shoebox."""
    dims = _dimensions(dimensions_m)
    volume = float(np.prod(dims))
    surfaces = shoebox_surface_areas(dims)
    sab = sabine_rt60(volume, surfaces, absorption, c=c)
    eyr = eyring_rt60(volume, surfaces, absorption, c=c)
    mil = millington_sette_rt60(volume, surfaces, absorption, c=c)
    arau = arau_puchades_rt60(dims, absorption, c=c)
    return ShoeboxAcousticStats(
        dimensions_m=dims,
        volume_m3=volume,
        surface_area_m2=float(np.sum(surfaces)),
        surface_areas_m2=surfaces,
        equivalent_absorption_area_m2=equivalent_absorption_area(surfaces, absorption),
        mean_absorption=mean_absorption(surfaces, absorption),
        room_constant_m2=room_constant(surfaces, absorption),
        sabine_rt60_s=sab,
        eyring_rt60_s=eyr,
        millington_sette_rt60_s=mil,
        arau_puchades_rt60_s=arau,
        schroeder_frequency_hz=schroeder_frequency(eyr, volume),
        critical_distance_m=critical_distance(
            surfaces,
            absorption,
            directivity_factor=directivity_factor,
        ),
    )


__all__ = [
    "ShoeboxAcousticStats",
    "air_absorption_attenuation_iso9613",
    "air_absorption_coefficient_iso9613",
    "arau_puchades_rt60",
    "classify_room_modes",
    "critical_distance",
    "critical_distance_from_rt60",
    "equivalent_absorption_area",
    "eyring_rt60",
    "mean_absorption",
    "millington_sette_rt60",
    "rectangular_room_modes",
    "room_constant",
    "sabine_rt60",
    "schroeder_frequency",
    "shoebox_acoustic_stats",
    "shoebox_axis_surface_areas",
    "shoebox_surface_areas",
    "shoebox_volume",
]
