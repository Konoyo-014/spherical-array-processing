"""Inverse room-acoustics design helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .statistical import (
    critical_distance,
    eyring_rt60,
    rectangular_room_modes,
    room_constant,
    sabine_rt60,
    shoebox_surface_areas,
    shoebox_volume,
)


@dataclass(frozen=True)
class RoomDesignTarget:
    """Inverse-design summary for a target reverberation time."""

    volume_m3: float
    surface_area_m2: float
    target_rt60_s: float
    equivalent_absorption_area_m2: float
    mean_absorption_sabine: float
    mean_absorption_eyring: float
    uniform_reflection_sabine: float
    uniform_reflection_eyring: float


def _positive(value: float, name: str) -> float:
    val = float(value)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be positive")
    return val


def rt_constant(c: float = 343.0) -> float:
    """Room-acoustics RT constant ``24 ln(10) / c``."""

    return float(24.0 * np.log(10.0) / _positive(c, "c"))


def target_absorption_area_sabine(volume_m3: float, target_rt60_s: float, *, c: float = 343.0) -> float:
    """Equivalent absorption area needed for a Sabine RT60 target."""

    return rt_constant(c) * _positive(volume_m3, "volume_m3") / _positive(target_rt60_s, "target_rt60_s")


def target_mean_absorption_sabine(
    volume_m3: float,
    surface_area_m2: float,
    target_rt60_s: float,
    *,
    c: float = 343.0,
) -> float:
    """Mean absorption coefficient needed under the Sabine model."""

    surface = _positive(surface_area_m2, "surface_area_m2")
    return target_absorption_area_sabine(volume_m3, target_rt60_s, c=c) / surface


def target_mean_absorption_eyring(
    volume_m3: float,
    surface_area_m2: float,
    target_rt60_s: float,
    *,
    c: float = 343.0,
) -> float:
    """Mean absorption coefficient needed under the Eyring model."""

    volume = _positive(volume_m3, "volume_m3")
    surface = _positive(surface_area_m2, "surface_area_m2")
    rt = _positive(target_rt60_s, "target_rt60_s")
    alpha = 1.0 - np.exp(-rt_constant(c) * volume / (surface * rt))
    return float(np.clip(alpha, 0.0, 1.0))


def mean_absorption_to_reflection(mean_absorption: float) -> float:
    """Uniform pressure reflection magnitude from energy absorption."""

    alpha = float(mean_absorption)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("mean_absorption must lie in [0, 1]")
    return float(np.sqrt(max(0.0, 1.0 - alpha)))


def reflection_to_mean_absorption(reflection: float) -> float:
    """Energy absorption coefficient from uniform reflection magnitude."""

    r = float(reflection)
    if not 0.0 <= r <= 1.0:
        raise ValueError("reflection must lie in [0, 1]")
    return float(1.0 - r * r)


def target_uniform_reflection_sabine(
    volume_m3: float,
    surface_area_m2: float,
    target_rt60_s: float,
    *,
    c: float = 343.0,
) -> float:
    """Uniform pressure reflection coefficient for a Sabine RT60 target."""

    return mean_absorption_to_reflection(
        target_mean_absorption_sabine(volume_m3, surface_area_m2, target_rt60_s, c=c)
    )


def target_uniform_reflection_eyring(
    volume_m3: float,
    surface_area_m2: float,
    target_rt60_s: float,
    *,
    c: float = 343.0,
) -> float:
    """Uniform pressure reflection coefficient for an Eyring RT60 target."""

    return mean_absorption_to_reflection(
        target_mean_absorption_eyring(volume_m3, surface_area_m2, target_rt60_s, c=c)
    )


def absorption_budget_per_surface(
    surface_areas_m2: ArrayLike,
    target_absorption_area_m2: float,
    *,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Allocate a target absorption area over surfaces as coefficients."""

    surfaces = np.asarray(surface_areas_m2, dtype=float).reshape(-1)
    if np.any(surfaces <= 0.0):
        raise ValueError("surface areas must be positive")
    target = _positive(target_absorption_area_m2, "target_absorption_area_m2")
    if weights is None:
        w = surfaces / float(np.sum(surfaces))
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != surfaces.shape:
            raise ValueError("weights must match surfaces")
        if np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
            raise ValueError("weights must be non-negative with positive sum")
        w = w / float(np.sum(w))
    alpha = target * w / surfaces
    return np.clip(alpha, 0.0, 1.0)


def room_design_target(
    dimensions_m: ArrayLike,
    target_rt60_s: float,
    *,
    c: float = 343.0,
) -> RoomDesignTarget:
    """Return inverse-design quantities for a shoebox room."""

    volume = shoebox_volume(dimensions_m)
    surfaces = shoebox_surface_areas(dimensions_m)
    surface = float(np.sum(surfaces))
    area = target_absorption_area_sabine(volume, target_rt60_s, c=c)
    alpha_s = target_mean_absorption_sabine(volume, surface, target_rt60_s, c=c)
    alpha_e = target_mean_absorption_eyring(volume, surface, target_rt60_s, c=c)
    return RoomDesignTarget(
        volume_m3=volume,
        surface_area_m2=surface,
        target_rt60_s=float(target_rt60_s),
        equivalent_absorption_area_m2=area,
        mean_absorption_sabine=alpha_s,
        mean_absorption_eyring=alpha_e,
        uniform_reflection_sabine=mean_absorption_to_reflection(alpha_s),
        uniform_reflection_eyring=mean_absorption_to_reflection(alpha_e),
    )


def verify_target_rt60(
    dimensions_m: ArrayLike,
    absorption: ArrayLike,
    *,
    model: str = "sabine",
    c: float = 343.0,
) -> float:
    """Compute predicted RT60 for a proposed absorption design."""

    volume = shoebox_volume(dimensions_m)
    surfaces = shoebox_surface_areas(dimensions_m)
    if model == "sabine":
        return float(sabine_rt60(volume, surfaces, absorption, c=c))
    if model == "eyring":
        return float(eyring_rt60(volume, surfaces, absorption, c=c))
    raise ValueError("model must be 'sabine' or 'eyring'")


def room_ratio(dimensions_m: ArrayLike) -> NDArray[np.float64]:
    """Room dimensions normalised by the smallest dimension."""

    dims = np.asarray(dimensions_m, dtype=float).reshape(-1)
    if dims.size != 3 or np.any(dims <= 0.0):
        raise ValueError("dimensions_m must contain three positive lengths")
    return np.sort(dims) / float(np.min(dims))


def bolt_area_score(dimensions_m: ArrayLike) -> float:
    """Heuristic distance from Bolt's classic room-ratio region."""

    ratio = room_ratio(dimensions_m)
    x = ratio[1]
    y = ratio[2]
    lower = 1.1 * x - 0.14
    upper = 4.0 * x - 4.0
    if lower <= y <= upper:
        return 0.0
    return float(min(abs(y - lower), abs(y - upper)))


def is_inside_bolt_area(dimensions_m: ArrayLike) -> bool:
    """Whether the sorted room ratio lies inside a simple Bolt-area approximation."""

    return bool(bolt_area_score(dimensions_m) == 0.0)


def modal_density_weyl(frequency_hz: ArrayLike, volume_m3: float, *, c: float = 343.0) -> NDArray[np.float64]:
    """Approximate cumulative modal count below frequency by Weyl's law."""

    f = np.asarray(frequency_hz, dtype=float)
    if np.any(f < 0.0):
        raise ValueError("frequency_hz must be non-negative")
    volume = _positive(volume_m3, "volume_m3")
    speed = _positive(c, "c")
    return 4.0 * np.pi * volume * f**3 / (3.0 * speed**3)


def modal_density_per_hz(frequency_hz: ArrayLike, volume_m3: float, *, c: float = 343.0) -> NDArray[np.float64]:
    """Approximate modal density ``dN/df`` by Weyl's law."""

    f = np.asarray(frequency_hz, dtype=float)
    if np.any(f < 0.0):
        raise ValueError("frequency_hz must be non-negative")
    return 4.0 * np.pi * _positive(volume_m3, "volume_m3") * f**2 / (_positive(c, "c") ** 3)


def count_modes_in_band(
    dimensions_m: ArrayLike,
    f_low_hz: float,
    f_high_hz: float,
    *,
    c: float = 343.0,
) -> int:
    """Count exact rectangular-room modes within a frequency band."""

    lo = float(f_low_hz)
    hi = _positive(f_high_hz, "f_high_hz")
    if lo < 0.0 or hi < lo:
        raise ValueError("band must satisfy 0 <= f_low_hz <= f_high_hz")
    modes = rectangular_room_modes(dimensions_m, hi, c=c)
    return int(np.count_nonzero((modes[:, 3] >= lo) & (modes[:, 3] <= hi)))


def modal_overlap_factor(frequency_hz: ArrayLike, volume_m3: float, rt60_s: float, *, c: float = 343.0) -> NDArray[np.float64]:
    """Approximate modal overlap factor from modal density and bandwidth."""

    density = modal_density_per_hz(frequency_hz, volume_m3, c=c)
    bandwidth = 2.2 / _positive(rt60_s, "rt60_s")
    return density * bandwidth


def critical_distance_for_target(
    dimensions_m: ArrayLike,
    target_rt60_s: float,
    *,
    directivity_factor: float = 1.0,
    c: float = 343.0,
) -> float:
    """Critical distance implied by a Sabine target RT60."""

    surfaces = shoebox_surface_areas(dimensions_m)
    alpha = target_mean_absorption_sabine(
        shoebox_volume(dimensions_m),
        float(np.sum(surfaces)),
        target_rt60_s,
        c=c,
    )
    return float(critical_distance(surfaces, alpha, directivity_factor=directivity_factor))


def room_constant_for_target(
    dimensions_m: ArrayLike,
    target_rt60_s: float,
    *,
    c: float = 343.0,
) -> float:
    """Room constant implied by a Sabine target RT60."""

    surfaces = shoebox_surface_areas(dimensions_m)
    alpha = target_mean_absorption_sabine(
        shoebox_volume(dimensions_m),
        float(np.sum(surfaces)),
        target_rt60_s,
        c=c,
    )
    return float(room_constant(surfaces, alpha))


__all__ = [
    "RoomDesignTarget",
    "absorption_budget_per_surface",
    "bolt_area_score",
    "count_modes_in_band",
    "critical_distance_for_target",
    "is_inside_bolt_area",
    "mean_absorption_to_reflection",
    "modal_density_per_hz",
    "modal_density_weyl",
    "modal_overlap_factor",
    "reflection_to_mean_absorption",
    "room_constant_for_target",
    "room_design_target",
    "room_ratio",
    "rt_constant",
    "target_absorption_area_sabine",
    "target_mean_absorption_eyring",
    "target_mean_absorption_sabine",
    "target_uniform_reflection_eyring",
    "target_uniform_reflection_sabine",
    "verify_target_rt60",
]
