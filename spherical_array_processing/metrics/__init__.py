"""Cross-module spatial-audio evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from ..coords import angular_distance, unit_sph_to_cart


DirectionConvention = Literal["az_el", "az_colat"]


@dataclass(frozen=True)
class DirectionErrorReport:
    """Summary of angular localization errors."""

    errors_rad: NDArray[np.float64]
    errors_deg: NDArray[np.float64]
    mean_deg: float
    median_deg: float
    max_deg: float
    rmse_deg: float


@dataclass(frozen=True)
class DirectionAssignment:
    """Optimal assignment between estimated and reference directions."""

    estimate_indices: NDArray[np.int64]
    reference_indices: NDArray[np.int64]
    errors_rad: NDArray[np.float64]
    cost: float


def _directions(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("direction vectors must have length 2")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("directions must have shape (N, 2)")
    return arr


def wrap_azimuth_rad(azimuth: ArrayLike) -> float | NDArray[np.float64]:
    """Wrap azimuth to ``[-pi, pi)``."""

    out = (np.asarray(azimuth, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    if out.ndim == 0 or out.size == 1:
        return float(out.reshape(-1)[0])
    return out


def angular_error_rad(
    estimates_rad: ArrayLike,
    references_rad: ArrayLike,
    *,
    convention: DirectionConvention = "az_el",
) -> NDArray[np.float64]:
    """Great-circle angular error between paired direction rows."""

    est = _directions(estimates_rad)
    ref = _directions(references_rad)
    if est.shape != ref.shape:
        raise ValueError("estimates and references must have matching shape")
    return angular_distance(est[:, 0], est[:, 1], ref[:, 0], ref[:, 1], convention=convention)


def angular_error_deg(
    estimates_rad: ArrayLike,
    references_rad: ArrayLike,
    *,
    convention: DirectionConvention = "az_el",
) -> NDArray[np.float64]:
    """Great-circle angular error in degrees."""

    return np.degrees(angular_error_rad(estimates_rad, references_rad, convention=convention))


def direction_error_report(
    estimates_rad: ArrayLike,
    references_rad: ArrayLike,
    *,
    convention: DirectionConvention = "az_el",
) -> DirectionErrorReport:
    """Bundle common angular-error statistics."""

    err_rad = angular_error_rad(estimates_rad, references_rad, convention=convention)
    err_deg = np.degrees(err_rad)
    return DirectionErrorReport(
        errors_rad=err_rad,
        errors_deg=err_deg,
        mean_deg=float(np.mean(err_deg)),
        median_deg=float(np.median(err_deg)),
        max_deg=float(np.max(err_deg)),
        rmse_deg=float(np.sqrt(np.mean(err_deg * err_deg))),
    )


def angular_distance_matrix(
    estimates_rad: ArrayLike,
    references_rad: ArrayLike,
    *,
    convention: DirectionConvention = "az_el",
) -> NDArray[np.float64]:
    """Pairwise angular distances between two direction sets."""

    est = _directions(estimates_rad)
    ref = _directions(references_rad)
    return angular_distance(
        est[:, None, 0],
        est[:, None, 1],
        ref[None, :, 0],
        ref[None, :, 1],
        convention=convention,
    )


def assign_directions(
    estimates_rad: ArrayLike,
    references_rad: ArrayLike,
    *,
    convention: DirectionConvention = "az_el",
) -> DirectionAssignment:
    """Minimum-cost one-to-one assignment of estimated to reference directions."""

    cost = angular_distance_matrix(estimates_rad, references_rad, convention=convention)
    rows, cols = linear_sum_assignment(cost)
    errors = cost[rows, cols]
    return DirectionAssignment(
        estimate_indices=rows.astype(np.int64),
        reference_indices=cols.astype(np.int64),
        errors_rad=errors.astype(float),
        cost=float(np.sum(errors)),
    )


def matched_direction_error_report(
    estimates_rad: ArrayLike,
    references_rad: ArrayLike,
    *,
    convention: DirectionConvention = "az_el",
) -> DirectionErrorReport:
    """Angular-error report after optimal direction assignment."""

    assignment = assign_directions(estimates_rad, references_rad, convention=convention)
    return direction_error_report(
        _directions(estimates_rad)[assignment.estimate_indices],
        _directions(references_rad)[assignment.reference_indices],
        convention=convention,
    )


def resultant_vector(
    directions_rad: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    convention: DirectionConvention = "az_el",
) -> NDArray[np.float64]:
    """Weighted Cartesian resultant vector for a direction set."""

    dirs = _directions(directions_rad)
    xyz = unit_sph_to_cart(dirs[:, 0], dirs[:, 1], convention=convention)
    if weights is None:
        w = np.ones(dirs.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != (dirs.shape[0],):
            raise ValueError("weights must match number of directions")
    if float(np.sum(np.abs(w))) <= 0.0:
        raise ValueError("weights must not sum to zero magnitude")
    return np.sum(w[:, None] * xyz, axis=0)


def resultant_length(
    directions_rad: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    convention: DirectionConvention = "az_el",
) -> float:
    """Normalised resultant-vector length in ``[0, 1]`` for non-negative weights."""

    dirs = _directions(directions_rad)
    if weights is None:
        w = np.ones(dirs.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
    vec = resultant_vector(dirs, weights=w, convention=convention)
    denom = float(np.sum(w))
    if denom <= 0.0:
        raise ValueError("weights must have positive sum")
    return float(np.linalg.norm(vec) / denom)


def energy_vector(
    directions_rad: ArrayLike,
    energies: ArrayLike,
    *,
    convention: DirectionConvention = "az_el",
) -> NDArray[np.float64]:
    """Gerzon-style energy vector from loudspeaker directions and energies."""

    e = np.asarray(energies, dtype=float).reshape(-1)
    if np.any(e < 0.0) or float(np.sum(e)) <= 0.0:
        raise ValueError("energies must be non-negative with positive sum")
    return resultant_vector(directions_rad, weights=e, convention=convention) / float(np.sum(e))


def velocity_vector(
    directions_rad: ArrayLike,
    amplitudes: ArrayLike,
    *,
    convention: DirectionConvention = "az_el",
) -> NDArray[np.float64]:
    """Velocity vector from loudspeaker directions and signed amplitudes."""

    a = np.asarray(amplitudes, dtype=float).reshape(-1)
    denom = float(np.sum(a))
    if abs(denom) <= 1e-15:
        raise ValueError("amplitudes must have non-zero sum")
    return resultant_vector(directions_rad, weights=a, convention=convention) / denom


def vector_direction_error_deg(vector: ArrayLike, reference_direction_rad: ArrayLike, *, convention: DirectionConvention = "az_el") -> float:
    """Angular error between a Cartesian vector and a reference direction."""

    v = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError("vector must be non-zero")
    ref = _directions(reference_direction_rad)
    ref_xyz = unit_sph_to_cart(ref[:, 0], ref[:, 1], convention=convention)[0]
    dot = np.clip(float(np.dot(v / norm, ref_xyz)), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def normalized_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Complex-safe normalized inner-product magnitude."""

    a = np.asarray(x)
    b = np.asarray(y)
    if a.shape != b.shape:
        raise ValueError("x and y must have matching shape")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        raise ValueError("inputs must have non-zero norm")
    return float(np.abs(np.vdot(a, b)) / denom)


def scale_invariant_error_db(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Scale-invariant error in dB after optimal scalar gain."""

    ref = np.asarray(reference)
    est = np.asarray(estimate)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have matching shape")
    denom = np.vdot(est, est)
    if abs(denom) <= 0.0:
        raise ValueError("estimate must have non-zero energy")
    gain = np.vdot(est, ref) / denom
    err = ref - gain * est
    ratio = float(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-15))
    if ratio <= 0.0:
        return -float("inf")
    return float(20.0 * np.log10(ratio))


def log_spectral_distance_db(
    reference_magnitude: ArrayLike,
    estimate_magnitude: ArrayLike,
    *,
    floor_db: float = -120.0,
) -> float:
    """Root-mean-square log-spectral distance in dB."""

    ref = np.asarray(reference_magnitude, dtype=float)
    est = np.asarray(estimate_magnitude, dtype=float)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have matching shape")
    floor = 10.0 ** (float(floor_db) / 20.0)
    ref_db = 20.0 * np.log10(np.maximum(np.abs(ref), floor))
    est_db = 20.0 * np.log10(np.maximum(np.abs(est), floor))
    return float(np.sqrt(np.mean((ref_db - est_db) ** 2)))


def magnitude_response_error_db(reference: ArrayLike, estimate: ArrayLike) -> NDArray[np.float64]:
    """Pointwise magnitude-response error ``estimate - reference`` in dB."""

    ref = np.asarray(reference)
    est = np.asarray(estimate)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have matching shape")
    return 20.0 * np.log10(np.maximum(np.abs(est), 1e-15) / np.maximum(np.abs(ref), 1e-15))


def phase_error_rad(reference: ArrayLike, estimate: ArrayLike) -> NDArray[np.float64]:
    """Wrapped phase error between two complex responses."""

    ref = np.asarray(reference)
    est = np.asarray(estimate)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have matching shape")
    return np.angle(est * np.conj(ref))


def rms_error(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Root-mean-square error."""

    ref = np.asarray(reference)
    est = np.asarray(estimate)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have matching shape")
    return float(np.sqrt(np.mean(np.abs(est - ref) ** 2)))


def relative_error(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Relative Euclidean error norm."""

    ref = np.asarray(reference)
    est = np.asarray(estimate)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have matching shape")
    denom = float(np.linalg.norm(ref))
    if denom <= 0.0:
        raise ValueError("reference must have non-zero norm")
    return float(np.linalg.norm(est - ref) / denom)


__all__ = [
    "DirectionAssignment",
    "DirectionConvention",
    "DirectionErrorReport",
    "angular_distance_matrix",
    "angular_error_deg",
    "angular_error_rad",
    "assign_directions",
    "direction_error_report",
    "energy_vector",
    "log_spectral_distance_db",
    "magnitude_response_error_db",
    "matched_direction_error_report",
    "normalized_correlation",
    "phase_error_rad",
    "relative_error",
    "resultant_length",
    "resultant_vector",
    "rms_error",
    "scale_invariant_error_db",
    "vector_direction_error_deg",
    "velocity_vector",
    "wrap_azimuth_rad",
]
