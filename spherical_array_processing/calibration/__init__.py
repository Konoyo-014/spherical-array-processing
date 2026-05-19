"""Geometry calibration helpers for spherical microphone arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..coords import cart_to_sph, unit_sph_to_cart
from ..types import ArrayGeometry, SphericalGrid


@dataclass(frozen=True)
class RigidAlignment:
    """Rigid transform aligning one point cloud to another."""

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    scale: float
    rms_error: float


@dataclass(frozen=True)
class SphereFit:
    """Least-squares fitted sphere."""

    center: NDArray[np.float64]
    radius: float
    residuals: NDArray[np.float64]
    rms_error: float


def grid_to_cartesian(grid: SphericalGrid, *, radius: float = 1.0) -> NDArray[np.float64]:
    """Convert a spherical grid to Cartesian positions."""

    return unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention) * float(radius)


def cartesian_to_grid(points: ArrayLike, *, convention: str = "az_el") -> tuple[SphericalGrid, NDArray[np.float64]]:
    """Convert Cartesian points to a spherical grid."""

    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    az, angle2, radius = cart_to_sph(p[:, 0], p[:, 1], p[:, 2], convention=convention)
    return SphericalGrid(azimuth=az, angle2=angle2, weights=None, convention=convention), radius


def normalize_points(points: ArrayLike) -> NDArray[np.float64]:
    """Normalize Cartesian vectors to unit length."""

    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("points must be non-zero")
    return p / norm


def fit_sphere(points: ArrayLike) -> SphereFit:
    """Least-squares sphere fit from Cartesian points."""

    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 4:
        raise ValueError("points must have shape (N>=4, 3)")
    a = np.column_stack([2.0 * p, np.ones(p.shape[0])])
    b = np.sum(p * p, axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = sol[:3]
    radius = float(np.sqrt(max(0.0, sol[3] + np.dot(center, center))))
    residuals = np.linalg.norm(p - center[None, :], axis=1) - radius
    return SphereFit(
        center=center.astype(float),
        radius=radius,
        residuals=residuals.astype(float),
        rms_error=float(np.sqrt(np.mean(residuals * residuals))),
    )


def center_points(points: ArrayLike, center: ArrayLike | None = None) -> NDArray[np.float64]:
    """Subtract a point-cloud centre."""

    p = np.asarray(points, dtype=float)
    c = np.mean(p, axis=0) if center is None else np.asarray(center, dtype=float).reshape(3)
    return p - c[None, :]


def radial_errors(points: ArrayLike, *, center: ArrayLike | None = None, radius: float | None = None) -> NDArray[np.float64]:
    """Radial distance errors relative to a fitted or supplied sphere."""

    p = np.asarray(points, dtype=float)
    if center is None or radius is None:
        fit = fit_sphere(p)
        c = fit.center
        r = fit.radius
    else:
        c = np.asarray(center, dtype=float).reshape(3)
        r = float(radius)
    return np.linalg.norm(p - c[None, :], axis=1) - r


def radial_rms_error(points: ArrayLike, *, center: ArrayLike | None = None, radius: float | None = None) -> float:
    """RMS radial error relative to a sphere."""

    e = radial_errors(points, center=center, radius=radius)
    return float(np.sqrt(np.mean(e * e)))


def kabsch_alignment(
    source_points: ArrayLike,
    target_points: ArrayLike,
    *,
    allow_scale: bool = False,
) -> RigidAlignment:
    """Rigid Kabsch/Procrustes alignment from source to target points."""

    src = np.asarray(source_points, dtype=float)
    dst = np.asarray(target_points, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source_points and target_points must both have shape (N, 3)")
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    h = src0.T @ dst0
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    scale = 1.0
    if allow_scale:
        denom = float(np.sum(src0 * src0))
        if denom <= 0.0:
            raise ValueError("source_points have zero spread")
        scale = float(np.sum((src0 @ r.T) * dst0) / denom)
    t = dst_mean - scale * (r @ src_mean)
    aligned = apply_rigid_alignment(src, RigidAlignment(r, t, scale, 0.0))
    err = np.linalg.norm(aligned - dst, axis=1)
    return RigidAlignment(
        rotation=r.astype(float),
        translation=t.astype(float),
        scale=float(scale),
        rms_error=float(np.sqrt(np.mean(err * err))),
    )


def apply_rigid_alignment(points: ArrayLike, alignment: RigidAlignment) -> NDArray[np.float64]:
    """Apply a rigid alignment to Cartesian points."""

    p = np.asarray(points, dtype=float)
    return float(alignment.scale) * (p @ alignment.rotation.T) + alignment.translation[None, :]


def rotation_matrix_from_vectors(source: ArrayLike, target: ArrayLike) -> NDArray[np.float64]:
    """Shortest rotation matrix mapping one vector direction to another."""

    a = np.asarray(source, dtype=float).reshape(3)
    b = np.asarray(target, dtype=float).reshape(3)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        return -np.eye(3) + 2.0 * np.outer(v, v)
    s = float(np.linalg.norm(v))
    if s < 1e-15:
        return np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / (s * s))


def rotate_points(points: ArrayLike, rotation: ArrayLike) -> NDArray[np.float64]:
    """Rotate Cartesian points by a 3x3 matrix."""

    p = np.asarray(points, dtype=float)
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    return p @ r.T


def align_grid_to_grid(
    source_grid: SphericalGrid,
    target_grid: SphericalGrid,
    *,
    allow_scale: bool = False,
) -> RigidAlignment:
    """Estimate rigid alignment between two spherical grids."""

    src = grid_to_cartesian(source_grid)
    dst = grid_to_cartesian(target_grid)
    return kabsch_alignment(src, dst, allow_scale=allow_scale)


def apply_alignment_to_grid(
    grid: SphericalGrid,
    alignment: RigidAlignment,
    *,
    convention: str | None = None,
) -> SphericalGrid:
    """Apply a rigid alignment to a spherical grid and return directions."""

    points = apply_rigid_alignment(grid_to_cartesian(grid), alignment)
    out_grid, _radius = cartesian_to_grid(points, convention=grid.convention if convention is None else convention)
    return out_grid


def angular_position_errors(
    estimated_grid: SphericalGrid,
    reference_grid: SphericalGrid,
) -> NDArray[np.float64]:
    """Per-sensor angular errors between two same-size grids."""

    est = grid_to_cartesian(estimated_grid)
    ref = grid_to_cartesian(reference_grid)
    if est.shape != ref.shape:
        raise ValueError("grids must have the same size")
    dots = np.sum(est * ref, axis=1)
    dots[np.isclose(dots, 1.0, atol=1e-14)] = 1.0
    return np.arccos(np.clip(dots, -1.0, 1.0))


def angular_rms_error(
    estimated_grid: SphericalGrid,
    reference_grid: SphericalGrid,
) -> float:
    """RMS angular grid error in radians."""

    err = angular_position_errors(estimated_grid, reference_grid)
    return float(np.sqrt(np.mean(err * err)))


def array_geometry_from_points(
    points: ArrayLike,
    *,
    array_type: str = "rigid",
) -> ArrayGeometry:
    """Fit a spherical :class:`ArrayGeometry` from Cartesian sensor positions."""

    p = np.asarray(points, dtype=float)
    fit = fit_sphere(p)
    centered = p - fit.center[None, :]
    grid, radii = cartesian_to_grid(centered, convention="az_el")
    grid.weights = np.full(grid.size, 4.0 * np.pi / grid.size)
    return ArrayGeometry(
        radius_m=float(np.mean(radii)),
        sensor_grid=grid,
        array_type=array_type,  # type: ignore[arg-type]
        metadata={
            "fit_center_m": fit.center.tolist(),
            "fit_radius_m": fit.radius,
            "fit_rms_error_m": fit.rms_error,
        },
    )


def sensor_position_error_report(
    estimated_points: ArrayLike,
    reference_points: ArrayLike,
) -> dict[str, float]:
    """Small dictionary of Euclidean sensor-position error statistics."""

    est = np.asarray(estimated_points, dtype=float)
    ref = np.asarray(reference_points, dtype=float)
    if est.shape != ref.shape:
        raise ValueError("estimated_points and reference_points must match")
    err = np.linalg.norm(est - ref, axis=1)
    return {
        "mean_m": float(np.mean(err)),
        "median_m": float(np.median(err)),
        "max_m": float(np.max(err)),
        "rms_m": float(np.sqrt(np.mean(err * err))),
    }


__all__ = [
    "RigidAlignment",
    "SphereFit",
    "align_grid_to_grid",
    "angular_position_errors",
    "angular_rms_error",
    "apply_alignment_to_grid",
    "apply_rigid_alignment",
    "array_geometry_from_points",
    "cartesian_to_grid",
    "center_points",
    "fit_sphere",
    "grid_to_cartesian",
    "kabsch_alignment",
    "normalize_points",
    "radial_errors",
    "radial_rms_error",
    "rotate_points",
    "rotation_matrix_from_vectors",
    "sensor_position_error_report",
]
