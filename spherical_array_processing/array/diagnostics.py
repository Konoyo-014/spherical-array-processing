"""Sampling-grid and spherical-array diagnostic helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..coords import unit_sph_to_cart
from ..sh import matrix as sh_matrix
from ..types import ArrayGeometry, SHBasisSpec, SphericalGrid
from .sampling import spatial_aliasing_frequency


@dataclass(frozen=True)
class GridDiagnostics:
    """Numerical diagnostics for a spherical sampling grid."""

    n_points: int
    weight_sum: float
    min_weight: float | None
    max_weight: float | None
    min_angular_distance_rad: float
    mean_nearest_neighbor_rad: float
    max_nearest_neighbor_rad: float
    quadrature_condition: float
    gram_error_fro: float
    gram_error_max: float


@dataclass(frozen=True)
class ArrayDiagnostics:
    """Geometry and modal-sampling diagnostics for a microphone array."""

    n_sensors: int
    radius_m: float
    max_order: int
    aliasing_frequency_hz: float
    recommended_order: int
    sampling_condition: float
    rank: int
    underdetermined: bool
    aperture_diameter_m: float


def grid_cartesian(grid: SphericalGrid) -> NDArray[np.float64]:
    """Return unit Cartesian direction vectors for a spherical grid."""

    return unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention)


def pairwise_angular_distance_matrix(grid: SphericalGrid) -> NDArray[np.float64]:
    """Pairwise great-circle distances between all grid directions."""

    xyz = grid_cartesian(grid)
    dots = np.clip(xyz @ xyz.T, -1.0, 1.0)
    out = np.arccos(dots)
    np.fill_diagonal(out, 0.0)
    return out


def nearest_neighbor_angles(grid: SphericalGrid) -> NDArray[np.float64]:
    """Nearest-neighbour angular distance for every grid point."""

    if grid.size < 2:
        raise ValueError("grid must contain at least two points")
    d = pairwise_angular_distance_matrix(grid)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def grid_min_angular_distance(grid: SphericalGrid) -> float:
    """Smallest non-zero angular separation in a grid."""

    return float(np.min(nearest_neighbor_angles(grid)))


def grid_covering_radius(grid: SphericalGrid) -> float:
    """Approximate covering radius by the worst nearest-neighbour angle."""

    return float(np.max(nearest_neighbor_angles(grid)))


def grid_weight_sum(grid: SphericalGrid) -> float:
    """Sum quadrature weights, or ``4*pi`` for an unweighted grid."""

    if grid.weights is None:
        return float(4.0 * np.pi)
    return float(np.sum(np.asarray(grid.weights, dtype=float)))


def normalized_grid_weights(grid: SphericalGrid) -> NDArray[np.float64]:
    """Return weights normalised to integrate constants to ``4*pi``."""

    if grid.weights is None:
        return np.full(grid.size, 4.0 * np.pi / grid.size, dtype=float)
    w = np.asarray(grid.weights, dtype=float).reshape(-1)
    if w.shape != (grid.size,):
        raise ValueError("grid weights must match grid size")
    if np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
        raise ValueError("grid weights must be non-negative with positive sum")
    return w * (4.0 * np.pi / float(np.sum(w)))


def sh_sampling_matrix(
    grid: SphericalGrid,
    max_order: int,
    *,
    basis: str = "real",
    normalization: str = "orthonormal",
) -> NDArray:
    """Evaluate a spherical-harmonic sampling matrix for diagnostics."""

    spec = SHBasisSpec(
        max_order=int(max_order),
        basis=basis,  # type: ignore[arg-type]
        normalization=normalization,  # type: ignore[arg-type]
        angle_convention=grid.convention,
    )
    return np.asarray(sh_matrix(spec, grid))


def sh_weighted_gram(
    grid: SphericalGrid,
    max_order: int,
    *,
    basis: str = "real",
    normalization: str = "orthonormal",
) -> NDArray:
    """Weighted SH Gram matrix ``Y^H W Y``."""

    y = sh_sampling_matrix(
        grid,
        max_order,
        basis=basis,
        normalization=normalization,
    )
    w = normalized_grid_weights(grid)
    return np.asarray(y).conj().T @ (w[:, None] * y)


def sh_gram_error(
    grid: SphericalGrid,
    max_order: int,
    *,
    basis: str = "real",
    normalization: str = "orthonormal",
) -> NDArray:
    """Quadrature Gram error relative to the expected identity scale."""

    gram = sh_weighted_gram(
        grid,
        max_order,
        basis=basis,
        normalization=normalization,
    )
    if normalization == "orthonormal":
        expected_scale = 1.0
    elif normalization in ("sn3d", "n3d"):
        expected_scale = 4.0 * np.pi
    else:
        raise ValueError("normalization must be 'orthonormal', 'sn3d', or 'n3d'")
    return gram - expected_scale * np.eye(gram.shape[0], dtype=gram.dtype)


def sh_condition_number(
    grid: SphericalGrid,
    max_order: int,
    *,
    basis: str = "real",
    normalization: str = "orthonormal",
    weighted: bool = True,
) -> float:
    """Condition number of the SH sampling matrix."""

    y = sh_sampling_matrix(
        grid,
        max_order,
        basis=basis,
        normalization=normalization,
    )
    if weighted:
        w = normalized_grid_weights(grid)
        y = np.sqrt(w)[:, None] * y
    return float(np.linalg.cond(y))


def sh_sampling_rank(
    grid: SphericalGrid,
    max_order: int,
    *,
    basis: str = "real",
    rtol: float = 1e-10,
) -> int:
    """Numerical rank of a grid's SH sampling matrix."""

    y = sh_sampling_matrix(grid, max_order, basis=basis)
    s = np.linalg.svd(y, compute_uv=False)
    if s.size == 0:
        return 0
    return int(np.count_nonzero(s > float(rtol) * s[0]))


def exact_quadrature_order(
    grid: SphericalGrid,
    *,
    max_search_order: int = 12,
    tolerance: float = 1e-8,
    basis: str = "real",
    normalization: str = "orthonormal",
) -> int:
    """Largest SH order whose weighted Gram error stays below tolerance."""

    best = -1
    for order in range(int(max_search_order) + 1):
        err = sh_gram_error(
            grid,
            order,
            basis=basis,
            normalization=normalization,
        )
        if float(np.max(np.abs(err))) <= float(tolerance):
            best = order
        else:
            break
    return best


def grid_diagnostics(
    grid: SphericalGrid,
    max_order: int,
    *,
    basis: str = "real",
    normalization: str = "orthonormal",
) -> GridDiagnostics:
    """Compute a compact diagnostic report for a sampling grid."""

    nn = nearest_neighbor_angles(grid)
    err = sh_gram_error(
        grid,
        max_order,
        basis=basis,
        normalization=normalization,
    )
    weights = None if grid.weights is None else normalized_grid_weights(grid)
    return GridDiagnostics(
        n_points=int(grid.size),
        weight_sum=grid_weight_sum(grid),
        min_weight=None if weights is None else float(np.min(weights)),
        max_weight=None if weights is None else float(np.max(weights)),
        min_angular_distance_rad=float(np.min(nn)),
        mean_nearest_neighbor_rad=float(np.mean(nn)),
        max_nearest_neighbor_rad=float(np.max(nn)),
        quadrature_condition=sh_condition_number(
            grid,
            max_order,
            basis=basis,
            normalization=normalization,
        ),
        gram_error_fro=float(np.linalg.norm(err)),
        gram_error_max=float(np.max(np.abs(err))),
    )


def recommended_order_from_sensor_count(n_sensors: int) -> int:
    """Conservative full-3D SH order supported by a sensor count."""

    n = int(n_sensors)
    if n < 1:
        raise ValueError("n_sensors must be positive")
    return max(0, int(np.floor(np.sqrt(n))) - 1)


def array_aperture_diameter(array: ArrayGeometry) -> float:
    """Aperture diameter of a spherical array geometry."""

    if float(array.radius_m) <= 0.0:
        raise ValueError("array radius_m must be positive")
    return float(2.0 * array.radius_m)


def array_spatial_aliasing_frequency(
    array: ArrayGeometry,
    max_order: int,
    *,
    c: float = 343.0,
) -> float:
    """Spatial-aliasing frequency for a spherical array and SH order."""

    return spatial_aliasing_frequency(array.radius_m, max_order, c=c)


def array_sampling_matrix(
    array: ArrayGeometry,
    max_order: int,
    *,
    basis: str = "real",
    normalization: str = "orthonormal",
) -> NDArray:
    """SH sampling matrix at the array sensor directions."""

    return sh_sampling_matrix(
        array.sensor_grid,
        max_order,
        basis=basis,
        normalization=normalization,
    )


def array_sampling_condition(
    array: ArrayGeometry,
    max_order: int,
    *,
    basis: str = "real",
    normalization: str = "orthonormal",
) -> float:
    """Condition number of an array's SH sampling matrix."""

    return sh_condition_number(
        array.sensor_grid,
        max_order,
        basis=basis,
        normalization=normalization,
    )


def array_is_order_supported(
    array: ArrayGeometry,
    max_order: int,
    *,
    require_overdetermined: bool = False,
) -> bool:
    """Whether a sensor count can support a full-3D SH order."""

    needed = (int(max_order) + 1) ** 2
    return array.n_sensors > needed if require_overdetermined else array.n_sensors >= needed


def array_diagnostics(
    array: ArrayGeometry,
    max_order: int,
    *,
    c: float = 343.0,
    basis: str = "real",
) -> ArrayDiagnostics:
    """Compute basic spherical-array sampling diagnostics."""

    y = array_sampling_matrix(array, max_order, basis=basis)
    rank = int(np.linalg.matrix_rank(y))
    return ArrayDiagnostics(
        n_sensors=int(array.n_sensors),
        radius_m=float(array.radius_m),
        max_order=int(max_order),
        aliasing_frequency_hz=array_spatial_aliasing_frequency(array, max_order, c=c),
        recommended_order=recommended_order_from_sensor_count(array.n_sensors),
        sampling_condition=array_sampling_condition(array, max_order, basis=basis),
        rank=rank,
        underdetermined=bool(array.n_sensors < (int(max_order) + 1) ** 2),
        aperture_diameter_m=array_aperture_diameter(array),
    )


def sensor_distance_matrix(array: ArrayGeometry) -> NDArray[np.float64]:
    """Euclidean sensor-distance matrix in metres."""

    xyz = grid_cartesian(array.sensor_grid) * float(array.radius_m)
    diff = xyz[:, None, :] - xyz[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def sensor_angular_distance_matrix(array: ArrayGeometry) -> NDArray[np.float64]:
    """Angular sensor-distance matrix in radians."""

    return pairwise_angular_distance_matrix(array.sensor_grid)


def modal_noise_gain(
    array: ArrayGeometry,
    max_order: int,
    *,
    basis: str = "real",
    rcond: float = 1e-12,
) -> NDArray[np.float64]:
    """Per-mode least-squares noise gain for SH analysis on the array."""

    y = array_sampling_matrix(array, max_order, basis=basis)
    pinv = np.linalg.pinv(y, rcond=float(rcond))
    return np.sqrt(np.sum(np.abs(pinv) ** 2, axis=1)).astype(float)


def modal_noise_gain_db(
    array: ArrayGeometry,
    max_order: int,
    *,
    basis: str = "real",
    rcond: float = 1e-12,
) -> NDArray[np.float64]:
    """Per-mode least-squares noise gain in decibels."""

    gain = modal_noise_gain(array, max_order, basis=basis, rcond=rcond)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(gain)


__all__ = [
    "ArrayDiagnostics",
    "GridDiagnostics",
    "array_aperture_diameter",
    "array_diagnostics",
    "array_is_order_supported",
    "array_sampling_condition",
    "array_sampling_matrix",
    "array_spatial_aliasing_frequency",
    "exact_quadrature_order",
    "grid_cartesian",
    "grid_covering_radius",
    "grid_diagnostics",
    "grid_min_angular_distance",
    "grid_weight_sum",
    "modal_noise_gain",
    "modal_noise_gain_db",
    "nearest_neighbor_angles",
    "normalized_grid_weights",
    "pairwise_angular_distance_matrix",
    "recommended_order_from_sensor_count",
    "sensor_angular_distance_matrix",
    "sensor_distance_matrix",
    "sh_condition_number",
    "sh_gram_error",
    "sh_sampling_matrix",
    "sh_sampling_rank",
    "sh_weighted_gram",
]
