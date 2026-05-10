"""SH-domain beamformer steering utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sh import matrix as _sh_matrix
from ..types import SHBasisSpec, SphericalGrid


def steer_sh_weights(
    b_n: ArrayLike,
    look_azimuth: float,
    look_angle2: float,
    basis: SHBasisSpec,
) -> NDArray[np.complex128]:
    """Build SH-domain weights for a steered axisymmetric beam pattern."""
    weights_by_order = np.asarray(b_n, dtype=float).reshape(-1)
    max_order = weights_by_order.size - 1
    if max_order != basis.max_order:
        raise ValueError(
            f"len(b_n)={weights_by_order.size} implies max_order={max_order}, "
            f"but basis.max_order={basis.max_order}"
        )

    look_grid = SphericalGrid(
        azimuth=np.array([look_azimuth], dtype=float),
        angle2=np.array([look_angle2], dtype=float),
        weights=np.array([4.0 * np.pi], dtype=float),
        convention=basis.angle_convention,
    )
    y_look = np.asarray(_sh_matrix(basis, look_grid))[0]
    expanded = np.empty(basis.n_coeffs, dtype=float)
    cursor = 0
    for degree in range(max_order + 1):
        count = 2 * degree + 1
        expanded[cursor: cursor + count] = weights_by_order[degree]
        cursor += count
    return np.asarray(expanded * y_look, dtype=np.complex128)


def beamform_sh(
    sh_signals: ArrayLike,
    weights: ArrayLike,
) -> NDArray[np.complex128]:
    """Apply SH-domain beamforming weights along the last axis."""
    sig = np.asarray(sh_signals, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if sig.shape[-1] != w.size:
        raise ValueError("last axis of sh_signals must match weights length")
    return sig @ w


__all__ = ["steer_sh_weights", "beamform_sh"]
