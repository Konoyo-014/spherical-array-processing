from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..types import SphericalGrid


def direct_sht(
    samples: ArrayLike,
    y_matrix: ArrayLike,
    grid: SphericalGrid | None = None,
    weights: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Direct weighted spherical harmonic transform via weighted least squares.

    Computes the spherical-harmonic coefficient vector ``c`` that minimizes
    the weighted residual energy

    ``||diag(sqrt(w)) @ (Y @ c - x)||_2``.

    For exact quadrature grids this reduces to the usual weighted projection.
    On approximate grids such as Fibonacci sampling, this avoids the bias of
    using the simple adjoint ``Y^H W`` as if the basis were exactly orthonormal.

    Parameters
    ----------
    samples : array_like, shape (..., n_points)
        Spatial samples on the grid. Leading dimensions are preserved.
    y_matrix : array_like, shape (n_points, n_coeffs)
        SH basis matrix evaluated at the grid points (from
        :func:`~spherical_array_processing.sh.basis.matrix`).
    grid : SphericalGrid or None, optional
        If provided and *weights* is ``None``, ``grid.weights`` is used.
    weights : array_like or None, optional
        Quadrature weights of length ``n_points``. When ``None`` and *grid*
        is also ``None``, uniform weights ``1/n_points`` are used.

    Returns
    -------
    ndarray, shape (..., n_coeffs)
        Spherical harmonic coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> n_pts, n_coeffs = 6, 4
    >>> Y = np.eye(n_pts, n_coeffs)
    >>> samples = np.ones(n_pts)
    >>> c = direct_sht(samples, Y)
    >>> c.shape
    (4,)
    """
    x = np.asarray(samples)
    y = np.asarray(y_matrix)
    if y.ndim != 2:
        raise ValueError("y_matrix must be 2D")
    n_points = y.shape[0]
    if x.shape[-1] != n_points:
        raise ValueError("samples last axis must equal number of grid points")
    if weights is None and grid is not None:
        weights = grid.weights
    if weights is None:
        w = np.ones(n_points, dtype=float) / n_points
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != n_points:
            raise ValueError("weights length mismatch")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.any(w > 0.0):
        raise ValueError("at least one weight must be positive")
    sqrt_w = np.sqrt(w)
    a = sqrt_w[:, None] * y
    b = np.asarray(x * sqrt_w, dtype=np.result_type(x, y))
    a_pinv = np.linalg.pinv(a)
    return np.tensordot(b, a_pinv.T, axes=([-1], [0]))


def inverse_sht(
    coeffs: ArrayLike,
    y_matrix: ArrayLike,
) -> NDArray[np.complex128]:
    """Inverse spherical harmonic transform: synthesise spatial samples from SH coefficients.

    Parameters
    ----------
    coeffs : array_like, shape (..., n_coeffs)
        SH coefficient vectors.
    y_matrix : array_like, shape (n_points, n_coeffs)
        SH basis matrix (same as used for the forward transform).

    Returns
    -------
    NDArray, shape (..., n_points)
        Reconstructed spatial samples on the grid.

    Examples
    --------
    >>> import numpy as np
    >>> n_pts, n_coeffs = 6, 4
    >>> Y = np.eye(n_pts, n_coeffs)
    >>> coeffs = np.array([1.0, 2.0, 3.0, 4.0])
    >>> s = inverse_sht(coeffs, Y)
    >>> s.shape
    (6,)
    """
    c = np.asarray(coeffs)
    y = np.asarray(y_matrix)
    if y.ndim != 2:
        raise ValueError("y_matrix must be 2D")
    if c.shape[-1] != y.shape[1]:
        raise ValueError("coeffs last axis must equal number of SH coefficients")
    return np.tensordot(c, y.T, axes=([-1], [0]))
