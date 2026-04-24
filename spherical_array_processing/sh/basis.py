from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # SciPy < 1.15: sph_harm(m, n, theta_az, phi_colat)
    from scipy.special import sph_harm as _scipy_sph_harm
except ImportError:
    # SciPy >= 1.15: sph_harm_y(n, m, theta_colat, phi_az)
    from scipy.special import sph_harm_y as _sph_harm_y  # type: ignore

    def _scipy_sph_harm(m: int, n: int, az: float, colat: float) -> complex:  # type: ignore[misc]
        """Shim mapping the legacy sph_harm(m,n,az,colat) call convention
        to the new sph_harm_y(n,m,colat,az) API."""
        return _sph_harm_y(n, m, colat, az)  # type: ignore[return-value]

from ..coords import azel_to_az_colat
from ..types import SHBasisSpec, SphericalGrid


def acn_index(n: int, m: int) -> int:
    """Return the Ambisonic Channel Number (ACN) for degree *n* and order *m*.

    Parameters
    ----------
    n : int
        Spherical harmonic degree (n >= 0).
    m : int
        Spherical harmonic order (-n <= m <= n).

    Returns
    -------
    int
        ACN index, computed as ``n * (n + 1) + m``.

    Examples
    --------
    >>> acn_index(0, 0)
    0
    >>> acn_index(1, -1)
    1
    >>> acn_index(2, 2)
    8
    """
    return n * (n + 1) + m


def acn_to_nm(acn: int | ArrayLike) -> tuple[int, int] | tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return spherical harmonic degree/order pairs from ACN indices.

    Parameters
    ----------
    acn : int or array_like
        Ambisonic Channel Number index or indices.

    Returns
    -------
    n, m : int or ndarray
        Degree and order satisfying ``acn = n * (n + 1) + m``.

    Examples
    --------
    >>> acn_to_nm(3)
    (1, 1)
    """
    idx = np.asarray(acn, dtype=np.int64)
    if np.any(idx < 0):
        raise ValueError("ACN indices must be non-negative")
    n = np.floor(np.sqrt(idx)).astype(np.int64)
    m = idx - n * (n + 1)
    if idx.ndim == 0:
        return int(n), int(m)
    return n, m


def degree_order_pairs(max_order: int) -> NDArray[np.int64]:
    """Return ``(n, m)`` pairs in ACN order up to *max_order*.

    Parameters
    ----------
    max_order : int
        Maximum spherical harmonic degree.

    Returns
    -------
    ndarray, shape ((max_order + 1)**2, 2)
        Rows are ``[n, m]`` in ACN order.

    Examples
    --------
    >>> degree_order_pairs(1).tolist()
    [[0, 0], [1, -1], [1, 0], [1, 1]]
    """
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    return np.asarray([(n, m) for n in range(max_order + 1) for m in range(-n, n + 1)], dtype=np.int64)


def _norm_scale(n: int, normalization: str) -> float:
    if normalization == "orthonormal":
        return 1.0
    if normalization == "sn3d":
        return np.sqrt(4.0 * np.pi / (2 * n + 1))
    if normalization == "n3d":
        return np.sqrt(4.0 * np.pi)
    raise ValueError(f"unsupported normalization: {normalization}")


def _grid_to_az_colat(grid: SphericalGrid) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if grid.convention == "az_colat":
        return grid.azimuth, grid.angle2
    return azel_to_az_colat(grid.azimuth, grid.angle2)


def complex_matrix(
    spec: SHBasisSpec,
    grid: SphericalGrid,
) -> NDArray[np.complex128]:
    """Compute the complex spherical harmonics matrix for the given grid.

    Evaluates complex spherical harmonics Y_n^m at every grid point,
    applying the normalization specified in *spec*.

    Parameters
    ----------
    spec : SHBasisSpec
        Specification object containing ``max_order``, ``normalization``
        (``"orthonormal"``, ``"sn3d"``, or ``"n3d"``), ``basis``
        (should be ``"complex"``), and ``angle_convention``.
    grid : SphericalGrid
        Grid of directions with ``.azimuth``, ``.angle2``, ``.convention``,
        and ``.size`` attributes.

    Returns
    -------
    ndarray, shape (n_points, n_coeffs)
        Complex spherical harmonics matrix where ``n_coeffs = (max_order+1)**2``.

    Examples
    --------
    >>> import numpy as np
    >>> from spherical_array_processing.types import SHBasisSpec, SphericalGrid
    >>> spec = SHBasisSpec(max_order=1, basis="complex", normalization="orthonormal")
    >>> grid = SphericalGrid(azimuth=np.array([0.0]), angle2=np.array([np.pi/2]))
    >>> Y = complex_matrix(spec, grid)
    >>> Y.shape
    (1, 4)
    """
    if spec.angle_convention not in {"az_el", "az_colat"}:
        raise ValueError(f"unsupported angle convention: {spec.angle_convention}")
    az, colat = _grid_to_az_colat(grid)
    y = np.zeros((grid.size, spec.n_coeffs), dtype=np.complex128)
    for n in range(spec.max_order + 1):
        s = _norm_scale(n, spec.normalization)
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            y[:, idx] = _scipy_sph_harm(m, n, az, colat) * s
    return y


def real_matrix(spec: SHBasisSpec, grid: SphericalGrid) -> NDArray[np.float64]:
    """Compute the real-valued spherical harmonics matrix for the given grid.

    Derives the real SH matrix from the complex matrix using the standard
    real-to-complex conversion (Condon-Shortley phase).

    Parameters
    ----------
    spec : SHBasisSpec
        Specification with ``basis="real"`` (complex is used internally).
    grid : SphericalGrid
        Grid of directions.

    Returns
    -------
    ndarray, shape (n_points, n_coeffs)
        Real-valued spherical harmonics matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from spherical_array_processing.types import SHBasisSpec, SphericalGrid
    >>> spec = SHBasisSpec(max_order=1, basis="real", normalization="sn3d")
    >>> grid = SphericalGrid(azimuth=np.array([0.0]), angle2=np.array([np.pi/2]))
    >>> Y = real_matrix(spec, grid)
    >>> Y.shape
    (1, 4)
    >>> np.isrealobj(Y)
    True
    """
    yc = complex_matrix(
        SHBasisSpec(
            max_order=spec.max_order,
            basis="complex",
            normalization=spec.normalization,
            angle_convention=spec.angle_convention,
        ),
        grid,
    )
    yr = np.zeros_like(yc.real)
    for n in range(spec.max_order + 1):
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            if m < 0:
                yr[:, idx] = np.sqrt(2.0) * ((-1) ** m) * yc[:, acn_index(n, -m)].imag
            elif m == 0:
                yr[:, idx] = yc[:, idx].real
            else:
                yr[:, idx] = np.sqrt(2.0) * ((-1) ** m) * yc[:, idx].real
    return yr


def matrix(spec: SHBasisSpec, grid: SphericalGrid) -> NDArray[np.complex128] | NDArray[np.float64]:
    """Return the SH basis matrix (real or complex) according to *spec*.

    Parameters
    ----------
    spec : SHBasisSpec
        Specification object; ``spec.basis`` selects ``"real"`` or ``"complex"``.
    grid : SphericalGrid
        Grid of directions.

    Returns
    -------
    ndarray, shape (n_points, n_coeffs)
        Complex or real SH basis matrix depending on ``spec.basis``.

    Examples
    --------
    >>> import numpy as np
    >>> from spherical_array_processing.types import SHBasisSpec, SphericalGrid
    >>> spec = SHBasisSpec(max_order=0, basis="complex", normalization="orthonormal")
    >>> grid = SphericalGrid(azimuth=np.array([0.0]), angle2=np.array([0.0]))
    >>> matrix(spec, grid).shape
    (1, 1)
    """
    if spec.basis == "complex":
        return complex_matrix(spec, grid)
    if spec.basis == "real":
        return real_matrix(spec, grid)
    raise ValueError(f"unsupported basis: {spec.basis}")


def replicate_per_order(values: ArrayLike) -> NDArray[np.float64]:
    """Replicate each value (2n+1) times to expand per-order values to per-coefficient.

    Parameters
    ----------
    values : array_like, shape (N+1,)
        One value per SH order, where ``values[n]`` corresponds to order *n*.

    Returns
    -------
    ndarray, shape ((N+1)**2,)
        Expanded array where each ``values[n]`` is repeated ``2*n + 1`` times.

    Examples
    --------
    >>> import numpy as np
    >>> replicate_per_order([10, 20, 30])
    array([10., 20., 20., 20., 30., 30., 30., 30., 30.])
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    out = []
    for n, v in enumerate(vals):
        out.extend([v] * (2 * n + 1))
    return np.asarray(out, dtype=float)


def complex_to_real_coeffs(coeffs: ArrayLike, max_order: int, axis: int = -1) -> NDArray[np.float64]:
    """Convert complex SH coefficients to their real-valued counterparts.

    Parameters
    ----------
    coeffs : array_like
        Complex SH coefficient array with ``(max_order+1)**2`` entries
        along *axis*.
    max_order : int
        Maximum SH order.
    axis : int, optional
        Axis along which the SH coefficients are stored. Default is -1.

    Returns
    -------
    ndarray
        Real-valued SH coefficients with the same shape as *coeffs*.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array([1.0 + 0j, 0.0 + 0j, 0.5 + 0j, 0.0 + 0j])
    >>> r = complex_to_real_coeffs(c, max_order=1)
    >>> np.isrealobj(r)
    True
    >>> r.shape
    (4,)
    """
    c = np.asarray(coeffs, dtype=np.complex128)
    c = np.moveaxis(c, axis, -1)
    if c.shape[-1] != (max_order + 1) ** 2:
        raise ValueError("last axis does not match max_order")
    r = np.zeros(c.shape, dtype=float)
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            if m < 0:
                r[..., idx] = -np.sqrt(2.0) * ((-1) ** m) * c[..., acn_index(n, -m)].imag
            elif m == 0:
                r[..., idx] = c[..., idx].real
            else:
                r[..., idx] = np.sqrt(2.0) * ((-1) ** m) * c[..., idx].real
    return np.moveaxis(r, -1, axis)


def real_to_complex_coeffs(coeffs: ArrayLike, max_order: int, axis: int = -1) -> NDArray[np.complex128]:
    """Inverse of `complex_to_real_coeffs` under the real-field symmetry assumption.

    Parameters
    ----------
    coeffs : array_like
        Real-valued SH coefficient array with ``(max_order+1)**2`` entries
        along *axis*.
    max_order : int
        Maximum SH order.
    axis : int, optional
        Axis along which the SH coefficients are stored. Default is -1.

    Returns
    -------
    ndarray
        Complex SH coefficients with the same shape as *coeffs*.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.array([1.0, 0.0, 0.5, 0.0])
    >>> c = real_to_complex_coeffs(r, max_order=1)
    >>> c.shape
    (4,)
    >>> np.iscomplexobj(c)
    True
    """
    r = np.asarray(coeffs, dtype=float)
    r = np.moveaxis(r, axis, -1)
    if r.shape[-1] != (max_order + 1) ** 2:
        raise ValueError("last axis does not match max_order")
    c = np.zeros(r.shape, dtype=np.complex128)
    for n in range(max_order + 1):
        c[..., acn_index(n, 0)] = r[..., acn_index(n, 0)]
        for m in range(1, n + 1):
            rp = r[..., acn_index(n, m)]
            rn = r[..., acn_index(n, -m)]
            cpos = ((-1) ** m) / np.sqrt(2.0) * (rp - 1j * rn)
            cneg = (1.0 / np.sqrt(2.0)) * (rp + 1j * rn)
            c[..., acn_index(n, m)] = cpos
            c[..., acn_index(n, -m)] = cneg
    return np.moveaxis(c, -1, axis)
