from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mvdr_weights(cov: ArrayLike, steering: ArrayLike, diagonal_loading: float = 1e-8) -> np.ndarray:
    """Compute Minimum Variance Distortionless Response (MVDR) beamformer weights.

    Parameters
    ----------
    cov : array_like, shape (M, M)
        Spatial covariance (or cross-spectral) matrix of the sensor signals.
    steering : array_like, shape (M,) or (M, K)
        Steering vector(s) pointing toward the look direction(s).  If 2-D,
        each column is treated as an independent steering vector and the
        returned weights have the same number of columns.
    diagonal_loading : float, optional
        Relative diagonal loading factor.  The regularisation added is
        ``diagonal_loading * trace(cov) / M * I``.  Default is 1e-8.

    Returns
    -------
    np.ndarray, shape (M,) or (M, K)
        Complex weight vector(s) satisfying the distortionless constraint
        ``w^H d = 1`` while minimising output power.

    Raises
    ------
    ValueError
        If *cov* is not a square 2-D matrix or if the dimensions of
        *steering* are incompatible.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.eye(4, dtype=complex)
    >>> d = np.ones(4, dtype=complex)
    >>> w = mvdr_weights(R, d)
    >>> w.shape
    (4,)
    >>> bool(abs(d.conj() @ w - 1.0) < 1e-10)
    True
    """
    r = np.asarray(cov, dtype=np.complex128)
    d = np.asarray(steering, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be square 2D matrix")
    if d.ndim == 1:
        d = d[:, None]
    if d.shape[0] != r.shape[0]:
        raise ValueError("steering length mismatch")
    rl = r + diagonal_loading * np.trace(r).real / max(r.shape[0], 1) * np.eye(r.shape[0], dtype=r.dtype)
    x = np.linalg.solve(rl, d)
    denom = np.sum(np.conj(d) * x, axis=0, keepdims=True)
    w = x / denom
    return w[:, 0] if w.shape[1] == 1 else w


def lcmv_weights(
    cov: ArrayLike,
    constraint_matrix: ArrayLike,
    response: ArrayLike,
    diagonal_loading: float = 1e-8,
) -> np.ndarray:
    """Compute Linearly Constrained Minimum Variance (LCMV) beamformer weights.

    Parameters
    ----------
    cov : array_like, shape (M, M)
        Spatial covariance matrix of the sensor signals.
    constraint_matrix : array_like, shape (M, K)
        Matrix whose columns are the constraint steering vectors.
    response : array_like, shape (K,)
        Desired response for each constraint (e.g., ``[1, 0]`` for a
        distortionless constraint in one direction and a null in another).
    diagonal_loading : float, optional
        Relative diagonal loading factor applied as
        ``diagonal_loading * trace(cov) / M * I``.  Default is 1e-8.

    Returns
    -------
    np.ndarray, shape (M,)
        Complex weight vector satisfying ``C^H w = f``.

    Raises
    ------
    ValueError
        If matrix dimensions are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.eye(4, dtype=complex)
    >>> C = np.eye(4, 2, dtype=complex)
    >>> f = np.array([1.0, 0.0])
    >>> w = lcmv_weights(R, C, f)
    >>> w.shape
    (4,)
    """
    r = np.asarray(cov, dtype=np.complex128)
    c = np.asarray(constraint_matrix, dtype=np.complex128)
    f = np.asarray(response, dtype=np.complex128).reshape(-1, 1)
    if c.ndim != 2:
        raise ValueError("constraint_matrix must be 2D")
    if c.shape[0] != r.shape[0]:
        raise ValueError("constraint rows must match cov size")
    if c.shape[1] != f.shape[0]:
        raise ValueError("response length mismatch")
    rl = r + diagonal_loading * np.trace(r).real / max(r.shape[0], 1) * np.eye(r.shape[0], dtype=r.dtype)
    rinv_c = np.linalg.solve(rl, c)
    gram = c.conj().T @ rinv_c
    lam = np.linalg.solve(gram, f)
    w = rinv_c @ lam
    return w[:, 0]

