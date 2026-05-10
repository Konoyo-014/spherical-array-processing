"""SH-domain covariance estimation and pre-processing helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def estimate_sh_cov(sh_snapshots: ArrayLike) -> NDArray[np.complex128]:
    """Estimate a Hermitian SH-domain sample covariance matrix."""
    snapshots = np.asarray(sh_snapshots, dtype=np.complex128)
    if snapshots.ndim != 2:
        raise ValueError(f"sh_snapshots must be 2-D, got shape {snapshots.shape}")
    n_snapshots, n_channels = snapshots.shape
    if n_snapshots < n_channels:
        snapshots = snapshots.T
        n_snapshots, n_channels = snapshots.shape
    cov = (snapshots.conj().T @ snapshots) / float(n_snapshots)
    return np.asarray((cov + cov.conj().T) / 2.0, dtype=np.complex128)


def forward_backward_cov(R: ArrayLike) -> NDArray[np.complex128]:
    """Apply forward-backward averaging to a square covariance matrix."""
    cov = np.asarray(R, dtype=np.complex128)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"R must be square 2-D, got shape {cov.shape}")
    exchange = np.eye(cov.shape[0], dtype=float)[::-1]
    averaged = (cov + exchange @ cov.conj() @ exchange) / 2.0
    return np.asarray((averaged + averaged.conj().T) / 2.0, dtype=np.complex128)


def diagonal_loading(
    R: ArrayLike,
    load: float | None = None,
    relative: bool = True,
) -> NDArray[np.complex128]:
    """Add diagonal loading to a covariance matrix."""
    cov = np.asarray(R, dtype=np.complex128)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"R must be square 2-D, got shape {cov.shape}")
    if load is None:
        load = 1e-4
    if relative:
        delta = float(load) * float(np.trace(cov).real) / max(cov.shape[0], 1)
    else:
        delta = float(load)
    return cov + delta * np.eye(cov.shape[0], dtype=np.complex128)


__all__ = ["estimate_sh_cov", "forward_backward_cov", "diagonal_loading"]
