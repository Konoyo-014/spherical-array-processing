"""Information-theoretic source-number estimators (AIC / MDL).

Given the eigenvalues of a sample SH covariance matrix and the number
of snapshots used to build it, Wax & Kailath's AIC and MDL criteria
pick the most likely number of spatially coherent sources by comparing
the geometric and arithmetic means of the trailing (noise-subspace)
eigenvalues.  The formulas below follow the original 1985 Wax-Kailath
paper, specialised to the SH covariance case where the signal
dimensionality is simply ``Q = (N+1)²``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


Criterion = Literal["aic", "mdl"]


def _neg_log_likelihood(
    eigvals: NDArray[np.float64], n_snapshots: int, k: int
) -> float:
    """Wax-Kailath ``-log L(k) = N (Q-k) · log[a(k) / g(k)]``.

    Uses the log-ratio of the arithmetic (``a``) to geometric (``g``)
    means of the trailing ``Q - k`` eigenvalues.  Always non-negative
    since ``a ≥ g`` by the AM-GM inequality; equals zero when all
    trailing eigenvalues coincide (pure noise).
    """
    q = eigvals.size
    if k >= q:
        return 0.0
    trailing = eigvals[k:]
    # Clip to a floor to avoid log(0); any true zero implies rank
    # deficiency, so treating it as a very small positive value gives
    # a well-defined, monotone criterion without overflowing.
    trailing = np.maximum(trailing, np.finfo(float).tiny)
    log_geo = np.mean(np.log(trailing))
    arith = np.mean(trailing)
    if arith <= 0:
        return 0.0
    return float(n_snapshots * (q - k) * (np.log(arith) - log_geo))


def estimate_n_sources(
    cov_or_eigvals: ArrayLike,
    n_snapshots: int,
    *,
    criterion: Criterion = "mdl",
    max_sources: int | None = None,
) -> int:
    """Estimate the number of coherent sources in an SH covariance.

    Parameters
    ----------
    cov_or_eigvals : array_like
        Either a ``(Q, Q)`` Hermitian SH covariance matrix or a 1-D
        array of its real-valued eigenvalues (any ordering — the
        function sorts them internally).
    n_snapshots : int
        Number of independent snapshots used to build the covariance.
        Must be positive.  For a single SHT bin averaged over ``T``
        STFT frames, this is ``T``.
    criterion : {"mdl", "aic"}, optional
        Selection criterion.  MDL (Minimum Description Length) is the
        Wax-Kailath default and tends to be consistent (returns the
        true source count as ``n_snapshots → ∞``); AIC (Akaike
        Information Criterion) is less penalising and often
        over-estimates at finite ``n_snapshots``.
    max_sources : int or None, optional
        Optional ceiling on the returned count.  Defaults to ``Q - 1``.

    Returns
    -------
    int
        The estimated number of sources ``k̂ ∈ [0, max_sources]``.

    References
    ----------
    .. [1] M. Wax and T. Kailath, "Detection of signals by information
       theoretic criteria", *IEEE Trans. Acoust. Speech Signal Process.*,
       33(2), 1985.
    """
    arr = np.asarray(cov_or_eigvals)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        eigvals = np.linalg.eigvalsh(arr).real
    elif arr.ndim == 1:
        eigvals = arr.real.astype(float, copy=True)
    else:
        raise ValueError(
            "cov_or_eigvals must be a square 2-D matrix or a 1-D eigenvalue array"
        )
    if n_snapshots <= 0:
        raise ValueError("n_snapshots must be positive")
    if criterion not in ("aic", "mdl"):
        raise ValueError(f"criterion must be 'aic' or 'mdl', got {criterion!r}")

    # Sort descending so that index k means "first k eigvals are signal".
    eigvals = np.sort(eigvals)[::-1]
    q = eigvals.size
    ceiling = q - 1 if max_sources is None else min(int(max_sources), q - 1)
    if ceiling < 0:
        raise ValueError("SH covariance must have at least 2 channels")

    best_k = 0
    best_score = np.inf
    log_n = float(np.log(n_snapshots))
    for k in range(0, ceiling + 1):
        # Penalty term — Wax-Kailath free-parameter count for the
        # maximum-likelihood fit: k(2Q - k) real parameters in the
        # signal-subspace covariance.
        penalty_free_params = float(k) * (2.0 * q - k)
        neg_log_l = _neg_log_likelihood(eigvals, int(n_snapshots), k)
        if criterion == "aic":
            score = 2.0 * neg_log_l + 2.0 * penalty_free_params
        else:  # mdl
            score = neg_log_l + 0.5 * penalty_free_params * log_n
        if score < best_score:
            best_score = score
            best_k = k
    return int(best_k)


__all__ = ["estimate_n_sources"]
