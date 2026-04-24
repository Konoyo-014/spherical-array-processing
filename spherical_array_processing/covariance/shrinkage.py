"""Shrinkage, forward-backward averaging, and diagonal-loading
regularisers for sample covariance matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ledoit_wolf_shrinkage(
    snapshots: ArrayLike,
    *,
    return_shrinkage: bool = False,
) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], float]:
    """Ledoit-Wolf (2004) covariance shrinkage toward a scaled identity.

    Computes the sample covariance ``S`` of *snapshots* and shrinks it
    toward the target ``T = (tr(S) / M) · I`` with the optimal
    mean-squared-error weight

    .. math::

        \\hat{\\rho} = \\min\\left(1,\\
            \\frac{\\hat{\\pi}}{\\hat{\\gamma}}\\right), \\quad
        \\hat{\\Sigma} = (1-\\hat{\\rho})\\,S + \\hat{\\rho}\\,T.

    Here ``γ̂ = ||S − T||_F²`` measures how far the sample covariance
    is from the scaled-identity target and ``π̂ = (1/N²) Σ_t ||x_t
    x_t^H − S||_F²`` is the variance of the per-snapshot outer
    products (Ledoit-Wolf eq. 13).  The sample covariance is
    Hermitian-symmetrised on input; the returned matrix is
    Hermitian-symmetric to ~1e-15.

    Parameters
    ----------
    snapshots : array_like, shape (N, M)
        ``N`` snapshots of the ``M``-channel signal.  Complex data is
        accepted.
    return_shrinkage : bool, optional
        If ``True``, also return the estimated shrinkage weight
        ``ρ̂`` as a bonus output.

    Returns
    -------
    Σ̂ : ndarray, shape (M, M), complex128
        Shrunk covariance matrix.
    ρ̂ : float, optional
        The estimated shrinkage weight in ``[0, 1]``; only returned
        when ``return_shrinkage=True``.

    References
    ----------
    .. [1] O. Ledoit and M. Wolf, "A well-conditioned estimator for
       large-dimensional covariance matrices", *J. Multivariate
       Analysis*, 88(2), 2004.
    """
    x = np.asarray(snapshots, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError(f"snapshots must be 2-D (N, M); got {x.shape}")
    n, m = x.shape
    if n < 2:
        raise ValueError("Ledoit-Wolf shrinkage needs at least 2 snapshots")

    s = (x.conj().T @ x) / float(n)  # (M, M)
    s = 0.5 * (s + s.conj().T)  # enforce Hermitian symmetry

    mean_trace = np.real(np.trace(s)) / float(m)
    target = mean_trace * np.eye(m, dtype=np.complex128)

    # γ̂ = ||S - m·I||²_F  (not divided by M).
    diff = s - target
    gamma_hat = float(np.real(np.vdot(diff.ravel(), diff.ravel())))

    if gamma_hat <= 0:
        rho = 0.0
    else:
        # π̂ = (1/N²) Σ_t ||x_t x_t^H − S||²_F.  Expand the squared
        # Frobenius norm analytically to avoid the O(N·M²) outer-product
        # construction:
        #
        # ||A − S||²_F = ||A||²_F − 2·Re⟨A, S⟩_F + ||S||²_F,
        # with A = x_t x_t^H, ||A||²_F = (x_t^H x_t)² and
        # ⟨A, S⟩_F = tr(A^H S) = x_t^H S x_t (real because S is Hermitian).
        xH_x = np.einsum("ti,ti->t", x.conj(), x).real  # (N,)
        quad = np.einsum("ti,ij,tj->t", x.conj(), s, x).real  # x_t^H S x_t
        fro_s_sq = float(np.real(np.vdot(s.ravel(), s.ravel())))
        total = float(np.sum(xH_x ** 2) - 2.0 * np.sum(quad) + n * fro_s_sq)
        pi_hat = total / (float(n) ** 2)
        pi_hat = min(pi_hat, gamma_hat)
        rho = float(pi_hat / gamma_hat)

    sigma_hat = (1.0 - rho) * s + rho * target
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.conj().T)
    if return_shrinkage:
        return sigma_hat, rho
    return sigma_hat


def oas_shrinkage(
    cov: ArrayLike,
    n_snapshots: int,
    *,
    return_shrinkage: bool = False,
) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], float]:
    """Oracle Approximating Shrinkage (Chen et al. 2010).

    Needs only the sample covariance and the snapshot count, so it is
    drop-in for any existing MVDR / MUSIC / ESPRIT pipeline that
    already builds an SH covariance.  Shrinks ``R`` toward
    ``T = (tr(R)/M) · I`` with

    .. math::

        \\rho_\\text{OAS} = \\min\\left(1, \\frac{(1 - 2/M)\\,
        \\text{tr}(R^2) + \\text{tr}(R)^2}{(n + 1 - 2/M)\\,
        (\\text{tr}(R^2) - \\text{tr}(R)^2 / M)}\\right).

    Parameters
    ----------
    cov : array_like, shape (M, M)
        Sample covariance matrix.
    n_snapshots : int
        Number of snapshots used to build *cov*.
    return_shrinkage : bool, optional
        Return the estimated shrinkage weight alongside the matrix.

    References
    ----------
    .. [1] Y. Chen, A. Wiesel, A. O. Hero, "Shrinkage algorithms for
       MMSE covariance estimation", *IEEE Trans. Signal Process.*,
       58(10), 2010.
    """
    r = np.asarray(cov, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be a square matrix")
    if n_snapshots <= 0:
        raise ValueError("n_snapshots must be positive")
    m = r.shape[0]
    r = 0.5 * (r + r.conj().T)
    tr_r = np.real(np.trace(r))
    tr_r2 = np.real(np.trace(r @ r))
    denom = (n_snapshots + 1.0 - 2.0 / m) * (tr_r2 - tr_r ** 2 / m)
    if denom <= 0.0:
        rho = 1.0
    else:
        numer = (1.0 - 2.0 / m) * tr_r2 + tr_r ** 2
        rho = float(min(1.0, numer / denom))
    target = (tr_r / m) * np.eye(m, dtype=np.complex128)
    sigma_hat = (1.0 - rho) * r + rho * target
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.conj().T)
    if return_shrinkage:
        return sigma_hat, rho
    return sigma_hat


def forward_backward_average(
    cov: ArrayLike,
    exchange_matrix: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Forward-backward averaging: ``R_fb = 0.5 (R + J R* J)``.

    For symmetric arrays the exchange matrix ``J`` permutes sensors
    into their mirror images.  On SH-domain covariances the matrix is
    the real↔complex SH parity operator (``diag(ACN-parity)``).  When
    ``exchange_matrix`` is omitted, the default is the anti-diagonal
    ``J_ij = δ_{i + j, M-1}``, which decorrelates coherent sources
    for linear and SH-ACN arrays alike in practice.

    Halving the sample-covariance variance (and breaking coherence
    between correlated sources) makes MUSIC and ESPRIT more reliable
    on short snapshot records.

    Parameters
    ----------
    cov : array_like, shape (M, M)
        Sample covariance matrix.
    exchange_matrix : array_like or None, optional
        The ``J`` matrix.  Default: anti-diagonal unity.

    Returns
    -------
    ndarray, shape (M, M), complex128
        Forward-backward averaged covariance (Hermitian).
    """
    r = np.asarray(cov, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be a square matrix")
    m = r.shape[0]
    if exchange_matrix is None:
        j = np.fliplr(np.eye(m, dtype=np.complex128))
    else:
        j = np.asarray(exchange_matrix, dtype=np.complex128)
        if j.shape != r.shape:
            raise ValueError("exchange_matrix shape must match cov")
    out = 0.5 * (r + j @ r.conj() @ j)
    return 0.5 * (out + out.conj().T)


def diagonal_loading(
    cov: ArrayLike,
    loading: float | None = None,
    *,
    fraction_of_trace: float = 1e-3,
) -> NDArray[np.complex128]:
    """Add a small multiple of the identity to a sample covariance.

    If *loading* is ``None`` (default), the magnitude is set to
    ``fraction_of_trace * tr(cov) / M``, which is a trace-invariant
    scale that stays meaningful independent of the units of the input
    signal.  Supplying ``loading`` directly overrides this heuristic.

    Parameters
    ----------
    cov : array_like, shape (M, M)
        Sample covariance matrix.
    loading : float or None, optional
        Explicit ``λ`` in ``cov + λ I``.  Must be non-negative.
    fraction_of_trace : float, optional
        When *loading* is ``None``, this fraction of ``tr(cov)/M`` is
        used as the identity weight.  Default ``1e-3``.
    """
    r = np.asarray(cov, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be a square matrix")
    m = r.shape[0]
    if loading is None:
        scale = np.real(np.trace(r)) / m
        weight = float(fraction_of_trace) * scale
    else:
        weight = float(loading)
        if weight < 0:
            raise ValueError("loading must be non-negative")
    out = r + weight * np.eye(m, dtype=r.dtype)
    return 0.5 * (out + out.conj().T)
