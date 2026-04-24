"""Wigner-D rotation matrices for spherical-harmonic signals.

The ZYZ Euler-angle convention is used throughout: a rotation
``R(α, β, γ) = R_z(α) · R_y(β) · R_z(γ)`` acting on a function
``f`` produces ``f'(r̂) = f(R^{-1} r̂)`` whose orthonormal complex SH
coefficients transform as

``c'_{nm} = Σ_{m'} D^n_{m m'}(α, β, γ) · c_{n m'}``,

with ``D^n_{m m'}(α, β, γ) = e^{-i m α} · d^n_{m m'}(β) · e^{-i m' γ}``.

Real (ACN / tesseral) SH signals are rotated by conjugating the
corresponding complex Wigner-D block with the real↔complex basis
transformation used everywhere else in this package.
"""

from __future__ import annotations

from math import factorial, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import acn_index


SmallDMethod = Literal["sakurai", "jy"]


def _small_d_entry(n: int, m_row: int, m_col: int, beta: float) -> float:
    """Single entry d^n_{m_row, m_col}(β) of the Wigner small-d matrix.

    Uses the Sakurai summation; numerically stable for ``n ≲ 30``.
    """
    s_min = max(0, m_col - m_row)
    s_max = min(n + m_col, n - m_row)
    if s_min > s_max:
        return 0.0
    pre = sqrt(
        factorial(n + m_row)
        * factorial(n - m_row)
        * factorial(n + m_col)
        * factorial(n - m_col)
    )
    cos_half = np.cos(beta / 2.0)
    sin_half = np.sin(beta / 2.0)
    total = 0.0
    for s in range(s_min, s_max + 1):
        sign = (-1.0) ** (m_row - m_col + s)
        denom = (
            factorial(n + m_col - s)
            * factorial(s)
            * factorial(m_row - m_col + s)
            * factorial(n - m_row - s)
        )
        p_cos = 2 * n + m_col - m_row - 2 * s
        p_sin = m_row - m_col + 2 * s
        total += sign * pre / denom * (cos_half ** p_cos) * (sin_half ** p_sin)
    return float(total)


def _wigner_small_d_sakurai(n: int, beta: float) -> NDArray[np.float64]:
    """Direct Sakurai summation of ``d^n_{m' m}(β)``.

    Fast and exact at low order; suffers from cancellation around ``n ≈
    30`` (unitarity residual ~ 1e-9) and degrades further past ``n = 40``.
    """
    dim = 2 * n + 1
    out = np.zeros((dim, dim), dtype=float)
    for i, m_row in enumerate(range(-n, n + 1)):
        for j, m_col in enumerate(range(-n, n + 1)):
            out[i, j] = _small_d_entry(n, m_row, m_col, beta)
    return out


def _wigner_small_d_jy(n: int, beta: float) -> NDArray[np.float64]:
    """Compute ``d^n(β) = ⟨m'|exp(-i β J_y^{(n)})|m⟩`` by Hermitian
    diagonalization of the angular-momentum operator.

    The ``(2n+1)×(2n+1)`` operator ``J_y = (J_+ − J_-)/(2i)`` is
    tridiagonal with matrix elements

    ``⟨m|J_y|m+1⟩ = +i/2 · √((n-m)(n+m+1))``,

    obtained from ``⟨m+1|J_+|m⟩ = √((n-m)(n+m+1))`` and its Hermitian
    conjugate.  The spectrum is exactly ``{-n, …, +n}`` and ``J_y`` is
    well-conditioned for every finite ``n``, so ``numpy.linalg.eigh``
    recovers the eigensystem to near-machine precision.  The resulting
    matrix exponential is unitary to within ``~1e-14`` through at least
    ``n = 80``, while the direct Sakurai summation starts to lose
    precision past ``n ≈ 30``.
    """
    dim = 2 * n + 1
    if n == 0:
        return np.ones((1, 1), dtype=float)
    m_lower = np.arange(-n, n, dtype=float)  # length 2n, iterates m = -n..n-1
    # ⟨m|J_y|m+1⟩ = +i/2 · √((n-m)(n+m+1))
    upper = 0.5j * np.sqrt((n - m_lower) * (n + m_lower + 1.0))
    jy = np.zeros((dim, dim), dtype=np.complex128)
    idx = np.arange(dim - 1)
    jy[idx, idx + 1] = upper
    jy[idx + 1, idx] = np.conj(upper)
    eigvals, vecs = np.linalg.eigh(jy)
    phase = np.exp(-1j * float(beta) * eigvals)
    d_complex = (vecs * phase) @ vecs.conj().T
    # d^n is purely real; discard the machine-noise imaginary component.
    return np.asarray(np.real(d_complex), dtype=float)


def wigner_small_d(
    n: int,
    beta: float,
    *,
    method: SmallDMethod = "jy",
) -> NDArray[np.float64]:
    """Wigner small-d matrix ``d^n(β)`` of shape ``(2n+1, 2n+1)``.

    Rows and columns are indexed by ``m = -n, -n+1, ..., +n`` in ascending
    order (so row/column ``i`` corresponds to ``m = i - n``).

    Parameters
    ----------
    n : int
        SH degree (must be non-negative).
    beta : float
        Rotation angle about the body y-axis (ZYZ Euler convention), in
        radians.
    method : {"sakurai", "jy"}, optional
        Implementation to use.

        * ``"jy"`` (default since 0.4.0b12): diagonalize the ``J_y``
          angular-momentum operator and evaluate ``exp(-i β J_y)``.
          A few times slower per call than ``"sakurai"`` at low order
          but unitary to within ``~1e-14`` through at least ``n = 80``,
          so it is the safe default for general use.
        * ``"sakurai"``: direct single-index summation.  Fast and
          exact to machine precision at low order but starts to lose
          precision around ``n ≈ 30`` due to cancellation in the
          alternating sum.  Use only when you need the last few
          percent of speed at small ``n``.

    References
    ----------
    .. [1] T. Risbo, "Fourier transform summation of Legendre series and
       D-functions", J. Geodesy 70, 1996.
    .. [2] J. J. Sakurai, *Modern Quantum Mechanics*, 2nd ed.,
       Addison-Wesley, 1994, §3.6.
    """
    if n < 0:
        raise ValueError("degree n must be non-negative")
    if method == "sakurai":
        return _wigner_small_d_sakurai(n, float(beta))
    if method == "jy":
        return _wigner_small_d_jy(n, float(beta))
    raise ValueError(
        f"method must be 'sakurai' or 'jy', got {method!r}"
    )


def wigner_D(
    n: int,
    alpha: float,
    beta: float,
    gamma: float,
    *,
    method: SmallDMethod = "jy",
) -> NDArray[np.complex128]:
    """Complex Wigner-D matrix ``D^n(α, β, γ)`` of shape ``(2n+1, 2n+1)``.

    Uses the ZYZ convention
    ``D^n_{m m'}(α, β, γ) = e^{-imα} · d^n_{m m'}(β) · e^{-im'γ}``.

    Parameters
    ----------
    n : int
        SH degree.
    alpha, beta, gamma : float
        ZYZ Euler angles, radians.
    method : {"sakurai", "jy"}, optional
        Small-d backend.  See :func:`wigner_small_d`.
    """
    if n < 0:
        raise ValueError("degree n must be non-negative")
    d = wigner_small_d(n, beta, method=method)
    m_vec = np.arange(-n, n + 1)
    phase_alpha = np.exp(-1j * alpha * m_vec)  # row
    phase_gamma = np.exp(-1j * gamma * m_vec)  # column
    return phase_alpha[:, None] * d * phase_gamma[None, :]


def sh_rotation_matrix_complex(
    max_order: int,
    alpha: float,
    beta: float,
    gamma: float,
    *,
    method: SmallDMethod = "jy",
) -> NDArray[np.complex128]:
    """Block-diagonal rotation matrix for complex ACN-ordered SH coefficients.

    Returns a matrix ``R`` of shape ``((N+1)², (N+1)²)`` such that rotated
    coefficients are ``c_rot = R @ c_original``.  The matrix is block
    diagonal, with the n-th block equal to :func:`wigner_D`.

    Parameters
    ----------
    max_order : int
        Maximum SH order ``N``.
    alpha, beta, gamma : float
        ZYZ Euler angles, radians.
    method : {"sakurai", "jy"}, optional
        Small-d backend.  See :func:`wigner_small_d`.  Switch to
        ``"jy"`` for stable behaviour past ``N ≈ 25``.
    """
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    q = (max_order + 1) ** 2
    out = np.zeros((q, q), dtype=np.complex128)
    offset = 0
    for n in range(max_order + 1):
        dim = 2 * n + 1
        out[offset : offset + dim, offset : offset + dim] = wigner_D(
            n, alpha, beta, gamma, method=method
        )
        offset += dim
    return out


def _real_complex_transforms(max_order: int) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Build the (U, U^{-1}) pair that convert complex↔real SH coefficients.

    ``c_real = U · c_complex`` recovers the real-valued coefficients that
    correspond to a real (conjugate-symmetric) complex SH expansion.  The
    transformation is unitary (``U^{-1} = U^H``) on the real-signal
    subspace; inverting via ``np.linalg.inv`` works for the general
    coefficient rotation use case.
    """
    q = (max_order + 1) ** 2
    u = np.zeros((q, q), dtype=np.complex128)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for n in range(max_order + 1):
        # m = 0 — identity
        u[acn_index(n, 0), acn_index(n, 0)] = 1.0
        for m in range(1, n + 1):
            sgn = (-1.0) ** m
            row_pos = acn_index(n, m)
            row_neg = acn_index(n, -m)
            # c_real_{n+m} = √2 (-1)^m Re(c_complex_{n+m})
            #             = (-1)^m / √2 · (c_{n+m} + c_{n+m}*)
            # For a real-signal expansion, c_{n-m} = (-1)^m c_{n+m}*, so
            # Re(c_{n+m}) = 0.5 (c_{n+m} + c_{n-m} (-1)^m).
            # More generally, express c_real in terms of the full complex vector:
            #   c_real_{n+m}  = (-1)^m / √2 · c_{n+m} + (1/√2) · c_{n-m}    (m > 0)
            #   c_real_{n-m}  = +j · ((-1)^m / √2) · c_{n+m}  -  j · (1/√2) · c_{n-m}
            u[row_pos, acn_index(n, m)] = sgn * inv_sqrt2
            u[row_pos, acn_index(n, -m)] = inv_sqrt2
            u[row_neg, acn_index(n, m)] = 1j * sgn * inv_sqrt2
            u[row_neg, acn_index(n, -m)] = -1j * inv_sqrt2
    return u, np.linalg.inv(u)


def sh_rotation_matrix_real(
    max_order: int,
    alpha: float,
    beta: float,
    gamma: float,
    *,
    method: SmallDMethod = "jy",
) -> NDArray[np.float64]:
    """Block-diagonal real-valued rotation matrix for real (tesseral) SH.

    Computed via ``U · D_complex · U^{-1}`` where ``U`` is the
    complex→real coefficient transformation.  The imaginary part of the
    resulting matrix is zero to machine precision for any rotation.

    Parameters
    ----------
    max_order : int
        Maximum SH order.
    alpha, beta, gamma : float
        ZYZ Euler angles, radians.
    method : {"sakurai", "jy"}, optional
        Small-d backend; see :func:`wigner_small_d`.
    """
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    d_complex = sh_rotation_matrix_complex(
        max_order, alpha, beta, gamma, method=method
    )
    u, u_inv = _real_complex_transforms(max_order)
    r = u @ d_complex @ u_inv
    if np.max(np.abs(r.imag)) > 1e-9:
        # Numerical diagnostic: should be ~0 for any valid rotation.
        raise RuntimeError(
            f"real SH rotation has unexpected imaginary residual "
            f"{np.max(np.abs(r.imag)):.3e}; this suggests a bug in the "
            "complex↔real transformation."
        )
    return r.real


def sh_rotation_matrix(
    max_order: int,
    alpha: float,
    beta: float,
    gamma: float,
    basis: Literal["complex", "real"] = "complex",
    *,
    method: SmallDMethod = "jy",
) -> np.ndarray:
    """Return the SH-coefficient rotation matrix for the requested basis.

    Parameters
    ----------
    max_order : int
        Maximum SH order ``N``.
    alpha, beta, gamma : float
        ZYZ Euler angles in radians.  ``alpha`` rotates about z,
        ``beta`` about the new y, ``gamma`` about the final z.
    basis : {"complex", "real"}, optional
        Which SH convention the coefficient vector uses.
    method : {"sakurai", "jy"}, optional
        Small-d backend; see :func:`wigner_small_d`.

    Returns
    -------
    ndarray, shape ``((N+1)², (N+1)²)``
        Block-diagonal rotation matrix.  Multiplying an ACN-ordered
        coefficient vector by this matrix gives the coefficients of the
        rotated field.
    """
    if basis == "complex":
        return sh_rotation_matrix_complex(
            max_order, alpha, beta, gamma, method=method
        )
    if basis == "real":
        return sh_rotation_matrix_real(
            max_order, alpha, beta, gamma, method=method
        )
    raise ValueError(f"unsupported basis: {basis!r}")


def rotate_sh_coeffs(
    coeffs: ArrayLike,
    max_order: int,
    alpha: float,
    beta: float,
    gamma: float,
    basis: Literal["complex", "real"] = "complex",
    axis: int = -1,
    *,
    method: SmallDMethod = "jy",
) -> np.ndarray:
    """Rotate SH coefficient vectors along the requested axis.

    Parameters
    ----------
    coeffs : array_like
        SH coefficients with ``(N+1)²`` entries along *axis*.
    max_order : int
        Maximum SH order ``N``.
    alpha, beta, gamma : float
        ZYZ Euler angles in radians.
    basis : {"complex", "real"}, optional
        Coefficient convention.
    axis : int, optional
        Axis along which the SH coefficients are stored.  Default ``-1``.

    Returns
    -------
    ndarray
        Rotated coefficients with the same shape as *coeffs*.
    """
    c = np.asarray(coeffs)
    rot = sh_rotation_matrix(
        max_order, alpha, beta, gamma, basis=basis, method=method
    )
    c_m = np.moveaxis(c, axis, -1)
    expected = (max_order + 1) ** 2
    if c_m.shape[-1] != expected:
        raise ValueError(
            f"coefficient vector last axis has size {c_m.shape[-1]}, "
            f"expected (max_order+1)² = {expected}"
        )
    out = np.tensordot(c_m, rot.T, axes=([-1], [0]))
    return np.moveaxis(out, -1, axis)


def rotate_ambi_over_time(
    ambi_signal: ArrayLike,
    euler_zyz_rad: ArrayLike,
    *,
    max_order: int,
    basis: Literal["complex", "real"] = "real",
    block_samples: int = 480,
    crossfade_samples: int | None = None,
    method: SmallDMethod = "jy",
) -> NDArray[np.floating]:
    """Apply time-varying SH rotation to an ambisonic signal.

    Intended for head-tracked binaural rendering: divide *ambi_signal*
    into ``K`` equal blocks of ``block_samples`` samples, apply
    rotation ``R_k`` to block ``k``, and linearly crossfade adjacent
    blocks over ``crossfade_samples`` samples at each boundary so the
    transition is click-free.

    The crossfade uses the standard "matrix-lerp" trick
    ``y = (1 − α)·R_{k−1}·x + α·R_k·x`` which is equivalent to
    linearly interpolating the rotation matrices, sample-by-sample.
    Those interpolated matrices are generally not themselves perfectly
    orthogonal / unitary, so energy preservation inside the crossfade
    is only approximate.  For the intended head-tracking use case,
    adjacent keyframes are close enough that this behaves like a cheap,
    perceptually smooth approximation to a per-sample slerp.

    Parameters
    ----------
    ambi_signal : array_like, shape ``(Q, T)`` or ``(T, Q)``
        Ambisonic signal with ``Q = (max_order+1)²`` channels in ACN
        ordering.  The channel axis is detected automatically (it is
        the axis whose length matches ``(N+1)²``); if both axes look
        plausible, the shorter axis is assumed to be channels.
    euler_zyz_rad : array_like, shape ``(K, 3)`` or ``(3,)``
        Per-block ZYZ Euler angles in radians.  A single ``(3,)``
        vector is interpreted as a static rotation applied uniformly
        (no crossfade needed).
    max_order : int
        Maximum SH order ``N``.
    basis : {"complex", "real"}, optional
        SH basis of *ambi_signal*.  Defaults to ``"real"``.
    block_samples : int, optional
        Audio samples per rotation keyframe.  Default ``480`` (10 ms
        at 48 kHz, typical head-tracker update period).
    crossfade_samples : int, optional
        Linear-crossfade length in samples at each block boundary.
        The blend occupies the first ``crossfade_samples`` samples of
        each new block.
        Default ``block_samples // 8``.  Must satisfy
        ``0 ≤ crossfade_samples ≤ block_samples``.
    method : {"sakurai", "jy"}, optional
        Small-d backend.  Defaults to ``"jy"`` for numerical stability
        past ``N ≈ 25``.

    Returns
    -------
    ndarray, same shape and dtype kind as *ambi_signal*.
        Rotated ambisonic signal.  When *ambi_signal* has more than
        ``K · block_samples`` samples, the trailing tail is rotated
        with ``R_{K-1}``.
    """
    a = np.asarray(ambi_signal)
    if a.ndim != 2:
        raise ValueError(
            f"ambi_signal must be 2-D (Q, T) or (T, Q); got {a.shape}"
        )
    q_expected = (int(max_order) + 1) ** 2
    if a.shape[0] == q_expected and a.shape[1] != q_expected:
        channels_first = True
        sig = a
    elif a.shape[1] == q_expected and a.shape[0] != q_expected:
        channels_first = False
        sig = a.T
    elif a.shape[0] == q_expected and a.shape[1] == q_expected:
        # Square input — assume (Q, T) by convention.
        channels_first = True
        sig = a
    else:
        raise ValueError(
            f"ambi_signal must have one axis of length (max_order+1)² = "
            f"{q_expected}; got shape {a.shape}"
        )
    q, n_samples = sig.shape

    angles = np.asarray(euler_zyz_rad, dtype=float)
    if angles.ndim == 1:
        if angles.shape != (3,):
            raise ValueError(
                f"single-rotation euler_zyz_rad must have shape (3,); "
                f"got {angles.shape}"
            )
        rot = sh_rotation_matrix(
            int(max_order),
            float(angles[0]),
            float(angles[1]),
            float(angles[2]),
            basis=basis,
            method=method,
        )
        out = rot @ sig
        return out if channels_first else out.T
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(
            f"euler_zyz_rad must have shape (K, 3) or (3,); "
            f"got {angles.shape}"
        )

    block = int(block_samples)
    if block <= 0:
        raise ValueError("block_samples must be positive")
    if crossfade_samples is None:
        crossfade = max(1, block // 8)
    else:
        crossfade = int(crossfade_samples)
    if not 0 <= crossfade <= block:
        raise ValueError(
            "crossfade_samples must lie in [0, block_samples]"
        )

    k = angles.shape[0]
    covered = k * block
    if n_samples < covered:
        raise ValueError(
            f"signal length {n_samples} is shorter than the requested "
            f"{k} × {block} = {covered} samples of rotation keyframes"
        )

    out_dtype = np.result_type(sig.dtype, np.float64)
    out = np.empty((q, n_samples), dtype=out_dtype)

    # Precompute all K rotation matrices once.
    rots = np.stack(
        [
            sh_rotation_matrix(
                int(max_order),
                float(angles[i, 0]),
                float(angles[i, 1]),
                float(angles[i, 2]),
                basis=basis,
                method=method,
            )
            for i in range(k)
        ],
        axis=0,
    )

    for i in range(k):
        start = i * block
        stop = start + block
        block_sig = sig[:, start:stop]
        rotated_now = rots[i] @ block_sig
        if i == 0 or crossfade == 0:
            out[:, start:stop] = rotated_now
            continue
        # Crossfade first ``crossfade`` samples of block i from
        # R_{i-1} to R_i.
        head = block_sig[:, :crossfade]
        rotated_prev_head = rots[i - 1] @ head
        alpha = np.linspace(
            1.0 / crossfade, 1.0, crossfade, dtype=float
        )
        out[:, start : start + crossfade] = (
            (1.0 - alpha) * rotated_prev_head
            + alpha * rotated_now[:, :crossfade]
        )
        out[:, start + crossfade : stop] = rotated_now[:, crossfade:]

    # Trailing samples beyond K·block use R_{K-1}.
    if n_samples > covered:
        out[:, covered:] = rots[-1] @ sig[:, covered:]

    return out if channels_first else out.T


__all__ = [
    "rotate_ambi_over_time",
    "rotate_sh_coeffs",
    "sh_rotation_matrix",
    "sh_rotation_matrix_complex",
    "sh_rotation_matrix_real",
    "wigner_D",
    "wigner_small_d",
]
