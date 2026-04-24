from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh
from scipy.special import eval_legendre


def beam_weights_cardioid(order: int) -> np.ndarray:
    """Axisymmetric spherical coefficients for (1+cos(theta))^N normalized to unit front gain.

    Parameters
    ----------
    order : int
        Beamformer order *N*.  Higher orders yield a narrower main lobe.

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Axisymmetric SH coefficients ``b_n`` for degrees 0 to *order*.

    Examples
    --------
    >>> w = beam_weights_cardioid(1)
    >>> w.shape
    (2,)
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    return _project_cardioid_axisymmetric(order)


def beam_weights_hypercardioid(order: int) -> np.ndarray:
    """Maximum DI axisymmetric weights (max-directivity / hypercardioid).

    The maximum-directivity beamformer has DI = (N+1)^2.  In the SH
    per-order gain convention used by :func:`axisymmetric_pattern`, the
    optimal weights are *uniform*: ``b_n = 4*pi / (N+1)^2`` for all n.

    Parameters
    ----------
    order : int
        Beamformer order.

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Axisymmetric SH coefficients that maximise the directivity index.

    References
    ----------
    .. [1] B. Rafaely, *Fundamentals of Spherical Array Processing*,
       2nd ed., Springer, 2019, §6.3.

    Examples
    --------
    >>> w = beam_weights_hypercardioid(2)
    >>> w.shape
    (3,)
    """
    # Max-DI per-order gains are uniform in the (2n+1)/(4π)·P_n basis.
    # Note: the Polaris MATLAB formula d_n = (2n+1)/(N+1)² is in the
    # *plain Legendre* convention; converting to our basis via
    # b_n = d_n · 4π/(2n+1) gives the constant 4π/(N+1)².
    b = np.full(order + 1, 4.0 * math.pi / (order + 1) ** 2)
    # Front gain is already 1: sum b_n·(2n+1)/(4π) = (N+1)²/(N+1)² = 1
    return b


def beam_weights_supercardioid(order: int) -> np.ndarray:
    """Axisymmetric weights for the supercardioid (max front-back ratio) pattern.

    The design maximizes the front-to-back hemisphere power ratio in the same
    axisymmetric basis used by :func:`axisymmetric_pattern`.

    Parameters
    ----------
    order : int
        Beamformer order.

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Axisymmetric SH coefficients optimised for maximum front-to-back
        energy ratio.

    Examples
    --------
    >>> import numpy as np
    >>> w = beam_weights_supercardioid(1)
    >>> w.shape
    (2,)
    """
    b = _design_supercardioid_numerical(order)
    # Normalise to unit front gain through axisymmetric_pattern
    front = sum(b[n] * (2 * n + 1) / (4 * math.pi) for n in range(len(b)))
    if front != 0:
        b /= front
    return b


def beam_weights_maxev(order: int) -> np.ndarray:
    """Axisymmetric weights that maximise the energy vector (max-rE taper).

    Implements the Zotter & Frank (2012) max-rE design, matching the
    Polaris/Politis MATLAB reference ``beamWeightsMaxEV.m``.  The weights
    are Legendre polynomials evaluated at ``cos(2.4068 / (N + 1.51))``,
    normalised so that ``axisymmetric_pattern(0, w) == 1`` (unit front
    gain), consistent with `beam_weights_cardioid`.

    Parameters
    ----------
    order : int
        Beamformer order.

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Axisymmetric SH coefficients (one per SH degree 0 … *order*).

    References
    ----------
    .. [1] F. Zotter and M. Frank, "All-Round Ambisonic Panning and
       Decoding", *J. Audio Eng. Soc.*, 60(10), 2012, eq. (10).

    Examples
    --------
    >>> w = beam_weights_maxev(3)
    >>> w.shape
    (4,)
    """
    theta0 = 2.4068 / (order + 1.51)
    cos_theta0 = math.cos(theta0)
    # Legendre polynomial values at the max-rE angle
    b = np.array([float(eval_legendre(n, cos_theta0)) for n in range(order + 1)])
    # Normalise so that axisymmetric_pattern(0, b) == 1
    front_gain = sum(b[n] * (2 * n + 1) / (4 * math.pi) for n in range(order + 1))
    if front_gain != 0:
        b /= front_gain
    return b


def normalize_axisymmetric_weights(weights: ArrayLike) -> np.ndarray:
    """Normalize per-order axisymmetric weights to unit front gain.

    The convention matches :func:`axisymmetric_pattern`, where the front gain is
    ``sum_n b_n * (2n + 1) / (4*pi)``.
    """
    b = np.asarray(weights, dtype=float).reshape(-1)
    front_gain = sum(b[n] * (2 * n + 1) / (4 * math.pi) for n in range(b.size))
    if front_gain == 0:
        raise ValueError("cannot normalize weights with zero front gain")
    return b / front_gain


def beam_weights_butterworth(order: int, filter_order: float, cutoff_order: float) -> np.ndarray:
    """Spatial Butterworth taper for axisymmetric SH beamforming.

    Parameters
    ----------
    order : int
        Maximum SH order.
    filter_order : float
        Butterworth slope parameter.
    cutoff_order : float
        Cutoff SH order.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if filter_order <= 0:
        raise ValueError("filter_order must be positive")
    if cutoff_order <= 0:
        raise ValueError("cutoff_order must be positive")
    n = np.arange(order + 1, dtype=float)
    weights = 1.0 / np.sqrt(1.0 + (n / float(cutoff_order)) ** (2.0 * float(filter_order)))
    return normalize_axisymmetric_weights(weights)


def beam_weights_dolph_chebyshev(
    order: int,
    design_parameter: float,
    design_criterion: str = "sidelobe",
) -> np.ndarray:
    """Spherical Dolph-Chebyshev beamformer weights.

    Implements the Koretz & Rafaely (2009) mapping of a Dolph-Chebyshev
    polynomial of order ``2N`` onto axisymmetric SH weights, producing a
    beam pattern of the form ``T_{2N}(x0 cos(θ/2)) / R`` normalised to unit
    front gain.  The returned weights are already passed through
    :func:`normalize_axisymmetric_weights`.

    Parameters
    ----------
    order : int
        Maximum SH order ``N`` (must be ≥ 1).
    design_parameter : float
        Interpreted according to *design_criterion*:

        * ``"sidelobe"`` (default): linear side-lobe ratio ``R = mainlobe
          peak / peak side-lobe``.  Must satisfy ``R ≥ 1``; e.g. ``R = 10``
          gives 20 dB rejection.
        * ``"mainlobe"``: main-lobe *half-width* in radians, i.e. the
          angle of the **first null** measured from the main-beam axis.
          The first null symmetric pair is therefore at ``±design_parameter``
          so the null-to-null full width is ``2·design_parameter``.
    design_criterion : {"sidelobe", "mainlobe"}, optional
        Dolph-Chebyshev design criterion.

    Returns
    -------
    np.ndarray, shape (order + 1,)
        Axisymmetric SH coefficients normalised to unit front gain.

    References
    ----------
    .. [1] A. Koretz and B. Rafaely, "Dolph–Chebyshev beampattern design
       for spherical arrays", *IEEE Trans. Signal Process.*, 57(6), 2009.
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    if design_parameter <= 0:
        raise ValueError("design_parameter must be positive")

    m_total = 2 * order
    if design_criterion == "sidelobe":
        ratio = float(design_parameter)
        if ratio < 1.0:
            raise ValueError("sidelobe design_parameter must be >= 1")
        x0 = np.cosh(np.arccosh(ratio) / m_total)
    elif design_criterion == "mainlobe":
        width = float(design_parameter)
        x0 = np.cos(np.pi / (2.0 * m_total)) / np.cos(width / 2.0)
        if x0 < 1.0:
            raise ValueError("mainlobe width is too large for this order")
        ratio = np.cosh(m_total * np.arccosh(x0))
    else:
        raise ValueError("design_criterion must be 'sidelobe' or 'mainlobe'")

    cheb = _chebyshev_coefficients_ascending(2 * order)
    legendre = np.zeros((order + 1, order + 1), dtype=float)
    for n in range(order + 1):
        coeffs = _legendre_coefficients_ascending(n)
        legendre[: coeffs.size, n] = coeffs

    weights = np.zeros(order + 1, dtype=float)
    for n in range(order + 1):
        total = 0.0
        for i in range(n + 1):
            for j in range(order + 1):
                for m in range(j + 1):
                    total += (
                        (1.0 - (-1.0) ** (m + i + 1))
                        / (m + i + 1)
                        * math.factorial(j)
                        / (math.factorial(m) * math.factorial(j - m))
                        * (0.5**j)
                        * cheb[2 * j]
                        * legendre[i, n]
                        * (x0 ** (2 * j))
                    )
        weights[n] = (2.0 * math.pi / ratio) * total
    return normalize_axisymmetric_weights(weights)


def axisymmetric_pattern(theta: ArrayLike, b_n: ArrayLike) -> np.ndarray:
    """Evaluate an axisymmetric beam pattern from its SH coefficients at angles *theta*.

    Parameters
    ----------
    theta : array_like
        Polar angles (colatitude) in radians at which to evaluate the
        pattern.
    b_n : array_like
        Axisymmetric SH coefficients for degrees 0 to ``len(b_n) - 1``,
        as returned by the ``beam_weights_*`` functions.

    Returns
    -------
    np.ndarray
        Real-valued beam pattern sampled at each angle in *theta*.

    Examples
    --------
    >>> import numpy as np
    >>> b = beam_weights_hypercardioid(1)
    >>> p = axisymmetric_pattern(np.array([0.0, np.pi]), b)
    >>> p.shape
    (2,)
    """
    theta = np.asarray(theta, dtype=float)
    b = np.asarray(b_n, dtype=float).reshape(-1)
    x = np.cos(theta)
    out = np.zeros_like(theta, dtype=float)
    for n, bn in enumerate(b):
        out = out + bn * ((2 * n + 1) / (4 * np.pi)) * eval_legendre(n, x)
    return out


def _legendre_coefficients_ascending(order: int) -> np.ndarray:
    return np.polynomial.legendre.Legendre.basis(order).convert(kind=np.polynomial.Polynomial).coef


def _chebyshev_coefficients_ascending(order: int) -> np.ndarray:
    return np.polynomial.chebyshev.Chebyshev.basis(order).convert(kind=np.polynomial.Polynomial).coef


def _project_cardioid_axisymmetric(order: int) -> np.ndarray:
    """Project ((1 + x) / 2)^N onto the axisymmetric SH basis exactly.

    For cardioid weights the target is a degree-N polynomial in ``x = cos(theta)``.
    The coefficient integrand ``target(x) * P_n(x)`` therefore has degree at most
    ``2N``.  Gauss-Legendre quadrature with ``N + 1`` nodes integrates it exactly,
    which is substantially more stable than fitting on a fixed dense grid for
    higher orders.
    """
    x_gl, w_gl = np.polynomial.legendre.leggauss(order + 1)
    target = ((1.0 + x_gl) / 2.0) ** order
    b = np.empty(order + 1, dtype=float)
    for n in range(order + 1):
        # In the axisymmetric_pattern convention:
        # p(x) = Σ b_n * (2n+1)/(4π) * P_n(x)
        # so b_n = 2π ∫_{-1}^{1} p(x) P_n(x) dx.
        b[n] = 2.0 * math.pi * np.dot(w_gl, target * eval_legendre(n, x_gl))
    front_gain = sum(b[n] * ((2 * n + 1) / (4 * math.pi)) for n in range(order + 1))
    if front_gain != 0:
        b /= front_gain
    return b


def _design_supercardioid_numerical(order: int) -> np.ndarray:
    """Solve the supercardioid max front-back ratio design.

    The design criterion is the front-to-back hemisphere power ratio:

    ``argmax_b  (∫front |p(theta)|² dΩ) / (∫back |p(theta)|² dΩ)``

    with ``p(theta)`` represented in the same axisymmetric Legendre basis as
    :func:`axisymmetric_pattern`. This yields a symmetric generalized
    eigenvalue problem.
    """
    n_nodes = max(128, 16 * (order + 1))
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_nodes)

    x_front = 0.5 * (x_gl + 1.0)
    x_back = 0.5 * (x_gl - 1.0)
    w_half = 0.5 * w_gl

    basis_front = np.stack(
        [((2 * n + 1) / (4 * np.pi)) * eval_legendre(n, x_front) for n in range(order + 1)],
        axis=1,
    )
    basis_back = np.stack(
        [((2 * n + 1) / (4 * np.pi)) * eval_legendre(n, x_back) for n in range(order + 1)],
        axis=1,
    )

    front_power = basis_front.T @ (basis_front * w_half[:, None])
    back_power = basis_back.T @ (basis_back * w_half[:, None])
    back_power = back_power + 1e-12 * np.eye(order + 1)

    eigvals, eigvecs = eigh(front_power, back_power)
    b = eigvecs[:, np.argmax(eigvals)]
    b = np.real_if_close(b, tol=1_000).astype(float, copy=False)

    front_gain = sum(b[n] * ((2 * n + 1) / (4 * np.pi)) for n in range(order + 1))
    if front_gain < 0:
        b = -b
    return b
