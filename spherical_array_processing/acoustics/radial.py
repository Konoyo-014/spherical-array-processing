from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import spherical_jn, spherical_yn

from ..sh.basis import replicate_per_order


def _a(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=float)


def besseljs(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Spherical Bessel function of the first kind, j_n(x).

    Parameters
    ----------
    n : int or array_like
        Order of the Bessel function.
    x : array_like
        Argument (real-valued).

    Returns
    -------
    ndarray
        Values of j_n(x).

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose(besseljs(0, 0.0), 1.0)
    True
    """
    return spherical_jn(n, _a(x))


def besseljsd(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Derivative of the spherical Bessel function of the first kind, j_n'(x).

    Parameters
    ----------
    n : int or array_like
        Order of the Bessel function.
    x : array_like
        Argument (real-valued).

    Returns
    -------
    ndarray
        Values of j_n'(x).

    Examples
    --------
    >>> import numpy as np
    >>> besseljsd(0, 0.0).shape
    ()
    """
    return spherical_jn(n, _a(x), derivative=True)


def besselys(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Spherical Neumann function, y_n(x).

    This is the spherical Bessel function of the second kind, matching
    ``scipy.special.spherical_yn``.
    """
    with np.errstate(all="ignore"):
        return spherical_yn(n, _a(x))


def besselysd(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Derivative of the spherical Neumann function, y_n'(x)."""
    with np.errstate(all="ignore"):
        return spherical_yn(n, _a(x), derivative=True)


def besselhs(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.complex128]:
    """Spherical Hankel function of the first kind, h_n^(1)(x) = j_n(x) + i*y_n(x).

    Parameters
    ----------
    n : int or array_like
        Order of the Hankel function.
    x : array_like
        Argument (real-valued).

    Returns
    -------
    ndarray of complex128
        Values of h_n(x).

    Examples
    --------
    >>> import numpy as np
    >>> h = besselhs(0, 1.0)
    >>> np.iscomplexobj(h)
    True
    """
    x = _a(x)
    with np.errstate(all="ignore"):
        y = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    return np.asarray(y, dtype=np.complex128)


def besselhs2(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.complex128]:
    """Spherical Hankel function of the second kind, h_n^(2)(x)."""
    x = _a(x)
    with np.errstate(all="ignore"):
        y = spherical_jn(n, x) - 1j * spherical_yn(n, x)
    return np.asarray(y, dtype=np.complex128)


def besselhsd(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.complex128]:
    """Derivative of the spherical Hankel function, h_n'(x).

    Parameters
    ----------
    n : int or array_like
        Order of the Hankel function.
    x : array_like
        Argument (real-valued).

    Returns
    -------
    ndarray of complex128
        Values of h_n'(x).

    Examples
    --------
    >>> import numpy as np
    >>> hd = besselhsd(0, 1.0)
    >>> np.iscomplexobj(hd)
    True
    """
    x = _a(x)
    with np.errstate(all="ignore"):
        y = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    return np.asarray(y, dtype=np.complex128)


def besselhs2d(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.complex128]:
    """Derivative of the spherical Hankel function of the second kind."""
    x = _a(x)
    with np.errstate(all="ignore"):
        y = spherical_jn(n, x, derivative=True) - 1j * spherical_yn(n, x, derivative=True)
    return np.asarray(y, dtype=np.complex128)


def wavenumber(freqs_hz: ArrayLike, c: float = 343.0) -> NDArray[np.float64]:
    """Return acoustic wavenumber ``k = 2*pi*f/c``.

    Parameters
    ----------
    freqs_hz : array_like
        Frequency or frequencies in Hz.
    c : float, optional
        Speed of sound in m/s. Default is 343.0.
    """
    if c <= 0:
        raise ValueError("speed of sound must be positive")
    return 2.0 * np.pi * _a(freqs_hz) / float(c)


def kr(freqs_hz: ArrayLike, radius_m: float, c: float = 343.0) -> NDArray[np.float64]:
    """Return dimensionless ``k * radius`` values for acoustic frequencies."""
    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")
    return wavenumber(freqs_hz, c=c) * float(radius_m)


def _sphere_is_directional(sphere: int | str) -> bool:
    """Return True iff *sphere* names / numbers the generic directional
    first-order capsule branch (``"directional"`` or ``3``)."""
    if isinstance(sphere, str):
        return sphere == "directional"
    return int(sphere) == 3


def _validate_sphere_and_dir_coeff(
    sphere: int | str,
    dir_coeff: float | None,
    *,
    arg_name: str = "sphere",
) -> None:
    """Enforce the directional-capsule contract for every layer that
    accepts a sphere / array-type selector paired with an optional
    first-order coefficient.

    ``dir_coeff`` must be supplied (and live in ``[0, 1]``) when
    *sphere* selects the directional branch, and must be left ``None``
    otherwise.  This single helper is the **authoritative definition**
    of the contract; the equalizer, simulation, and acoustics public
    entry points all call it rather than re-implementing the same
    three-case check, which makes maintenance drift impossible.

    The *arg_name* keyword controls the user-facing vocabulary in the
    error message — pass ``"sphere"`` when the caller's kwarg is
    literally ``sphere=`` (raw acoustics API) and ``"array_type"``
    when the caller's kwarg is ``array_type=`` (simulation, modal
    wrapper, equalizer family), so the error messages always speak
    the same language as the call site that produced the mistake.
    """
    if _sphere_is_directional(sphere):
        if dir_coeff is None:
            raise ValueError(
                f"{arg_name}='directional' requires a dir_coeff in [0, 1]"
            )
        alpha = float(dir_coeff)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"dir_coeff must be in [0, 1] for numerically sensible"
                f" first-order capsules, got {alpha}"
            )
    elif dir_coeff is not None:
        raise ValueError(
            f"dir_coeff is only meaningful for {arg_name}='directional',"
            f" got {arg_name}={sphere!r}"
        )


def plane_wave_radial_bn(
    n: int,
    kr: ArrayLike,
    ka: ArrayLike | None = None,
    sphere: int | str = 1,
    *,
    dir_coeff: float | None = None,
) -> NDArray[np.complex128]:
    """Rafaely-style Bn radial function.

    Computes the plane-wave radial coefficient ``B_n(kr)`` for a single
    order *n*, following the formulation in Rafaely's *Fundamentals of
    Spherical Array Processing*.

    The returned coefficient is the per-SH-mode amplitude that multiplies
    ``Y_n^m*(k̂_src) · Y_n^m(r̂)`` in the decomposition of the pressure at
    the mic position, i.e. it carries the ``4π iⁿ`` factor that appears
    in the Jacobi–Anger expansion ``exp(ik·r) = Σ 4π iⁿ j_n(kr) Σ_m
    Y_n^m*(k̂) Y_n^m(r̂)``.

    Parameters
    ----------
    n : int
        Spherical harmonic order.
    kr : array_like
        Wavenumber-radius product(s) at the measurement radius.
    ka : array_like or None, optional
        Wavenumber-radius product(s) at the scatterer surface. Defaults
        to *kr* (microphones flush on the sphere).
    sphere : {0, 1, 2, ``"open"``, ``"rigid"``, ``"cardioid"``, ``"directional"``}, optional
        Array type. ``0`` / ``"open"`` — open sphere,
        ``1`` / ``"rigid"`` — rigid sphere (default),
        ``2`` / ``"cardioid"`` — cardioid capsules on an open sphere, i.e.
        the unit-front-gain ``0.5·(1 + cos θ)`` pattern (equivalent to
        ``"directional"`` with ``dir_coeff=0.5``),
        ``"directional"`` — first-order capsules ``α + (1-α)·cos θ``
        on an open sphere; requires *dir_coeff*.
    dir_coeff : float or None, optional
        Directional coefficient ``α`` in ``[0, 1]`` for
        ``sphere="directional"``.  ``α=1`` is omni, ``α=0.5`` is the
        standard cardioid, ``α=0`` is a figure-eight dipole aligned with
        the mic radial.  Closed form: ``4π iⁿ (α j_n(kr) − j(1-α) j_n'(kr))``.
        Ignored for other *sphere* types.

    Returns
    -------
    ndarray of complex128
        Radial coefficient values with the same shape as *kr*.

    Notes
    -----
    The rigid-sphere branch uses the spherical Hankel function of the
    second kind ``h_n^(2)`` so that the scattered wave is outgoing under
    the ``e^{+jωt}`` engineering time convention (the convention produced
    by ``numpy.fft.ifft`` / MATLAB ``ifft`` on a conjugate-symmetric
    spectrum).

    Examples
    --------
    >>> import numpy as np
    >>> b = plane_wave_radial_bn(0, np.array([0.5]))
    >>> b.shape
    (1,)
    """
    _validate_sphere_and_dir_coeff(sphere, dir_coeff)
    kr = _a(kr)
    if ka is None:
        ka = kr
    ka = _a(ka)
    kind = sphere
    if isinstance(kind, str):
        kind = {"open": 0, "rigid": 1, "cardioid": 2, "directional": 3}[kind]
    j = 1j
    if kind == 0:
        return 4 * np.pi * (j ** n) * besseljs(n, kr)
    if kind == 1:
        # Rigid sphere: b_n = 4π·iⁿ·[j_n(kr) - j_n'(ka)/h_n^(2)'(ka) · h_n^(2)(kr)]
        # The spherical Hankel function of the second kind carries the
        # outgoing-wave radiation condition under the e^{+jωt} engineering
        # time convention (the convention implied by numpy/MATLAB ``ifft``
        # when producing a causal real impulse response from a
        # conjugate-symmetric spectrum).
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = besseljsd(n, ka) / besselhs2d(n, ka)
            ratio = np.nan_to_num(ratio, nan=0.0)
            correction = ratio * besselhs2(n, kr)
            correction = np.nan_to_num(correction, nan=0.0)
        return 4 * np.pi * (j ** n) * (besseljs(n, kr) - correction)
    if kind == 2:
        # Cardioid (α·omni + (1-α)·dipole) with α = 0.5 on an open sphere.
        # The closed form is 4π·iⁿ·(α·j_n(kr) - j·(1-α)·j_n'(kr));
        # substituting α = 0.5 and pulling the scalar 0.5 out yields
        # 2π·iⁿ·(j_n(kr) - j·j_n'(kr)), matching the unit-front-gain
        # cardioid p(θ) = 0.5(1 + cos θ).
        return 2 * np.pi * (j ** n) * (besseljs(n, kr) - j * besseljsd(n, kr))
    if kind == 3:
        # Contract already enforced by _validate_sphere_and_dir_coeff
        # at the top of this function — dir_coeff is guaranteed to be
        # a float in [0, 1] here.
        alpha = float(dir_coeff)  # type: ignore[arg-type]
        # Directional first-order pattern α + (1-α) cos θ on an open sphere:
        # B_n = 4π iⁿ (α j_n - j (1-α) j_n').  α=1 reduces to open, α=0.5
        # to cardioid, α=0 to figure-of-eight.
        return 4 * np.pi * (j ** n) * (
            alpha * besseljs(n, kr) - j * (1.0 - alpha) * besseljsd(n, kr)
        )
    raise ValueError(f"unsupported sphere kind: {sphere}")


def bn_matrix(
    max_order: int,
    kr: ArrayLike,
    ka: ArrayLike | None = None,
    sphere: int | str = 1,
    repeat_per_order: bool = True,
    *,
    dir_coeff: float | None = None,
) -> NDArray[np.complex128]:
    """Build the radial filter matrix B_n(kr) for all orders up to *max_order*.

    Parameters
    ----------
    max_order : int
        Maximum SH order N.
    kr : array_like
        Wavenumber-radius product(s), length K.
    ka : array_like or None, optional
        Scatterer wavenumber-radius product(s). Defaults to *kr*.
    sphere : {0, 1, 2, ``"open"``, ``"rigid"``, ``"cardioid"``, ``"directional"``}, optional
        Array type (default ``1`` -- rigid).
    repeat_per_order : bool, optional
        If ``True`` (default), each ``B_n`` column is replicated ``2n+1``
        times so the output has ``(N+1)^2`` columns (one per ACN index).
        If ``False``, the output has ``N+1`` columns (one per order).
    dir_coeff : float or None, optional
        Directional coefficient ``α`` for ``sphere="directional"``.  See
        :func:`plane_wave_radial_bn`.

    Returns
    -------
    ndarray of complex128, shape (K, n_cols)
        Radial filter matrix where ``n_cols`` is ``(N+1)**2`` when
        *repeat_per_order* is ``True``, or ``N+1`` otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> B = bn_matrix(1, np.array([1.0, 2.0]))
    >>> B.shape
    (2, 4)
    >>> B_short = bn_matrix(1, np.array([1.0]), repeat_per_order=False)
    >>> B_short.shape
    (1, 2)
    """
    # Validate up front so misconfigurations fail before the per-order
    # loop instead of on the first iteration, and so the traceback
    # points at this function rather than the inner helper.
    _validate_sphere_and_dir_coeff(sphere, dir_coeff)
    kr_arr = _a(kr).reshape(-1)
    rows: list[NDArray[np.complex128]] = []
    for n in range(max_order + 1):
        rows.append(
            plane_wave_radial_bn(n, kr_arr, ka=ka, sphere=sphere, dir_coeff=dir_coeff)
        )
    b = np.stack(rows, axis=-1)  # [K, N+1]
    if not repeat_per_order:
        return b
    rep = replicate_per_order(np.arange(max_order + 1))
    out = np.zeros((kr_arr.size, rep.size), dtype=np.complex128)
    cursor = 0
    for n in range(max_order + 1):
        count = 2 * n + 1
        out[:, cursor : cursor + count] = b[:, [n]]
        cursor += count
    return out


def sph_modal_coeffs(
    max_order: int,
    kR: ArrayLike,
    array_type: str = "rigid",
    *,
    dir_coeff: float | None = None,
) -> NDArray[np.complex128]:
    """Compute spherical modal coefficients for an open, rigid, cardioid, or
    generic directional array.

    A convenience wrapper around :func:`bn_matrix` with ``ka = kR`` and
    ``repeat_per_order=False``, returning one column per SH order.

    Parameters
    ----------
    max_order : int
        Maximum SH order.
    kR : array_like
        Wavenumber-radius product(s), length K.
    array_type : str, optional
        ``"rigid"`` (default), ``"open"``, ``"cardioid"``, or
        ``"directional"``.
    dir_coeff : float or None, optional
        Directional coefficient ``α`` for ``array_type="directional"``.
        See :func:`plane_wave_radial_bn`.

    Returns
    -------
    ndarray of complex128, shape (K, max_order + 1)
        Modal coefficients, one column per order.

    Examples
    --------
    >>> import numpy as np
    >>> m = sph_modal_coeffs(2, np.array([1.0]))
    >>> m.shape
    (1, 3)
    """
    array_kind_map = {"open": 0, "rigid": 1, "cardioid": 2, "directional": 3}
    if array_type not in array_kind_map:
        raise ValueError(
            f"array_type must be one of "
            f"{tuple(array_kind_map.keys())}, got {array_type!r}"
        )
    # Validate with the caller's vocabulary (``array_type=``) rather
    # than the internal ``sphere=`` code so the error message keeps
    # pointing at the kwarg the user actually wrote.
    _validate_sphere_and_dir_coeff(
        array_type, dir_coeff, arg_name="array_type",
    )
    return bn_matrix(
        max_order,
        kR,
        ka=kR,
        sphere=array_kind_map[array_type],
        repeat_per_order=False,
        dir_coeff=dir_coeff,
    )


def equalize_modal_coeffs(
    sh_signals: ArrayLike,
    bn: ArrayLike,
    reg_param: float = 1e-4,
    reg_type: str = "tikhonov",
) -> NDArray[np.complex128]:
    """Apply regularized modal equalization to SH-domain array signals.

    ``bn`` may be ACN-expanded with shape ``(F, (N+1)^2)`` or compact
    per-order with shape ``(F, N+1)``. The SH coefficient axis is the last
    axis and the frequency axis is the second-to-last axis.
    """
    sig = np.asarray(sh_signals, dtype=np.complex128)
    coeffs = np.asarray(bn, dtype=np.complex128)
    if sig.ndim < 2:
        raise ValueError("sh_signals must have at least frequency and coefficient axes")
    if coeffs.ndim != 2:
        raise ValueError("bn must be a 2D array with shape (F, C)")
    if sig.shape[-2] != coeffs.shape[0]:
        raise ValueError("frequency-axis length mismatch between sh_signals and bn")

    n_coeffs = sig.shape[-1]
    if coeffs.shape[1] != n_coeffs:
        n_orders = coeffs.shape[1]
        if n_orders < 1 or n_coeffs != n_orders * n_orders:
            raise ValueError("bn channel count must be (N+1) or (N+1)^2")
        expanded = np.empty((coeffs.shape[0], n_coeffs), dtype=np.complex128)
        cursor = 0
        for degree in range(n_orders):
            count = 2 * degree + 1
            expanded[:, cursor: cursor + count] = coeffs[:, [degree]]
            cursor += count
        coeffs = expanded

    mag = np.abs(coeffs)
    mag_max = np.maximum(np.max(mag, axis=-1, keepdims=True), 1e-30)
    if reg_type == "tikhonov":
        inv = np.conj(coeffs) / (mag**2 + float(reg_param) * mag_max**2)
    elif reg_type == "softlimit":
        floor = float(reg_param) * mag_max
        mag_limited = np.maximum(mag, floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(mag > 0, mag_limited / mag * (1.0 / coeffs), 0.0)
    else:
        raise ValueError(f"unsupported reg_type: {reg_type!r}")

    return sig * inv
