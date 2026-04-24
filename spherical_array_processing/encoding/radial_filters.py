"""Radial equalization filters for SH-encoded microphone signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..acoustics.radial import bn_matrix, _validate_sphere_and_dir_coeff


RegularizationMode = Literal["tikhonov", "wng_limit", "none"]


def _validate_dir_coeff(array_type: str, dir_coeff: float | None) -> None:
    """Equalizer-layer façade over the shared
    :func:`spherical_array_processing.acoustics.radial._validate_sphere_and_dir_coeff`
    helper.

    Keeping a thin wrapper here lets the equalizer call sites read as
    ``_validate_dir_coeff(array_type, dir_coeff)`` while the actual
    contract lives in one place in the acoustics layer, so future
    changes to the directional-capsule rules only need to be made
    once.  ``arg_name="array_type"`` makes the error message track
    the caller's kwarg name.
    """
    _validate_sphere_and_dir_coeff(
        array_type, dir_coeff, arg_name="array_type",
    )


def radial_equalizer_tikhonov(
    max_order: int,
    kr: ArrayLike,
    *,
    array_type: Literal["open", "rigid", "cardioid", "directional"] = "rigid",
    ka: ArrayLike | None = None,
    regularization: float = 1e-3,
    repeat_per_order: bool = True,
    dir_coeff: float | None = None,
) -> NDArray[np.complex128]:
    """Tikhonov-regularized inverse of the modal radial function.

    Returns ``H_n(kr) = B_n*(kr) / (|B_n(kr)|² + λ²)`` where ``λ`` is the
    ``regularization`` argument.  Small ``λ`` ≈ naive ``1/B_n`` away
    from modal zeros; larger ``λ`` trades accuracy for noise rejection
    at low ``kr``.

    Parameters
    ----------
    max_order : int
        Maximum SH order ``N``.
    kr : array_like, shape (F,)
        Measurement-radius wavenumber product(s).
    array_type : {"open", "rigid", "cardioid", "directional"}, optional
        Baffle type passed through to :func:`bn_matrix`.
        ``"directional"`` needs a *dir_coeff*.
    ka : array_like or None, optional
        Baffle-radius wavenumber product(s).  Defaults to *kr*.
    regularization : float, optional
        Tikhonov parameter ``λ``.  Must be non-negative; ``0`` falls
        back to the true inverse (which will diverge at modal zeros).
    repeat_per_order : bool, optional
        If ``True`` (default) expand the per-order filter to ACN ordering
        with ``(N+1)²`` columns; otherwise return the compact
        ``(N+1)``-column form.
    dir_coeff : float or None, optional
        First-order directional coefficient ``α ∈ [0, 1]`` forwarded to
        :func:`bn_matrix` when ``array_type="directional"``.  Must be
        omitted for the other array types.
    """
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    _validate_dir_coeff(array_type, dir_coeff)
    kr_arr = np.asarray(kr, dtype=float).reshape(-1)
    bn = bn_matrix(
        max_order,
        kr=kr_arr,
        ka=ka,
        sphere=array_type,
        repeat_per_order=repeat_per_order,
        dir_coeff=dir_coeff,
    )
    if regularization == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(np.abs(bn) > 0, 1.0 / bn, 0.0)
    else:
        denom = np.abs(bn) ** 2 + float(regularization) ** 2
        out = np.conj(bn) / denom
    return np.asarray(out, dtype=np.complex128)


def radial_equalizer_wng_limited(
    max_order: int,
    kr: ArrayLike,
    *,
    array_type: Literal["open", "rigid", "cardioid", "directional"] = "rigid",
    ka: ArrayLike | None = None,
    max_gain_db: float = 30.0,
    repeat_per_order: bool = True,
    dir_coeff: float | None = None,
) -> NDArray[np.complex128]:
    """Soft-limited inverse filter bounded by a white-noise-gain ceiling.

    Mirrors Politis' ``arraySHTfiltersTheory_radInverse`` style: start
    from the plain inverse ``1/B_n``, then soft-limit each entry so that
    its magnitude never exceeds ``10**(max_gain_db / 20)``.  A
    continuously-differentiable ``tanh`` compressor is used to avoid the
    phase discontinuity of a hard cap.

    Parameters
    ----------
    max_order : int
        Maximum SH order ``N``.
    kr : array_like, shape (F,)
        Measurement-radius wavenumber product(s).
    array_type : {"open", "rigid", "cardioid", "directional"}, optional
        Baffle type.  ``"directional"`` needs a *dir_coeff*.
    ka : array_like or None, optional
        Baffle-radius wavenumber product(s).  Defaults to *kr*.
    max_gain_db : float, optional
        Magnitude ceiling in dB for the returned filter.
    repeat_per_order : bool, optional
        Whether to expand to ACN ordering (default) or return the
        compact per-order form.
    dir_coeff : float or None, optional
        First-order directional coefficient ``α ∈ [0, 1]`` forwarded to
        :func:`bn_matrix` when ``array_type="directional"``.

    Notes
    -----
    The filter is purely magnitude-limiting; the modal phase
    ``arg(1/B_n) = -arg(B_n)`` is preserved exactly.
    """
    _validate_dir_coeff(array_type, dir_coeff)
    kr_arr = np.asarray(kr, dtype=float).reshape(-1)
    bn = bn_matrix(
        max_order,
        kr=kr_arr,
        ka=ka,
        sphere=array_type,
        repeat_per_order=repeat_per_order,
        dir_coeff=dir_coeff,
    )
    max_abs = 10.0 ** (float(max_gain_db) / 20.0)

    abs_bn = np.abs(bn)
    # Let the modal zeros produce ``1/|B_n| = +∞`` and feed them directly
    # into ``tanh`` so the compressor saturates at the ceiling there
    # (``tanh(∞) = 1`` ⇒ ``max_abs·tanh(∞/max_abs) = max_abs``).  Masking
    # non-finite entries to zero — the pre-fix behaviour — collapsed the
    # filter to silence at modal zeros instead of honouring the
    # advertised ``max_gain_db`` ceiling.
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_gain = np.where(abs_bn > 0, 1.0 / abs_bn, np.inf)
    limited = max_abs * np.tanh(raw_gain / max_abs)
    # Phase of 1/B_n.  ``np.angle(0) == 0`` so the phasor stays well
    # defined at modal zeros; combined with the saturated magnitude the
    # filter remains bounded and continuous.
    phase = np.exp(-1j * np.angle(bn))
    out = limited * phase
    return np.asarray(out, dtype=np.complex128)


def radial_equalizer(
    max_order: int,
    kr: ArrayLike,
    *,
    array_type: Literal["open", "rigid", "cardioid", "directional"] = "rigid",
    ka: ArrayLike | None = None,
    regularization: RegularizationMode = "tikhonov",
    tikhonov_lambda: float = 1e-3,
    max_gain_db: float = 30.0,
    repeat_per_order: bool = True,
    dir_coeff: float | None = None,
) -> NDArray[np.complex128]:
    """Unified entry point for the built-in radial equalizers.

    Parameters
    ----------
    regularization : {"tikhonov", "wng_limit", "none"}
        Pick between :func:`radial_equalizer_tikhonov`,
        :func:`radial_equalizer_wng_limited`, or the unregularized
        ``1/B_n`` inverse.
    tikhonov_lambda, max_gain_db : float
        Mode-specific knobs; inactive modes are ignored.
    dir_coeff : float or None, optional
        First-order directional coefficient ``α ∈ [0, 1]`` forwarded to
        the underlying equalizer when ``array_type="directional"``.
    """
    if regularization == "tikhonov":
        return radial_equalizer_tikhonov(
            max_order,
            kr,
            array_type=array_type,
            ka=ka,
            regularization=float(tikhonov_lambda),
            repeat_per_order=repeat_per_order,
            dir_coeff=dir_coeff,
        )
    if regularization == "wng_limit":
        return radial_equalizer_wng_limited(
            max_order,
            kr,
            array_type=array_type,
            ka=ka,
            max_gain_db=float(max_gain_db),
            repeat_per_order=repeat_per_order,
            dir_coeff=dir_coeff,
        )
    if regularization == "none":
        return radial_equalizer_tikhonov(
            max_order,
            kr,
            array_type=array_type,
            ka=ka,
            regularization=0.0,
            repeat_per_order=repeat_per_order,
            dir_coeff=dir_coeff,
        )
    raise ValueError(f"unknown regularization: {regularization!r}")


def apply_radial_equalizer(
    sh_coeffs: ArrayLike,
    equalizer: ArrayLike,
    *,
    freq_axis: int = 0,
    coeff_axis: int = -1,
) -> NDArray[np.complex128]:
    """Apply a per-frequency radial equalizer to SH coefficients.

    Parameters
    ----------
    sh_coeffs : array_like
        SH coefficient tensor with a frequency axis (length ``F``) and a
        coefficient axis (length ``(N+1)²`` for ACN or ``N+1`` for the
        compact per-order layout; must match *equalizer*).
    equalizer : array_like, shape ``(F, C)``
        Radial filter matrix from :func:`radial_equalizer`.
    freq_axis : int, optional
        Axis in *sh_coeffs* that indexes frequency bins.
    coeff_axis : int, optional
        Axis in *sh_coeffs* that indexes SH coefficients.

    Returns
    -------
    ndarray
        Equalized SH coefficients with the same shape as *sh_coeffs*.
    """
    c = np.asarray(sh_coeffs)
    eq = np.asarray(equalizer, dtype=np.complex128)
    if eq.ndim != 2:
        raise ValueError("equalizer must be 2D (F, C)")

    # Move freq_axis to position 0, coeff_axis to position -1.
    c = np.moveaxis(c, freq_axis, 0)
    # coeff_axis was originally coeff_axis in the original array; after
    # moving freq_axis to 0, compute new coeff_axis index.
    # Simpler: move both axes in sequence.
    # (But moving twice may shift indices — use numpy's moveaxis variadic.)
    c_orig = np.asarray(sh_coeffs)
    n = c_orig.ndim
    faxis = freq_axis % n
    caxis = coeff_axis % n
    if faxis == caxis:
        raise ValueError("freq_axis and coeff_axis must be different")
    c_perm = np.moveaxis(c_orig, (faxis, caxis), (0, -1))
    if c_perm.shape[0] != eq.shape[0]:
        raise ValueError(
            f"frequency-axis length mismatch: sh_coeffs has "
            f"{c_perm.shape[0]} bins but equalizer has {eq.shape[0]}"
        )
    if c_perm.shape[-1] != eq.shape[1]:
        raise ValueError(
            f"coefficient-axis length mismatch: sh_coeffs has "
            f"{c_perm.shape[-1]} channels but equalizer has {eq.shape[1]}"
        )
    eq_broadcast = eq.reshape((eq.shape[0],) + (1,) * (c_perm.ndim - 2) + (eq.shape[1],))
    out = c_perm * eq_broadcast
    return np.moveaxis(out, (0, -1), (faxis, caxis))
