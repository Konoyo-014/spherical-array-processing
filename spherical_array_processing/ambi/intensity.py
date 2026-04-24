"""Active / reactive sound-intensity decomposition from an FOA STFT.

For a first-order ambisonic frequency-domain signal with pressure
``W(ω)`` and particle-velocity proxies ``X(ω), Y(ω), Z(ω)``, the
complex instantaneous intensity vector is

.. math::

    \\mathbf{I}(\\omega) = W^{\\*}(\\omega) \\cdot \\bigl(X(\\omega), Y(\\omega), Z(\\omega)\\bigr).

The **active** part ``Re{I}`` points toward the encoded source
direction under the package's plane-wave SH convention and is the
quantity used by DirAC for per-bin DOA estimation.  The **reactive** part ``Im{I}``
captures the stored oscillating energy local to the listener — it is
zero for a single propagating plane wave and non-zero for standing-
wave or near-field components.  The ratio between them underlies
diffuseness estimators such as Pulkki's DirAC-ψ.

This module handles any of the supported FOA normalisations and
returns Cartesian ``(I_x, I_y, I_z)`` triplets so downstream code
does not have to track ACN index gymnastics.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..types import NormalizationKind
from .format import convert_ambi_normalization


def _acn_to_cartesian(
    foa: NDArray, coeff_axis: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split an ACN-ordered FOA tensor along *coeff_axis* into
    ``(W, X, Y, Z)`` arrays (Cartesian triplet order).

    Recall ACN: q=0 ↔ W (n=0, m=0), q=1 ↔ Y (n=1, m=-1),
    q=2 ↔ Z (n=1, m=0), q=3 ↔ X (n=1, m=+1).
    """
    sig = np.moveaxis(foa, coeff_axis, 0)
    if sig.shape[0] < 4:
        raise ValueError(
            f"foa_stft must carry at least 4 SH channels; got "
            f"{sig.shape[0]} along coeff_axis={coeff_axis}"
        )
    w = sig[0]
    y = sig[1]
    z = sig[2]
    x = sig[3]
    return w, x, y, z


def _canonical_foa_pv(
    foa_stft: ArrayLike,
    *,
    normalization: NormalizationKind = "orthonormal",
    coeff_axis: int = -2,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return physical pressure / Cartesian-velocity FOA from an input STFT.

    The package uses **orthonormal** SH coefficients as its canonical
    internal convention.  Under that convention the first-order
    dipoles ``(X, Y, Z)`` carry an extra ``√3`` factor relative to
    the omni channel ``W``, so the physical velocity proxy satisfying
    ``|v|² = |p|² |û|²`` for a plane wave is ``v = (X, Y, Z) / √3``.
    This helper performs the two canonicalisations (normalisation →
    orthonormal, then ``1/√3`` on velocity) and returns the four
    Cartesian channels ready for intensity / energy formulas.

    Used by both :func:`intensity_vector` (via its ``physical_units``
    path — historically absent, added in 0.4.0b14 through this shared
    helper) and :func:`spherical_array_processing.dirac.dirac_analysis`
    so the two APIs agree bit-for-bit on the meaning of pressure and
    velocity.

    Returns
    -------
    w, vx, vy, vz : ndarrays
        Pressure ``w`` and Cartesian velocity proxies ``vx``, ``vy``,
        ``vz`` with *coeff_axis* removed (each has the same shape
        as the input STFT with that axis dropped).
    """
    sig = np.asarray(foa_stft)
    if normalization != "orthonormal":
        sig = convert_ambi_normalization(
            sig, max_order=1,
            from_=normalization, to="orthonormal", axis=coeff_axis,
        )
    w, x, y, z = _acn_to_cartesian(sig, coeff_axis)
    v_scale = 1.0 / np.sqrt(3.0)
    return w, x * v_scale, y * v_scale, z * v_scale


def intensity_vector(
    foa_stft: ArrayLike,
    *,
    normalization: NormalizationKind = "orthonormal",
    coeff_axis: int = -2,
    return_reactive: bool = False,
    physical_units: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """Cartesian active (and optionally reactive) intensity vectors.

    Parameters
    ----------
    foa_stft : array_like
        Complex FOA STFT with four SH channels along *coeff_axis* in
        ACN order.  A typical layout is ``(F, 4, T)`` (the default
        package-wide STFT layout), in which case the intensity axis
        replaces the SH axis and the output shape becomes
        ``(F, 3, T)``.
    normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation of *foa_stft*.  Default ``"orthonormal"``.
    coeff_axis : int, optional
        Axis along which the four SH channels live.  Default ``-2``
        (matches the ``(F, 4, T)`` convention).
    return_reactive : bool, optional
        If ``True`` also return the reactive intensity vector.  Default
        ``False``.
    physical_units : bool, optional
        When ``True``, divide the first-order Cartesian components by
        ``√3`` before forming the intensity.  Under the package's
        orthonormal SH convention this collapses the extra SH
        prefactor so a plane wave satisfies ``|I| = |p|² |û|`` — i.e.
        the intensity is the true pressure–velocity product rather than
        a coefficient-space proxy.  Default ``False`` (backward
        compatible; the historical output is proportional-but-not-
        equal to physical intensity).

    Returns
    -------
    active : ndarray
        ``Re{W^* · (X, Y, Z)}`` (or its physical-units variant) with
        the same shape as *foa_stft* except the SH axis has length
        ``3``.
    reactive : ndarray
        Only when *return_reactive* is true: ``Im{W^* · (X, Y, Z)}``.

    Notes
    -----
    Set ``physical_units=True`` when you need the intensity in units
    that match the textbook DirAC ``ψ = 1 − ||I||/E`` formula — this
    is what :func:`spherical_array_processing.dirac.dirac_analysis`
    uses internally via the shared ``_canonical_foa_pv`` helper, so
    the two entry points agree bit-for-bit on what "intensity vector"
    means.
    """
    if physical_units:
        w, vx, vy, vz = _canonical_foa_pv(
            foa_stft,
            normalization=normalization, coeff_axis=coeff_axis,
        )
        w_conj = np.conj(w)
        i_complex = np.stack(
            [w_conj * vx, w_conj * vy, w_conj * vz], axis=0,
        )
    else:
        sig = np.asarray(foa_stft)
        if normalization != "orthonormal":
            sig = convert_ambi_normalization(
                sig, max_order=1,
                from_=normalization, to="orthonormal", axis=coeff_axis,
            )
        w, x, y, z = _acn_to_cartesian(sig, coeff_axis)
        # Build I = W* · (X, Y, Z).
        i_complex = np.stack(
            [np.conj(w) * x, np.conj(w) * y, np.conj(w) * z], axis=0,
        )
    active = np.real(i_complex)
    # Put the Cartesian axis where the SH axis used to be.
    active = np.moveaxis(active, 0, coeff_axis)
    if not return_reactive:
        return np.asarray(active, dtype=np.float64)
    reactive = np.imag(i_complex)
    reactive = np.moveaxis(reactive, 0, coeff_axis)
    return (
        np.asarray(active, dtype=np.float64),
        np.asarray(reactive, dtype=np.float64),
    )


def doa_from_intensity(
    foa_stft: ArrayLike,
    *,
    normalization: NormalizationKind = "orthonormal",
    coeff_axis: int = -2,
) -> NDArray[np.float64]:
    """Unit-norm DOA estimate ``û = Î_active / ||Î_active||`` per bin.

    Parameters
    ----------
    foa_stft : array_like
        Complex FOA STFT.
    normalization, coeff_axis : see :func:`intensity_vector`.

    Returns
    -------
    ndarray
        Unit Cartesian DOA per bin with the same shape as the active
        intensity (SH axis replaced by length-3 Cartesian axis).  Bins
        with zero energy come out as zero.

    Notes
    -----
    The ambisonic "intensity" ``I = Re{W^* (X, Y, Z)}`` points
    **toward** the source, because the SH encoding stores the source
    direction ``û`` directly in the first-order coefficients rather
    than the wave-vector direction ``−û``.  DOA is therefore
    ``+I / ||I||``, not ``−I / ||I||`` as one would expect from a
    physical pressure / particle-velocity intensity.
    """
    active = intensity_vector(
        foa_stft,
        normalization=normalization, coeff_axis=coeff_axis,
    )
    # Move the Cartesian axis to -1 so that norm along -1 makes sense.
    active_last = np.moveaxis(active, coeff_axis, -1)
    norm = np.linalg.norm(active_last, axis=-1, keepdims=True)
    safe = np.where(norm > 0.0, norm, 1.0)
    doa = active_last / safe
    # Bins with zero energy → keep as zeros.
    doa = np.where(norm > 0.0, doa, 0.0)
    return np.moveaxis(doa, -1, coeff_axis).astype(np.float64)


__all__ = ["doa_from_intensity", "intensity_vector"]
