from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..ambi.intensity import intensity_vector
from ..types import NormalizationKind


ChannelOrder = Literal["acn", "fuma"]


def intensity_vectors_from_foa(
    foa: ArrayLike,
    *,
    channel_order: ChannelOrder = "acn",
    normalization: NormalizationKind = "orthonormal",
    physical_units: bool = False,
) -> np.ndarray:
    """Compute instantaneous Cartesian intensity vectors from FOA samples.

    For every time / frequency frame the intensity vector is
    ``I = Re{W^* · (X, Y, Z)}`` (Cartesian, right-handed).  Under the
    package's plane-wave encoding convention it points toward the
    source direction (see :func:`spherical_array_processing.ambi.intensity_vector`
    for the full derivation).

    .. note::
       This entry point is a **thin wrapper** around
       :func:`spherical_array_processing.ambi.intensity_vector`.  It
       preserves the historical last-axis layout
       (``foa.shape == (..., 4)`` in, ``(..., 3)`` out) and the
       ``channel_order`` switch; all other semantic knobs —
       ``normalization`` and ``physical_units`` — are forwarded as-is
       so that this function and ``ambi.intensity_vector`` always
       agree bit-for-bit on what the FOA intensity means under a given
       normalisation / unit convention.  Internally, FuMa input is
       reordered to the package-wide ACN convention before delegating,
       so the output is unchanged relative to previous releases.

    Parameters
    ----------
    foa : array_like
        First-order Ambisonics signal array of shape ``(..., 4)``.
        Interpretation of the four channels depends on
        *channel_order*.  May be real or complex.
    channel_order : {"acn", "fuma"}, optional
        Channel ordering convention of *foa*.  Default ``"acn"``
        (the package-wide convention: ``[W, Y, Z, X]``); switch to
        ``"fuma"`` for legacy B-format input in the ``[W, X, Y, Z]``
        order.  The returned intensity vector is always Cartesian
        ``(I_x, I_y, I_z)``.
    normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation of *foa*.  Default ``"orthonormal"`` (package
        canonical).  Forwarded to
        :func:`~spherical_array_processing.ambi.intensity_vector`.
    physical_units : bool, optional
        When ``True``, scale the velocity components by ``1/√3`` before
        forming the intensity so that a plane wave satisfies
        ``|I| = |p|² |û|`` — see
        :func:`~spherical_array_processing.ambi.intensity_vector` for
        the full derivation.  Default ``False`` (historical behaviour).

    Returns
    -------
    numpy.ndarray
        Real-valued intensity vectors of shape ``(..., 3)``.

    Raises
    ------
    ValueError
        If the last dimension of *foa* is less than 4, or if
        *channel_order* is not a recognised value.

    Notes
    -----
    .. versionchanged:: 0.4.0b12
       Default channel order changed from the legacy FuMa
       ``[W, X, Y, Z]`` to ACN ``[W, Y, Z, X]`` so the function is
       consistent with the rest of the package.  Users who were
       explicitly passing FuMa-ordered FOA must now pass
       ``channel_order="fuma"`` to keep the same numerical result.

    .. versionchanged:: 0.4.0b15
       Collapsed onto the canonical
       :func:`~spherical_array_processing.ambi.intensity_vector`
       implementation; added ``normalization`` and ``physical_units``
       keyword arguments.  The default ``channel_order="acn"``,
       ``normalization="orthonormal"``, ``physical_units=False`` path
       is numerically unchanged (byte-level identical output for the
       covered regression tests).

    Examples
    --------
    >>> import numpy as np
    >>> # ACN [W, Y, Z, X]: a positive X component means source on +x.
    >>> frame = np.array([[1+0j, 0+0j, 0+0j, 0.5+0j]])
    >>> intensity_vectors_from_foa(frame)
    array([[0.5, 0. , 0. ]])
    >>> # FuMa [W, X, Y, Z] (legacy) — same physical field, same output.
    >>> frame_fuma = np.array([[1+0j, 0.5+0j, 0+0j, 0+0j]])
    >>> intensity_vectors_from_foa(frame_fuma, channel_order="fuma")
    array([[0.5, 0. , 0. ]])
    """
    a = np.asarray(foa)
    # Preserve the historical error message so downstream code that
    # keys off ``match="4 channels"`` keeps working.
    if a.ndim == 0 or a.shape[-1] < 4:
        raise ValueError("FOA array must have at least 4 channels")
    if channel_order == "acn":
        acn = a
    elif channel_order == "fuma":
        # FuMa ``[W, X, Y, Z]`` → ACN ``[W, Y, Z, X]`` via the index
        # permutation [0, 2, 3, 1] on the last axis.  Use take() so
        # the operation is view-friendly and axis-agnostic.
        acn = np.take(a, [0, 2, 3, 1], axis=-1)
    else:
        raise ValueError(
            "channel_order must be 'acn' or 'fuma', got "
            f"{channel_order!r}"
        )
    # Delegate to the canonical implementation with the SH axis on -1.
    iv = intensity_vector(
        acn,
        normalization=normalization,
        coeff_axis=-1,
        return_reactive=False,
        physical_units=physical_units,
    )
    return np.asarray(iv, dtype=np.float64)


def diffuseness_ie(pv_cov: ArrayLike) -> float:
    """Estimate diffuseness from intensity-to-energy ratio of a pressure-velocity covariance.

    Parameters
    ----------
    pv_cov : array_like
        Pressure-velocity covariance matrix of shape ``(N, N)`` where
        ``N >= 4``.  The first row/column corresponds to the pressure
        channel (W) and channels 1--3 to the velocity components (X, Y, Z).

    Returns
    -------
    float
        Diffuseness estimate in the range ``[0, 1]``, where 0 indicates a
        fully directional field and 1 a fully diffuse field.

    Raises
    ------
    ValueError
        If *pv_cov* has fewer than 4 rows or columns.

    Examples
    --------
    >>> import numpy as np
    >>> cov = np.eye(4, dtype=complex)
    >>> diffuseness_ie(cov)
    1.0
    """
    c = np.asarray(pv_cov, dtype=np.complex128)
    if c.shape[0] < 4 or c.shape[1] < 4:
        raise ValueError("pv_cov must be at least 4x4")
    ia = np.real(c[1:4, 0])
    ia_norm = np.linalg.norm(ia)
    e = np.real(np.trace(c)) / 2.0
    if e <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - ia_norm / e, 0.0, 1.0))


def diffuseness_tv(i_vecs: ArrayLike) -> float:
    """Estimate diffuseness from temporal variation of intensity vectors.

    Implements the estimator from Ahonen & Pulkki (2009) [1]_ including the
    square-root mapping ``sqrt(1 - ||E[I]|| / E[||I||])``, which matches
    the original MATLAB reference.

    Parameters
    ----------
    i_vecs : array_like
        Intensity vectors of shape ``(T, 3)`` where *T* is the number of
        time frames and each row is a 3-D Cartesian intensity vector.

    Returns
    -------
    float
        Diffuseness estimate in the range ``[0, 1]``.  Returns 1.0 when
        the mean intensity magnitude is near zero.

    References
    ----------
    .. [1] J. Ahonen and V. Pulkki, "Diffuseness estimation using
       temporal variation of intensity vectors," *Proc. IEEE WASPAA*,
       2009.

    Raises
    ------
    ValueError
        If *i_vecs* is not two-dimensional or does not have 3 columns.

    Examples
    --------
    >>> import numpy as np
    >>> vecs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> diffuseness_tv(vecs)
    0.0
    """
    i = np.asarray(i_vecs, dtype=float)
    if i.ndim != 2 or i.shape[1] != 3:
        raise ValueError("i_vecs must be [T,3]")
    norm_i = np.linalg.norm(i, axis=1)
    mean_norm_i = float(np.mean(norm_i))
    if mean_norm_i <= 1e-12:
        return 1.0
    norm_mean_i = float(np.linalg.norm(np.mean(i, axis=0)))
    val = 1.0 - norm_mean_i / mean_norm_i
    return float(np.sqrt(np.clip(val, 0.0, 1.0)))


def diffuseness_sv(i_vecs: ArrayLike) -> float:
    """Estimate diffuseness from the spherical variance of intensity directions.

    Computes the resultant length of unit-normalised intensity vectors
    and returns ``1 - ||mean(d_t)||``.  This is a directional-statistics
    measure (Mardia & Jupp, *Directional Statistics*) sometimes called
    *spherical variance* (SV), **not** a subspace decomposition method.

    Parameters
    ----------
    i_vecs : array_like
        Intensity vectors of shape ``(T, 3)`` where *T* is the number of
        time frames and each row is a 3-D Cartesian intensity vector.

    Returns
    -------
    float
        Diffuseness estimate in the range ``[0, 1]``.  A value of 0
        indicates all direction vectors point the same way; 1 indicates
        fully diffuse (or zero-energy) conditions.

    Raises
    ------
    ValueError
        If *i_vecs* is not two-dimensional or does not have 3 columns.

    Examples
    --------
    >>> import numpy as np
    >>> vecs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> diffuseness_sv(vecs)
    0.0
    """
    i = np.asarray(i_vecs, dtype=float)
    if i.ndim != 2 or i.shape[1] != 3:
        raise ValueError("i_vecs must be [T,3]")
    mags = np.linalg.norm(i, axis=1)
    if np.all(mags <= 1e-12):
        return 1.0
    doa = i / np.maximum(mags[:, None], 1e-12)
    mean_doa = np.mean(doa, axis=0)
    return float(np.clip(1.0 - np.linalg.norm(mean_doa), 0.0, 1.0))


def diffuseness_cmd(sh_cov: ArrayLike) -> tuple[float, np.ndarray]:
    """Estimate diffuseness using the covariance-matrix-based distance (CMD) method.

    Implements the COMEDIE-style eigenvalue analysis from Epain & Jin
    (2016) [1]_.  In a diffuse field the SH covariance eigenvalues are
    equal; in a single plane-wave field one eigenvalue dominates.

    Parameters
    ----------
    sh_cov : array_like
        Spherical-harmonic domain covariance matrix of shape
        ``((N+1)^2, (N+1)^2)`` where *N* is the SH order.

    References
    ----------
    .. [1] N. Epain and C. T. Jin, "Spherical Harmonic Signal Covariance
       and Sound Field Diffuseness," *IEEE/ACM Trans. Audio, Speech, and
       Language Processing*, 24(10), pp. 1796–1807, 2016.

    Returns
    -------
    diff : float
        Overall diffuseness estimate in the range ``[0, 1]``.
    diff_ord : numpy.ndarray
        Per-order diffuseness estimates, array of length *N* where entry
        ``n-1`` corresponds to order *n* (1-based).  The last entry equals
        *diff*.

    Raises
    ------
    ValueError
        If *sh_cov* is not square or its size does not match a valid SH
        order.

    Examples
    --------
    >>> import numpy as np
    >>> cov = np.eye(4, dtype=complex)
    >>> diff, diff_ord = diffuseness_cmd(cov)
    >>> float(np.round(diff, 6))
    1.0
    """
    c = np.asarray(sh_cov, dtype=np.complex128)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("sh_cov must be square")
    n_sh = c.shape[0]
    order = int(round(np.sqrt(n_sh) - 1))
    if (order + 1) ** 2 != n_sh:
        raise ValueError("sh_cov size does not correspond to SH order")

    def _cmd_from_cov(cov: np.ndarray, n: int) -> float:
        eigvals = np.real(np.linalg.eigvals(cov))
        mean_ev = np.sum(eigvals) / ((n + 1) ** 2)
        if abs(mean_ev) <= 1e-12:
            return 1.0
        g0 = 2 * (((n + 1) ** 2) - 1)
        g = (1.0 / mean_ev) * np.sum(np.abs(eigvals - mean_ev))
        return float(np.clip(1.0 - g / np.maximum(g0, 1e-12), 0.0, 1.0))

    diff = _cmd_from_cov(c, order)
    diff_ord = np.zeros(order, dtype=float)
    for n in range(1, order):
        c_n = c[: (n + 1) ** 2, : (n + 1) ** 2]
        diff_ord[n - 1] = _cmd_from_cov(c_n, n)
    if order >= 1:
        diff_ord[order - 1] = diff
    return diff, diff_ord
