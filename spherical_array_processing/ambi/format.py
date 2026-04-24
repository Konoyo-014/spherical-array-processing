"""SH coefficient format conversions (ACN/FuMa, orthonormal/N3D/SN3D).

The coefficient-level scaling laws follow directly from the SH
normalisation definitions:

* Orthonormal: ``∫|Y_nm_ortho|² dΩ = 1``.
* N3D:         ``∫|Y_nm_N3D|²  dΩ = 4π``        ⇒ ``Y_N3D = √(4π) · Y_ortho``.
* SN3D:        ``∫|Y_nm_SN3D|² dΩ = 4π / (2n+1)``⇒ ``Y_SN3D = Y_N3D / √(2n+1)``.

Because the physical field ``f = Σ c · Y`` is invariant, the
coefficients scale *inversely* to the basis:

* ``c_N3D  = c_ortho / √(4π)``
* ``c_SN3D = c_ortho · √(2n+1) / √(4π)``
* ``c_SN3D = c_N3D   · √(2n+1)``

FuMa combines an alternative channel ordering with per-channel
weighting derived from the ``0.707·W`` B-format scaling plus
second/third-order correction factors tabulated in the Malham 2003
specification (see module docstring of :mod:`..ambi`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


Normalization = Literal["orthonormal", "n3d", "sn3d"]

# FuMa channel order through 3rd order, expressed as ACN indices:
# FuMa positions 0..15 ↔ ACN positions in `_FUMA_TO_ACN`.
_FUMA_TO_ACN: tuple[int, ...] = (
    0,   # W  (n=0, m=0)
    3,   # X  (n=1, m=+1)
    1,   # Y  (n=1, m=-1)
    2,   # Z  (n=1, m=0)
    6,   # R  (n=2, m=0)
    7,   # S  (n=2, m=+1)
    5,   # T  (n=2, m=-1)
    8,   # U  (n=2, m=+2)
    4,   # V  (n=2, m=-2)
    12,  # K  (n=3, m=0)
    13,  # L  (n=3, m=+1)
    11,  # M  (n=3, m=-1)
    14,  # N  (n=3, m=+2)
    10,  # O  (n=3, m=-2)
    15,  # P  (n=3, m=+3)
    9,   # Q  (n=3, m=-3)
)

# FuMa normalisation weights at each FuMa channel, defined as the
# factor that takes SN3D (ACN-permuted to FuMa order) TO FuMa:
#
# ``c_fuma[i] = c_sn3d_reordered[i] · _FUMA_WEIGHTS_FROM_SN3D[i]``.
#
# Source: Malham 2003, AmbiX paper (Nachbar et al. 2011, Table 1).
_FUMA_WEIGHTS_FROM_SN3D: tuple[float, ...] = (
    1.0 / np.sqrt(2.0),    # W
    1.0, 1.0, 1.0,          # X, Y, Z
    1.0,                    # R
    2.0 / np.sqrt(3.0), 2.0 / np.sqrt(3.0),          # S, T
    2.0 / np.sqrt(3.0), 2.0 / np.sqrt(3.0),          # U, V
    1.0,                    # K
    np.sqrt(45.0 / 32.0), np.sqrt(45.0 / 32.0),      # L, M
    3.0 / np.sqrt(5.0), 3.0 / np.sqrt(5.0),          # N, O
    np.sqrt(8.0 / 5.0), np.sqrt(8.0 / 5.0),          # P, Q
)


def _scaling_vector(max_order: int, from_: Normalization, to: Normalization) -> NDArray[np.float64]:
    """Return a length-``(N+1)²`` ACN-ordered per-channel scaling
    vector ``s`` such that ``c_to = s · c_from``.
    """
    n_ch = (int(max_order) + 1) ** 2
    if from_ == to:
        return np.ones(n_ch, dtype=float)
    # Per-channel n(q) = floor(sqrt(q)).
    q = np.arange(n_ch)
    n_of_q = np.floor(np.sqrt(q)).astype(int)
    sqrt_2np1 = np.sqrt(2 * n_of_q + 1).astype(float)
    sqrt_4pi = float(np.sqrt(4.0 * np.pi))
    # Factors relative to orthonormal basis.
    factor_ortho_to = {
        "orthonormal": np.ones(n_ch, dtype=float),
        "n3d": np.full(n_ch, 1.0 / sqrt_4pi, dtype=float),
        "sn3d": sqrt_2np1 / sqrt_4pi,
    }
    return factor_ortho_to[to] / factor_ortho_to[from_]


def convert_ambi_normalization(
    coeffs: ArrayLike,
    *,
    max_order: int,
    from_: Normalization,
    to: Normalization,
    axis: int = 0,
) -> NDArray:
    """Rescale SH coefficients between ``orthonormal``, ``n3d``, and
    ``sn3d`` conventions (all ACN-ordered).

    Parameters
    ----------
    coeffs : array_like
        SH coefficients with ``(max_order+1)²`` entries along *axis*.
    max_order : int
        SH order ``N``.
    from_, to : {"orthonormal", "n3d", "sn3d"}
        Source and target normalisation conventions.
    axis : int, optional
        Axis that indexes the SH channels.  Default ``0``.

    Returns
    -------
    ndarray
        Rescaled coefficients with the same shape / dtype kind as
        *coeffs*.

    Notes
    -----
    The package internally uses the ``"orthonormal"`` convention
    (``∫|Y|² dΩ = 1``), so a typical import / export sequence is::

        import_: c_sn3d → convert_ambi_normalization(..., from_="sn3d", to="orthonormal")
        export:  c_ortho → convert_ambi_normalization(..., from_="orthonormal", to="sn3d")
    """
    c = np.asarray(coeffs)
    scale = _scaling_vector(int(max_order), from_, to)
    expected = scale.size
    c_moved = np.moveaxis(c, axis, -1)
    if c_moved.shape[-1] != expected:
        raise ValueError(
            f"coeffs axis length {c_moved.shape[-1]} does not match "
            f"(max_order+1)² = {expected}"
        )
    out = c_moved * scale
    return np.moveaxis(out, -1, axis)


def _fuma_max_channels(max_order: int) -> int:
    n = int(max_order)
    if n < 0:
        raise ValueError("max_order must be non-negative")
    if n > 3:
        raise ValueError(
            "FuMa is only defined through 3rd order (16 channels); "
            f"got max_order={n}"
        )
    return (n + 1) ** 2


def acn_to_fuma(
    coeffs: ArrayLike,
    *,
    max_order: int,
    from_sn3d: bool = True,
    axis: int = 0,
) -> NDArray:
    """Convert ACN-ordered SH coefficients to the FuMa (B-format)
    channel order and normalisation.

    Parameters
    ----------
    coeffs : array_like
        ACN-ordered coefficients with ``(max_order+1)²`` entries along
        *axis*.  The input normalisation is controlled by *from_sn3d*:
        ``True`` assumes SN3D (AmbiX), ``False`` assumes orthonormal.
    max_order : int
        Ambisonic order ``N`` — must be ``0 ≤ N ≤ 3`` because FuMa is
        not standardised past third order.
    from_sn3d : bool, optional
        Interpret *coeffs* as SN3D (``True``, default — the
        AmbiX-to-FuMa path that most users need) or as the internal
        orthonormal basis (``False``).
    axis : int, optional
        Axis that indexes SH channels.  Default ``0``.

    Returns
    -------
    ndarray
        FuMa-ordered, FuMa-normalised coefficients with the same shape
        as *coeffs* except along *axis*, where the length is preserved
        (FuMa through order ``N`` has the same ``(N+1)²`` channel
        count as ACN).

    References
    ----------
    .. [1] D. Malham, "Second and Third Order Ambisonics — the
       Furse-Malham set", https://www.york.ac.uk/inst/mustech/3d_audio/secondor.html
    .. [2] C. Nachbar, F. Zotter, E. Deleflie, A. Sontacchi,
       "AmbiX — a Suggested Ambisonics Format", *AES 35th Int. Conf.*,
       2011, Table 1.
    """
    c = np.asarray(coeffs)
    n_ch = _fuma_max_channels(max_order)
    c_moved = np.moveaxis(c, axis, -1)
    if c_moved.shape[-1] != n_ch:
        raise ValueError(
            f"coeffs axis length {c_moved.shape[-1]} does not match "
            f"(max_order+1)² = {n_ch}"
        )
    # Step 1: ensure we have SN3D on input.
    if not from_sn3d:
        c_moved = convert_ambi_normalization(
            c_moved, max_order=int(max_order),
            from_="orthonormal", to="sn3d", axis=-1,
        )
    # Step 2: permute ACN → FuMa order.
    perm = np.asarray(_FUMA_TO_ACN[:n_ch], dtype=int)
    permuted = c_moved[..., perm]
    # Step 3: apply per-FuMa-channel weighting.
    weights = np.asarray(_FUMA_WEIGHTS_FROM_SN3D[:n_ch], dtype=float)
    fuma = permuted * weights
    return np.moveaxis(fuma, -1, axis)


def fuma_to_acn(
    coeffs: ArrayLike,
    *,
    max_order: int,
    to_sn3d: bool = True,
    axis: int = 0,
) -> NDArray:
    """Convert FuMa (B-format) coefficients to ACN ordering.

    Parameters
    ----------
    coeffs : array_like
        FuMa-ordered coefficients with ``(max_order+1)²`` entries
        along *axis*.
    max_order : int
        Ambisonic order ``N`` — must be ``0 ≤ N ≤ 3``.
    to_sn3d : bool, optional
        If ``True`` (default) return AmbiX-compatible SN3D.  If
        ``False`` return coefficients in the internal orthonormal
        basis.
    axis : int, optional
        Axis that indexes SH channels.  Default ``0``.

    Returns
    -------
    ndarray
        ACN-ordered coefficients in the chosen normalisation.
    """
    c = np.asarray(coeffs)
    n_ch = _fuma_max_channels(max_order)
    c_moved = np.moveaxis(c, axis, -1)
    if c_moved.shape[-1] != n_ch:
        raise ValueError(
            f"coeffs axis length {c_moved.shape[-1]} does not match "
            f"(max_order+1)² = {n_ch}"
        )
    # Undo FuMa weights → SN3D (still FuMa-ordered).
    weights = np.asarray(_FUMA_WEIGHTS_FROM_SN3D[:n_ch], dtype=float)
    unweighted = c_moved / weights
    # Invert the permutation: ACN index of fuma position i is
    # ``_FUMA_TO_ACN[i]``, so to get ACN-ordered coeffs we need
    # ``acn[_FUMA_TO_ACN[i]] = unweighted[i]``.
    inv = np.empty(n_ch, dtype=int)
    perm = np.asarray(_FUMA_TO_ACN[:n_ch], dtype=int)
    inv[perm] = np.arange(n_ch)
    acn = unweighted[..., inv]
    if not to_sn3d:
        acn = convert_ambi_normalization(
            acn, max_order=int(max_order),
            from_="sn3d", to="orthonormal", axis=-1,
        )
    return np.moveaxis(acn, -1, axis)


__all__ = [
    "acn_to_fuma",
    "convert_ambi_normalization",
    "fuma_to_acn",
]
