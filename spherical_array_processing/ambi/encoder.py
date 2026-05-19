"""Monaural → SH ambisonic plane-wave encoder.

For a monaural source ``s(t)`` at direction ``r̂``, the ideal ambisonic
encoding in a plane-wave basis is simply

``c_q(t) = Y_q(r̂) · s(t)``

where ``Y_q`` are the SH basis functions in ACN order.  When several
mono sources are provided with one direction each, their contributions
are **summed** into the output, which is the standard linear-ambisonics
mixdown.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sh import matrix as sh_matrix
from ..types import NormalizationKind, SHBasisSpec, SphericalGrid
from .format import convert_ambi_normalization
from .spec import AmbisonicFrame, AmbisonicSpec


BasisKind = Literal["real", "complex"]


def encode_plane_wave(
    mono_signal: ArrayLike,
    direction: SphericalGrid,
    *,
    max_order: int,
    basis: BasisKind = "real",
    normalization: NormalizationKind = "orthonormal",
) -> NDArray:
    """Encode one or more monaural signals as plane waves on an SH basis.

    Parameters
    ----------
    mono_signal : array_like
        Monaural source signal.  Shape ``(T,)`` for a single source,
        ``(K, T)`` for ``K`` sources (one per direction).  Real-valued.
    direction : SphericalGrid
        Source direction(s).  Must have ``size == 1`` (single source)
        or ``size == K`` (one direction per row of *mono_signal*).
    max_order : int
        Ambisonic order ``N``.  Output SH channel count is ``(N+1)²``.
    basis : {"real", "complex"}, optional
        SH basis to encode in.  ``"real"`` (default) returns a
        real-valued ``(Q, T)`` array; ``"complex"`` returns a
        ``complex128`` array.
    normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Output normalisation.  ``"orthonormal"`` (default) matches the
        package-internal convention; switch to ``"sn3d"`` to feed the
        result directly into an AmbiX pipeline.

    Returns
    -------
    ndarray, shape ``(Q, T)``
        Ambisonic signal with ``Q = (max_order+1)²`` SH channels.
        When ``K > 1`` the contributions from all sources are summed.
    """
    sig = np.asarray(mono_signal)
    if sig.ndim == 1:
        sig = sig[None, :]
    elif sig.ndim != 2:
        raise ValueError(
            f"mono_signal must be 1-D or 2-D; got shape {sig.shape}"
        )
    k_sources = sig.shape[0]
    if direction.size != k_sources:
        raise ValueError(
            f"direction has {direction.size} directions, but "
            f"mono_signal implies {k_sources}"
        )
    spec = SHBasisSpec(
        max_order=int(max_order),
        basis=basis,
        normalization="orthonormal",
    )
    y = np.asarray(sh_matrix(spec, direction))  # (K, Q)
    # sh_signal[q, t] = Σ_k Y_q(d_k) · s_k(t).
    out = y.T @ sig  # (Q, T)
    if normalization != "orthonormal":
        out = convert_ambi_normalization(
            out, max_order=int(max_order),
            from_="orthonormal", to=normalization, axis=0,
        )
    return out


def encode_plane_wave_frame(
    mono_signal: ArrayLike,
    direction: SphericalGrid,
    *,
    spec: AmbisonicSpec,
    sample_rate_hz: float | None = None,
    metadata: dict | None = None,
) -> AmbisonicFrame:
    """Encode one or more mono plane waves and return an AmbisonicFrame.

    This is the container-preserving counterpart to
    :func:`encode_plane_wave`.  The returned frame stores channels on
    axis 0, matching the low-level encoder's ``(Q, T)`` output.
    """
    data = encode_plane_wave(
        mono_signal,
        direction,
        max_order=spec.max_order,
        basis=spec.basis,
        normalization=spec.normalization,
    )
    if spec.mixed_order_mask is not None:
        data = data.copy()
        data[~spec.mixed_order_mask, ...] = 0
    return AmbisonicFrame(
        data,
        spec.with_domain("time"),
        channel_axis=0,
        sample_rate_hz=sample_rate_hz,
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = ["encode_plane_wave", "encode_plane_wave_frame"]
