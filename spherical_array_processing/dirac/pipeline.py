"""Time-domain DirAC rendering — STFT → analyse → synthesise → ISTFT.

A convenience wrapper that lets users go from an ambisonic
time-domain signal straight to a loudspeaker (or virtual-speaker)
signal without having to hand-manage the STFT frame bookkeeping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..stft import istft, stft
from ..types import SphericalGrid
from .analysis import dirac_analysis
from .synthesis import dirac_synthesize


def _looks_like_sh_channel_count(size: int) -> bool:
    """Return whether *size* can be ``(N+1)^2`` for ``N >= 1``."""
    n = int(size)
    if n < 4:
        return False
    root = int(round(np.sqrt(n)))
    return root * root == n


def dirac_render_time_domain(
    ambi_signal: ArrayLike,
    fs: float,
    loudspeaker_grid: SphericalGrid,
    *,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: Any = "hann",
    smoothing_alpha: float = 0.1,
    decorrelate_diffuse: bool = True,
    imaginary_loudspeakers: SphericalGrid | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Render a time-domain ambisonic signal to loudspeakers via DirAC.

    Parameters
    ----------
    ambi_signal : array_like, shape (Q, T) or (T, Q)
        Ambisonic signal with ``Q ≥ 4`` SH channels (W/Y/Z/X in the
        first four entries, ACN ordering).  Real-valued.  Axis
        ordering is auto-detected from shape; the larger axis is
        assumed to be time.
    fs : float
        Sampling rate in Hz.
    loudspeaker_grid : SphericalGrid
        Target loudspeaker directions.  Must form a 3-D convex hull;
        use *imaginary_loudspeakers* for hemispherical layouts.
    nperseg, noverlap, window : passed to :func:`~spherical_array_processing.stft.stft`.
    smoothing_alpha : float, optional
        IIR smoothing pole for :func:`dirac_analysis`.
    decorrelate_diffuse : bool, optional
        Forwarded to :func:`dirac_synthesize`.
    imaginary_loudspeakers : SphericalGrid or None, optional
        Auxiliary speakers for VBAP hull closure; see
        :func:`spherical_array_processing.decoding.vbap_gains`.
    rng : numpy.random.Generator or None, optional
        RNG for the diffuse-decorrelation phase dither.

    Returns
    -------
    ndarray, shape (L, T_out)
        Time-domain loudspeaker signal.  ``T_out`` matches the length
        produced by :func:`scipy.signal.istft` with the given STFT
        parameters and may differ from the input ``T`` by up to one
        hop due to the COLA boundary padding.
    """
    a = np.asarray(ambi_signal, dtype=float)
    if a.ndim != 2:
        raise ValueError(
            f"ambi_signal must be 2-D (Q, T) or (T, Q); got {a.shape}"
        )
    # Put channels on axis 0 for the STFT wrapper.
    # Prefer the axis whose length looks like an SH channel count
    # ``(N+1)^2`` so short signals such as ``(Q=4, T=2)`` are not
    # mis-identified as time-major solely because ``Q > T``.
    axis0_is_channels = _looks_like_sh_channel_count(a.shape[0])
    axis1_is_channels = _looks_like_sh_channel_count(a.shape[1])
    if axis0_is_channels and not axis1_is_channels:
        channels_first = True
        ambi = a
    elif axis1_is_channels and not axis0_is_channels:
        channels_first = False
        ambi = a.T
    elif a.shape[0] <= a.shape[1]:
        channels_first = True
        ambi = a
    else:
        channels_first = False
        ambi = a.T
    if ambi.shape[0] < 4:
        raise ValueError(
            "ambi_signal must carry at least 4 SH channels (W/Y/Z/X)"
        )

    freqs, _, Z = stft(
        ambi, fs, nperseg=nperseg, noverlap=noverlap, window=window
    )
    # Z layout: (F, M, T) as documented by sap.stft.stft.
    params = dirac_analysis(
        Z, freqs, smoothing_alpha=smoothing_alpha, coeff_axis=1
    )
    spk_stft = dirac_synthesize(
        params,
        loudspeaker_grid,
        imaginary_loudspeakers=imaginary_loudspeakers,
        decorrelate_diffuse=decorrelate_diffuse,
        rng=rng,
    )
    # spk_stft shape: (F, L, T).  Drop back to time domain.
    _, out = istft(
        spk_stft, fs, nperseg=nperseg, noverlap=noverlap, window=window
    )
    # istft multichannel returns (L, T_out).
    if not channels_first:
        out = out.T
    return np.asarray(out, dtype=float)


__all__ = ["dirac_render_time_domain"]
