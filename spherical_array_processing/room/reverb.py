"""Apply a spherical-harmonic room impulse response to a dry signal.

The monaural → ambisonic convolution is the standard way to turn a
dry mono source recording plus a geometry-derived SH-RIR (from
:func:`shoebox_sh_rir`, a measured SOFA-style multichannel IR, or any
other source) into a reverberant ambisonic capture at the listener
position.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import oaconvolve


def convolve_mono_to_ambi(
    dry_signal: ArrayLike,
    sh_rir: ArrayLike,
    *,
    axis: int = -1,
) -> NDArray[np.floating]:
    """Convolve a monaural dry signal with an ambisonic SH-RIR.

    For each SH channel ``q`` the output is
    ``y_q[t] = (dry * h_q)[t]`` (full convolution).  The dry signal can
    carry batch dimensions in front of the time axis — they're
    broadcast over the SH channels.

    Parameters
    ----------
    dry_signal : array_like
        Monaural dry signal.  The last axis (by default) indexes time.
        Extra leading axes are allowed for batched processing.
    sh_rir : array_like, shape ``(Q, T_rir)``
        Ambisonic room impulse response.  ``Q = (N+1)²`` channels in
        ACN order.
    axis : int, optional
        Time axis of *dry_signal*.  Default ``-1``.

    Returns
    -------
    ndarray
        Reverberant ambisonic signal.  If *dry_signal* has shape
        ``(..., T)``, the output has shape ``(..., Q, T + T_rir − 1)``
        (the SH channel axis is inserted just before the time axis).
    """
    dry = np.asarray(dry_signal)
    rir = np.asarray(sh_rir)
    if rir.ndim != 2:
        raise ValueError(
            f"sh_rir must be 2-D (Q, T_rir); got shape {rir.shape}"
        )
    q, t_rir = rir.shape
    if dry.ndim == 0:
        raise ValueError("dry_signal must have at least one axis (time)")
    ax_t = axis % dry.ndim
    # Normalise the time axis of ``dry`` to be the last axis.
    dry_t = np.moveaxis(dry, ax_t, -1)
    # Expand to broadcast a new SH axis just before time.
    dry_for_conv = dry_t[..., None, :]                  # (..., 1, T)
    rir_for_conv = rir.reshape((1,) * (dry_for_conv.ndim - 2) + rir.shape)
    out = oaconvolve(dry_for_conv, rir_for_conv, mode="full", axes=-1)
    # ``out`` now has shape ``batch_axes + (Q, T_out)``, where
    # ``batch_axes`` are the original ``dry`` axes with the time axis
    # removed.  Restore the user's axis order by inserting the new SH
    # axis at the original time-axis position and the convolved time
    # axis immediately after it.
    result = np.moveaxis(out, (-2, -1), (ax_t, ax_t + 1))
    return result


def convolve_sh_to_sh(
    sh_signal: ArrayLike,
    sh_rir: ArrayLike,
    *,
    signal_axis_channels: int = 0,
    signal_axis_time: int = 1,
) -> NDArray[np.floating]:
    """Apply a diagonal SH-domain impulse response to an SH signal.

    Convolves each SH channel of *sh_signal* with the matching SH
    channel of *sh_rir*.  This is the right thing when the RIR is a
    *channel-diagonal* room/monitor response — e.g. per-order
    equalisation curves or a matched-direction ambisonic capture —
    and not a general plane-wave-domain room response matrix.

    Parameters
    ----------
    sh_signal : array_like, shape ``(Q, T)`` (default layout).
    sh_rir : array_like, shape ``(Q, T_rir)``.
    signal_axis_channels, signal_axis_time : int, optional
        Override the axes of *sh_signal* if it is not ``(Q, T)``.

    Returns
    -------
    ndarray, shape ``(Q, T + T_rir − 1)`` with the user's axis order
    preserved.
    """
    sig = np.asarray(sh_signal)
    rir = np.asarray(sh_rir)
    if rir.ndim != 2:
        raise ValueError(
            f"sh_rir must be 2-D (Q, T_rir); got shape {rir.shape}"
        )
    if sig.ndim < 2:
        raise ValueError(
            "sh_signal must have at least two axes (channels, time)"
        )
    # Reorder signal to ``batch_axes + (Q, T)`` for the convolution.
    ax_q = signal_axis_channels % sig.ndim
    ax_t = signal_axis_time % sig.ndim
    if ax_q == ax_t:
        raise ValueError("signal_axis_channels and signal_axis_time must differ")
    sig_qt = np.moveaxis(sig, (ax_q, ax_t), (-2, -1))
    if sig_qt.shape[-2] != rir.shape[0]:
        raise ValueError(
            f"channel-count mismatch: signal has {sig_qt.shape[-2]} SH "
            f"channels, RIR has {rir.shape[0]}"
        )
    # Batched convolution along the last axis.
    rir_qt = rir.reshape((1,) * (sig_qt.ndim - 2) + rir.shape)
    out_qt = oaconvolve(sig_qt, rir_qt, mode="full", axes=-1)
    # Put axes back.
    return np.moveaxis(out_qt, (-2, -1), (ax_q, ax_t))


__all__ = ["convolve_mono_to_ambi", "convolve_sh_to_sh"]
