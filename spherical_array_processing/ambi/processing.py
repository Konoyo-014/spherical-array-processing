"""Ambisonic signal-order and level-processing helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spec import channel_count, infer_order, order_channel_slices


def move_channel_axis(data: ArrayLike, source: int, destination: int) -> NDArray:
    """Move an Ambisonic channel axis without changing data values."""

    return np.moveaxis(np.asarray(data), source, destination)


def ensure_channel_axis(data: ArrayLike, *, axis: int = -1, max_order: int | None = None) -> tuple[NDArray, int]:
    """Validate an Ambisonic channel axis and return array plus inferred order."""

    arr = np.asarray(data)
    if arr.ndim == 0:
        raise ValueError("data must have at least one dimension")
    ch_axis = int(axis) % arr.ndim
    order = infer_order(int(arr.shape[ch_axis]))
    if max_order is not None and int(max_order) != order:
        raise ValueError(f"data has max_order={order}, expected {max_order}")
    return arr, order


def truncate_order(data: ArrayLike, target_order: int, *, axis: int = -1) -> NDArray:
    """Truncate Ambisonic coefficients to a lower order."""

    arr, order = ensure_channel_axis(data, axis=axis)
    target = int(target_order)
    if target < 0 or target > order:
        raise ValueError("target_order must satisfy 0 <= target_order <= current order")
    moved = np.moveaxis(arr, axis, -1)
    out = moved[..., : channel_count(target)]
    return np.moveaxis(out, -1, axis)


def pad_order(data: ArrayLike, target_order: int, *, axis: int = -1, fill_value: float = 0.0) -> NDArray:
    """Zero-pad Ambisonic coefficients to a higher order."""

    arr, order = ensure_channel_axis(data, axis=axis)
    target = int(target_order)
    if target < order:
        raise ValueError("target_order must be >= current order")
    moved = np.moveaxis(arr, axis, -1)
    out = np.full(moved.shape[:-1] + (channel_count(target),), fill_value, dtype=moved.dtype)
    out[..., : moved.shape[-1]] = moved
    return np.moveaxis(out, -1, axis)


def change_order(data: ArrayLike, target_order: int, *, axis: int = -1) -> NDArray:
    """Truncate or pad Ambisonic data to a target order."""

    arr, order = ensure_channel_axis(data, axis=axis)
    if int(target_order) <= order:
        return truncate_order(arr, target_order, axis=axis)
    return pad_order(arr, target_order, axis=axis)


def ambi_peak(data: ArrayLike) -> float:
    """Peak absolute Ambisonic sample/coefficient value."""

    return float(np.max(np.abs(np.asarray(data))))


def ambi_rms(data: ArrayLike, *, axis: int | tuple[int, ...] | None = None) -> float | NDArray[np.float64]:
    """Root-mean-square Ambisonic value."""

    out = np.sqrt(np.mean(np.abs(np.asarray(data)) ** 2, axis=axis))
    if np.asarray(out).ndim == 0:
        return float(out)
    return np.asarray(out, dtype=float)


def normalize_peak(data: ArrayLike, *, target_peak: float = 1.0, eps: float = 1e-15) -> NDArray:
    """Scale Ambisonic data to a target peak value."""

    arr = np.asarray(data)
    peak = ambi_peak(arr)
    if peak <= float(eps):
        return arr.copy()
    return arr * (float(target_peak) / peak)


def normalize_rms(data: ArrayLike, *, target_rms: float = 1.0, eps: float = 1e-15) -> NDArray:
    """Scale Ambisonic data to a target RMS value."""

    arr = np.asarray(data)
    rms = float(ambi_rms(arr))
    if rms <= float(eps):
        return arr.copy()
    return arr * (float(target_rms) / rms)


def w_channel(data: ArrayLike, *, axis: int = -1) -> NDArray:
    """Extract the ACN 0 / W channel."""

    arr, _order = ensure_channel_axis(data, axis=axis)
    return np.take(arr, 0, axis=axis)


def mono_from_w(data: ArrayLike, *, axis: int = -1, w_gain: float = 1.0) -> NDArray:
    """Downmix Ambisonics to mono using the W channel."""

    return np.asarray(w_channel(data, axis=axis)) * float(w_gain)


def apply_channel_gains(data: ArrayLike, gains: ArrayLike, *, axis: int = -1) -> NDArray:
    """Apply per-channel gains along an Ambisonic channel axis."""

    arr, order = ensure_channel_axis(data, axis=axis)
    g = np.asarray(gains, dtype=float).reshape(-1)
    if g.size != channel_count(order):
        raise ValueError("gains length must match Ambisonic channel count")
    moved = np.moveaxis(arr, axis, -1)
    out = moved * g
    return np.moveaxis(out, -1, axis)


def per_order_gains(max_order: int, gains: ArrayLike) -> NDArray[np.float64]:
    """Expand one gain per Ambisonic order to ACN channel gains."""

    g = np.asarray(gains, dtype=float).reshape(-1)
    if g.size != int(max_order) + 1:
        raise ValueError("gains must contain max_order + 1 values")
    return np.concatenate([np.full(2 * n + 1, g[n]) for n in range(int(max_order) + 1)])


def apply_per_order_gains(data: ArrayLike, gains: ArrayLike, *, axis: int = -1) -> NDArray:
    """Apply one gain per Ambisonic order."""

    arr, order = ensure_channel_axis(data, axis=axis)
    return apply_channel_gains(arr, per_order_gains(order, gains), axis=axis)


def order_rms(data: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """RMS value per Ambisonic order."""

    arr, order = ensure_channel_axis(data, axis=axis)
    moved = np.moveaxis(arr, axis, -1)
    return np.asarray(
        [np.sqrt(np.mean(np.abs(moved[..., s]) ** 2)) for s in order_channel_slices(order)],
        dtype=float,
    )


def order_balance_db(data: ArrayLike, *, axis: int = -1, eps: float = 1e-15) -> NDArray[np.float64]:
    """Per-order RMS balance in dB relative to order 0."""

    rms = np.maximum(order_rms(data, axis=axis), float(eps))
    return 20.0 * np.log10(rms / rms[0])


def channel_covariance(data: ArrayLike, *, axis: int = -1) -> NDArray[np.complex128]:
    """Channel covariance matrix over all non-channel samples."""

    arr, _order = ensure_channel_axis(data, axis=axis)
    ch_axis = int(axis) % arr.ndim
    moved = np.moveaxis(arr, ch_axis, 0).reshape(arr.shape[ch_axis], -1)
    return (moved @ moved.conj().T) / max(1, moved.shape[1])


def channel_correlation(data: ArrayLike, *, axis: int = -1, eps: float = 1e-15) -> NDArray[np.complex128]:
    """Normalised channel correlation matrix."""

    cov = channel_covariance(data, axis=axis)
    denom = np.sqrt(np.maximum(np.real(np.diag(cov)), float(eps)))
    return cov / (denom[:, None] * denom[None, :])


def mix_ambi_frames(frames: ArrayLike, *, weights: ArrayLike | None = None, axis: int = 0) -> NDArray:
    """Weighted mix of a stack of Ambisonic frames."""

    arr = np.asarray(frames)
    mix_axis = int(axis) % arr.ndim
    if weights is None:
        return np.mean(arr, axis=mix_axis)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != arr.shape[mix_axis]:
        raise ValueError("weights must match mix axis length")
    moved = np.moveaxis(arr, mix_axis, 0)
    return np.tensordot(w / np.sum(w), moved, axes=(0, 0))


def fade_ambi_frames(first: ArrayLike, second: ArrayLike, alpha: float) -> NDArray:
    """Linear crossfade between two Ambisonic arrays."""

    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    x = np.asarray(first)
    y = np.asarray(second)
    if x.shape != y.shape:
        raise ValueError("first and second must have matching shape")
    return (1.0 - a) * x + a * y


__all__ = [
    "ambi_peak",
    "ambi_rms",
    "apply_channel_gains",
    "apply_per_order_gains",
    "change_order",
    "channel_correlation",
    "channel_covariance",
    "ensure_channel_axis",
    "fade_ambi_frames",
    "mix_ambi_frames",
    "mono_from_w",
    "move_channel_axis",
    "normalize_peak",
    "normalize_rms",
    "order_balance_db",
    "order_rms",
    "pad_order",
    "per_order_gains",
    "truncate_order",
    "w_channel",
]
