"""Ambisonic channel tables, labels, masks, and mixed-order helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sh import acn_index, acn_to_nm, degree_order_pairs
from .spec import channel_count, infer_order, order_channel_slices


ChannelLabelStyle = Literal["acn", "nm", "fuma", "sid"]
MixedOrderShape = Literal["full3d", "horizontal", "periphonic", "mixed"]


_FUMA_LABELS: tuple[str, ...] = (
    "W", "X", "Y", "Z",
    "R", "S", "T", "U", "V",
    "K", "L", "M", "N", "O", "P", "Q",
)
_FUMA_TO_ACN: tuple[int, ...] = (
    0, 3, 1, 2, 6, 7, 5, 8, 4, 12, 13, 11, 14, 10, 15, 9,
)
_ACN_TO_FUMA_LABEL = {
    acn: _FUMA_LABELS[fuma_idx]
    for fuma_idx, acn in enumerate(_FUMA_TO_ACN)
}


@dataclass(frozen=True)
class AmbisonicChannel:
    """Metadata for one ACN-indexed Ambisonic channel."""

    acn: int
    degree: int
    order: int
    label_acn: str
    label_nm: str
    label_fuma: str | None = None
    is_horizontal: bool = False
    is_zonal: bool = False
    is_sectoral: bool = False


@dataclass(frozen=True)
class MixedOrderSpec:
    """Horizontal and vertical order limits for mixed-order Ambisonics."""

    horizontal_order: int
    vertical_order: int

    def __post_init__(self) -> None:
        if self.horizontal_order < 0 or self.vertical_order < 0:
            raise ValueError("mixed-order limits must be non-negative")
        if self.vertical_order > self.horizontal_order:
            raise ValueError("vertical_order must be <= horizontal_order")

    @property
    def max_order(self) -> int:
        return int(self.horizontal_order)

    @property
    def n_channels(self) -> int:
        return int(np.count_nonzero(mixed_order_mask(self.horizontal_order, self.vertical_order)))

    def mask(self) -> NDArray[np.bool_]:
        """Return the ACN mask represented by this mixed-order spec."""

        return mixed_order_mask(self.horizontal_order, self.vertical_order)


def channel_degrees(max_order: int) -> NDArray[np.int64]:
    """Return the Ambisonic degree ``n`` for every ACN channel."""

    return degree_order_pairs(int(max_order))[:, 0]


def channel_orders(max_order: int) -> NDArray[np.int64]:
    """Return the signed Ambisonic order ``m`` for every ACN channel."""

    return degree_order_pairs(int(max_order))[:, 1]


def channel_degree_order(max_order: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return degree and signed-order vectors in ACN order."""

    pairs = degree_order_pairs(int(max_order))
    return pairs[:, 0], pairs[:, 1]


def acn_sequence(max_order: int) -> NDArray[np.int64]:
    """Return ``0..(N+1)^2-1`` for an Ambisonic order."""

    return np.arange(channel_count(int(max_order)), dtype=np.int64)


def channel_labels(max_order: int, *, style: ChannelLabelStyle = "acn") -> tuple[str, ...]:
    """Return channel labels in ACN order."""

    pairs = degree_order_pairs(int(max_order))
    labels: list[str] = []
    for acn, (n, m) in enumerate(pairs):
        if style == "acn":
            labels.append(f"ACN{acn}")
        elif style == "nm":
            labels.append(f"Y{int(n)}_{int(m):+d}")
        elif style == "sid":
            labels.append(f"{int(n)}.{int(m):+d}")
        elif style == "fuma":
            labels.append(_ACN_TO_FUMA_LABEL.get(acn, f"ACN{acn}"))
        else:
            raise ValueError("style must be 'acn', 'nm', 'sid', or 'fuma'")
    return tuple(labels)


def fuma_channel_labels(max_order: int = 3) -> tuple[str, ...]:
    """Return legacy FuMa labels through third order."""

    n_ch = channel_count(int(max_order))
    if max_order < 0 or max_order > 3:
        raise ValueError("FuMa labels are only defined through third order")
    return _FUMA_LABELS[:n_ch]


def channel_table(max_order: int) -> tuple[AmbisonicChannel, ...]:
    """Return a typed ACN channel table."""

    pairs = degree_order_pairs(int(max_order))
    labels_fuma = channel_labels(max_order, style="fuma")
    rows: list[AmbisonicChannel] = []
    for acn, (n_i, m_i) in enumerate(pairs):
        n = int(n_i)
        m = int(m_i)
        rows.append(
            AmbisonicChannel(
                acn=acn,
                degree=n,
                order=m,
                label_acn=f"ACN{acn}",
                label_nm=f"Y{n}_{m:+d}",
                label_fuma=labels_fuma[acn] if acn < len(_FUMA_LABELS) else None,
                is_horizontal=(m != 0),
                is_zonal=(m == 0),
                is_sectoral=(abs(m) == n),
            )
        )
    return tuple(rows)


def channel_index(degree: int, order: int) -> int:
    """Validate and return the ACN index for ``(degree, order)``."""

    n = int(degree)
    m = int(order)
    if n < 0:
        raise ValueError("degree must be non-negative")
    if abs(m) > n:
        raise ValueError("order must satisfy -degree <= order <= degree")
    return int(acn_index(n, m))


def channel_name(acn: int, *, style: ChannelLabelStyle = "nm") -> str:
    """Name a single ACN channel."""

    idx = int(acn)
    if idx < 0:
        raise ValueError("acn must be non-negative")
    n, m = acn_to_nm(idx)
    if style == "acn":
        return f"ACN{idx}"
    if style == "nm":
        return f"Y{int(n)}_{int(m):+d}"
    if style == "sid":
        return f"{int(n)}.{int(m):+d}"
    if style == "fuma":
        return _ACN_TO_FUMA_LABEL.get(idx, f"ACN{idx}")
    raise ValueError("style must be 'acn', 'nm', 'sid', or 'fuma'")


def channel_label_to_acn(label: str) -> int:
    """Parse an ACN/FuMa/``Y_n_m``/SID channel label to an ACN index."""

    text = str(label).strip()
    upper = text.upper()
    if upper.startswith("ACN"):
        return int(upper[3:])
    if upper in _FUMA_LABELS:
        return int(_FUMA_TO_ACN[_FUMA_LABELS.index(upper)])
    if text.startswith("Y") and "_" in text:
        n_s, m_s = text[1:].split("_", 1)
        return channel_index(int(n_s), int(m_s))
    if "." in text:
        n_s, m_s = text.split(".", 1)
        return channel_index(int(n_s), int(m_s))
    raise ValueError(f"cannot parse Ambisonic channel label {label!r}")


def first_acn_for_degree(degree: int) -> int:
    """First ACN index for degree ``n``."""

    n = int(degree)
    if n < 0:
        raise ValueError("degree must be non-negative")
    return n * n


def last_acn_for_degree(degree: int) -> int:
    """Last ACN index for degree ``n``."""

    n = int(degree)
    if n < 0:
        raise ValueError("degree must be non-negative")
    return (n + 1) * (n + 1) - 1


def order_block_start(degree: int) -> int:
    """Alias for :func:`first_acn_for_degree`."""

    return first_acn_for_degree(degree)


def order_block_stop(degree: int) -> int:
    """Exclusive stop index for one Ambisonic degree block."""

    return last_acn_for_degree(degree) + 1


def order_block_slice(degree: int) -> slice:
    """ACN slice for one Ambisonic degree block."""

    return slice(order_block_start(degree), order_block_stop(degree))


def per_order_channel_counts(max_order: int) -> NDArray[np.int64]:
    """Number of channels in each Ambisonic degree block."""

    n = int(max_order)
    if n < 0:
        raise ValueError("max_order must be non-negative")
    return np.asarray([2 * k + 1 for k in range(n + 1)], dtype=np.int64)


def validate_channel_indices(indices: ArrayLike, max_order: int) -> NDArray[np.int64]:
    """Validate ACN indices for a maximum order."""

    idx = np.asarray(indices, dtype=int).reshape(-1)
    n_ch = channel_count(int(max_order))
    if np.any(idx < 0) or np.any(idx >= n_ch):
        raise ValueError("channel indices out of range")
    return idx.astype(np.int64)


def validate_channel_mask(mask: ArrayLike, max_order: int | None = None) -> NDArray[np.bool_]:
    """Validate and return a boolean ACN mask."""

    active = np.asarray(mask, dtype=bool).reshape(-1)
    if max_order is None:
        infer_order(active.size)
    elif active.size != channel_count(int(max_order)):
        raise ValueError("mask length must match max_order")
    return active


def mask_to_indices(mask: ArrayLike) -> NDArray[np.int64]:
    """Convert a boolean channel mask to ACN indices."""

    active = validate_channel_mask(mask)
    return np.nonzero(active)[0].astype(np.int64)


def indices_to_mask(indices: ArrayLike, max_order: int) -> NDArray[np.bool_]:
    """Convert ACN indices to a boolean channel mask."""

    idx = validate_channel_indices(indices, int(max_order))
    mask = np.zeros(channel_count(int(max_order)), dtype=bool)
    mask[idx] = True
    return mask


def validate_ambisonic_channel_axis(
    data: ArrayLike,
    *,
    axis: int = -1,
    max_order: int | None = None,
) -> tuple[int, int]:
    """Validate an Ambisonic channel axis and return ``(axis, max_order)``."""

    arr = np.asarray(data)
    if arr.ndim == 0:
        raise ValueError("data must have at least one dimension")
    ch_axis = int(axis) % arr.ndim
    try:
        inferred = infer_order(int(arr.shape[ch_axis]))
    except ValueError as exc:
        raise ValueError(f"channel axis has invalid Ambisonic length: {exc}") from exc
    if max_order is not None and inferred != int(max_order):
        raise ValueError(
            f"channel axis implies max_order={inferred}, expected {max_order}"
        )
    return ch_axis, inferred


def degree_mask(max_order: int, degree: int) -> NDArray[np.bool_]:
    """Boolean mask for one Ambisonic degree ``n``."""

    n = int(degree)
    if n < 0 or n > int(max_order):
        raise ValueError("degree must satisfy 0 <= degree <= max_order")
    mask = np.zeros(channel_count(int(max_order)), dtype=bool)
    mask[n * n : (n + 1) * (n + 1)] = True
    return mask


def degree_channel_indices(max_order: int, degree: int) -> NDArray[np.int64]:
    """ACN indices for one Ambisonic degree."""

    return mask_to_indices(degree_mask(max_order, degree))


def signed_order_mask(max_order: int, order: int) -> NDArray[np.bool_]:
    """Boolean mask for all channels with signed order ``m``."""

    m = int(order)
    pairs = degree_order_pairs(int(max_order))
    if abs(m) > int(max_order):
        raise ValueError("abs(order) must be <= max_order")
    return pairs[:, 1] == m


def signed_order_channel_indices(max_order: int, order: int) -> NDArray[np.int64]:
    """ACN indices for a signed Ambisonic order ``m``."""

    return mask_to_indices(signed_order_mask(max_order, order))


def horizontal_channel_mask(max_order: int) -> NDArray[np.bool_]:
    """Mask channels with non-zero signed order ``m``."""

    return channel_orders(int(max_order)) != 0


def horizontal_channel_indices(max_order: int) -> NDArray[np.int64]:
    """ACN indices with non-zero signed order ``m``."""

    return mask_to_indices(horizontal_channel_mask(max_order))


def zonal_channel_mask(max_order: int) -> NDArray[np.bool_]:
    """Mask zonal channels, i.e. ``m == 0``."""

    return channel_orders(int(max_order)) == 0


def zonal_channel_indices(max_order: int) -> NDArray[np.int64]:
    """ACN indices of zonal channels."""

    return mask_to_indices(zonal_channel_mask(max_order))


def sectoral_channel_mask(max_order: int) -> NDArray[np.bool_]:
    """Mask sectoral channels, i.e. ``|m| == n``."""

    n, m = channel_degree_order(int(max_order))
    return np.abs(m) == n


def sectoral_channel_indices(max_order: int) -> NDArray[np.int64]:
    """ACN indices of sectoral channels."""

    return mask_to_indices(sectoral_channel_mask(max_order))


def omni_channel_mask(max_order: int) -> NDArray[np.bool_]:
    """Mask containing only the zeroth-order omni channel."""

    mask = np.zeros(channel_count(int(max_order)), dtype=bool)
    mask[0] = True
    return mask


def non_omni_channel_mask(max_order: int) -> NDArray[np.bool_]:
    """Mask containing every channel except ACN 0."""

    mask = np.ones(channel_count(int(max_order)), dtype=bool)
    mask[0] = False
    return mask


def first_order_channel_mask(max_order: int) -> NDArray[np.bool_]:
    """Mask for first-order channels when present."""

    if int(max_order) < 1:
        return np.zeros(channel_count(int(max_order)), dtype=bool)
    return degree_mask(int(max_order), 1)


def truncate_channel_mask(max_order: int, target_order: int) -> NDArray[np.bool_]:
    """Mask that keeps all channels up to ``target_order``."""

    n = int(max_order)
    target = int(target_order)
    if target < 0 or target > n:
        raise ValueError("target_order must satisfy 0 <= target_order <= max_order")
    mask = np.zeros(channel_count(n), dtype=bool)
    mask[: channel_count(target)] = True
    return mask


def invert_channel_mask(mask: ArrayLike) -> NDArray[np.bool_]:
    """Invert a valid Ambisonic channel mask."""

    return ~validate_channel_mask(mask)


def combine_channel_masks(*masks: ArrayLike, mode: Literal["or", "and", "xor"] = "or") -> NDArray[np.bool_]:
    """Combine multiple channel masks with a logical operation."""

    if not masks:
        raise ValueError("at least one mask is required")
    arrays = [validate_channel_mask(mask) for mask in masks]
    first_shape = arrays[0].shape
    if any(mask.shape != first_shape for mask in arrays):
        raise ValueError("all masks must have matching shape")
    if mode == "or":
        return np.logical_or.reduce(arrays)
    if mode == "and":
        return np.logical_and.reduce(arrays)
    if mode == "xor":
        return np.logical_xor.reduce(arrays)
    raise ValueError("mode must be 'or', 'and', or 'xor'")


def acn_to_fuma_permutation(max_order: int = 3) -> NDArray[np.int64]:
    """Indices that reorder ACN channels into FuMa channel order."""

    n = int(max_order)
    if n < 0 or n > 3:
        raise ValueError("FuMa channel order is only defined through third order")
    return np.asarray(_FUMA_TO_ACN[: channel_count(n)], dtype=np.int64)


def fuma_to_acn_permutation(max_order: int = 3) -> NDArray[np.int64]:
    """Indices that reorder FuMa channels into ACN channel order."""

    return np.argsort(acn_to_fuma_permutation(max_order)).astype(np.int64)


def reorder_acn_to_fuma(data: ArrayLike, *, max_order: int | None = None, axis: int = -1) -> NDArray:
    """Reorder an ACN-layout tensor to FuMa channel order."""

    arr = np.asarray(data)
    ch_axis, order = validate_ambisonic_channel_axis(arr, axis=axis, max_order=max_order)
    perm = acn_to_fuma_permutation(order)
    return np.take(arr, perm, axis=ch_axis)


def reorder_fuma_to_acn(data: ArrayLike, *, max_order: int | None = None, axis: int = -1) -> NDArray:
    """Reorder a FuMa-layout tensor to ACN channel order."""

    arr = np.asarray(data)
    ch_axis, order = validate_ambisonic_channel_axis(arr, axis=axis, max_order=max_order)
    perm = fuma_to_acn_permutation(order)
    return np.take(arr, perm, axis=ch_axis)


def select_channels(data: ArrayLike, indices: ArrayLike, *, axis: int = -1) -> NDArray:
    """Select ACN channels along an arbitrary channel axis."""

    arr = np.asarray(data)
    ch_axis, order = validate_ambisonic_channel_axis(arr, axis=axis)
    idx = validate_channel_indices(indices, order)
    return np.take(arr, idx, axis=ch_axis)


def zero_channels(data: ArrayLike, indices: ArrayLike, *, axis: int = -1) -> NDArray:
    """Return a copy with selected channels set to zero."""

    arr = np.asarray(data).copy()
    ch_axis, order = validate_ambisonic_channel_axis(arr, axis=axis)
    idx = validate_channel_indices(indices, order)
    moved = np.moveaxis(arr, ch_axis, -1)
    moved[..., idx] = 0
    return np.moveaxis(moved, -1, ch_axis)


def drop_channels(data: ArrayLike, indices: ArrayLike, *, axis: int = -1) -> NDArray:
    """Drop selected channels from an Ambisonic tensor."""

    arr = np.asarray(data)
    ch_axis, order = validate_ambisonic_channel_axis(arr, axis=axis)
    drop = indices_to_mask(indices, order)
    keep = np.nonzero(~drop)[0]
    return np.take(arr, keep, axis=ch_axis)


def channel_energy(coeffs: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """Energy per Ambisonic channel over all non-channel axes."""

    c = np.asarray(coeffs)
    ch_axis, _order = validate_ambisonic_channel_axis(c, axis=axis)
    moved = np.moveaxis(c, ch_axis, -1)
    return np.sum(np.abs(moved) ** 2, axis=tuple(range(moved.ndim - 1))).astype(float)


def channel_rms(coeffs: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """RMS magnitude per Ambisonic channel."""

    c = np.asarray(coeffs)
    ch_axis, _order = validate_ambisonic_channel_axis(c, axis=axis)
    moved = np.moveaxis(c, ch_axis, -1)
    return np.sqrt(np.mean(np.abs(moved) ** 2, axis=tuple(range(moved.ndim - 1)))).astype(float)


def channel_peak(coeffs: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """Peak magnitude per Ambisonic channel."""

    c = np.asarray(coeffs)
    ch_axis, _order = validate_ambisonic_channel_axis(c, axis=axis)
    moved = np.moveaxis(c, ch_axis, -1)
    return np.max(np.abs(moved), axis=tuple(range(moved.ndim - 1))).astype(float)


def active_channel_mask(coeffs: ArrayLike, *, axis: int = -1, threshold: float = 1e-12) -> NDArray[np.bool_]:
    """Boolean mask of channels whose energy exceeds a threshold."""

    return channel_energy(coeffs, axis=axis) > float(threshold)


def active_channel_indices(coeffs: ArrayLike, *, axis: int = -1, threshold: float = 1e-12) -> NDArray[np.int64]:
    """ACN indices of active channels."""

    return mask_to_indices(active_channel_mask(coeffs, axis=axis, threshold=threshold))


def per_order_energy(coeffs: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """Energy per Ambisonic degree."""

    c = np.asarray(coeffs)
    _, order = validate_ambisonic_channel_axis(c, axis=axis)
    energy = channel_energy(c, axis=axis)
    return np.asarray([float(np.sum(energy[s])) for s in order_channel_slices(order)], dtype=float)


def per_order_peak(coeffs: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """Peak magnitude per Ambisonic degree."""

    c = np.asarray(coeffs)
    _, order = validate_ambisonic_channel_axis(c, axis=axis)
    peak = channel_peak(c, axis=axis)
    return np.asarray([float(np.max(peak[s])) for s in order_channel_slices(order)], dtype=float)


def per_order_energy_fraction(coeffs: ArrayLike, *, axis: int = -1, eps: float = 1e-30) -> NDArray[np.float64]:
    """Fraction of total energy in each Ambisonic degree."""

    energy = per_order_energy(coeffs, axis=axis)
    return energy / max(float(np.sum(energy)), float(eps))


def order_weight_vector(max_order: int, values: ArrayLike) -> NDArray[np.float64]:
    """Expand one scalar per Ambisonic degree to one value per channel."""

    weights = np.asarray(values, dtype=float).reshape(-1)
    if weights.size != int(max_order) + 1:
        raise ValueError("values must contain max_order + 1 entries")
    return np.repeat(weights, per_order_channel_counts(int(max_order))).astype(float)


def channel_metadata_dicts(max_order: int) -> tuple[dict[str, object], ...]:
    """JSON-friendly channel table dictionaries."""

    return tuple(row.__dict__.copy() for row in channel_table(max_order))


def mixed_order_mask(
    horizontal_order: int,
    vertical_order: int,
) -> NDArray[np.bool_]:
    """Mask for mixed-order Ambisonics with reduced vertical order.

    Channels are kept when ``degree <= vertical_order`` or when the
    channel is horizontal enough to satisfy ``|m| <= horizontal_order``
    while its vertical complexity ``degree - |m|`` stays within
    ``vertical_order``.
    """

    h = int(horizontal_order)
    v = int(vertical_order)
    if h < 0 or v < 0 or v > h:
        raise ValueError("must satisfy 0 <= vertical_order <= horizontal_order")
    n, m = channel_degree_order(h)
    return (n - np.abs(m)) <= v


def mixed_order_channel_indices(horizontal_order: int, vertical_order: int) -> NDArray[np.int64]:
    """ACN indices retained by :func:`mixed_order_mask`."""

    return np.nonzero(mixed_order_mask(horizontal_order, vertical_order))[0].astype(np.int64)


def mixed_order_channel_count(horizontal_order: int, vertical_order: int) -> int:
    """Number of active channels in a mixed-order Ambisonic set."""

    return int(np.count_nonzero(mixed_order_mask(horizontal_order, vertical_order)))


def infer_mixed_order(mask: ArrayLike) -> MixedOrderSpec:
    """Infer a simple mixed-order spec from an ACN mask."""

    active = np.asarray(mask, dtype=bool).reshape(-1)
    full_order = infer_order(active.size)
    n, m = channel_degree_order(full_order)
    if not np.any(active):
        raise ValueError("mask must contain at least one active channel")
    h = int(np.max(np.abs(m[active])))
    v = int(np.max(n[active] - np.abs(m[active])))
    try:
        expected = mixed_order_mask(h, v)
    except ValueError as exc:
        raise ValueError(f"mask does not match the simple mixed-order model: {exc}") from exc
    if expected.shape != active.shape or not np.array_equal(expected, active):
        raise ValueError("mask does not match the simple mixed-order model")
    return MixedOrderSpec(horizontal_order=h, vertical_order=v)


def compact_mixed_order_coeffs(
    coeffs: ArrayLike,
    mask: ArrayLike,
    *,
    axis: int = -1,
) -> NDArray:
    """Drop inactive ACN channels from a full coefficient array."""

    c = np.asarray(coeffs)
    active = np.asarray(mask, dtype=bool).reshape(-1)
    moved = np.moveaxis(c, axis, -1)
    if moved.shape[-1] != active.size:
        raise ValueError("mask length must match the selected coefficient axis")
    out = moved[..., active]
    return np.moveaxis(out, -1, axis)


def expand_mixed_order_coeffs(
    compact_coeffs: ArrayLike,
    mask: ArrayLike,
    *,
    axis: int = -1,
    fill_value: float = 0.0,
) -> NDArray:
    """Expand compact mixed-order coefficients back to a full ACN array."""

    c = np.asarray(compact_coeffs)
    active = np.asarray(mask, dtype=bool).reshape(-1)
    moved = np.moveaxis(c, axis, -1)
    if moved.shape[-1] != int(np.count_nonzero(active)):
        raise ValueError("compact coefficient axis length must equal mask active count")
    out_shape = moved.shape[:-1] + (active.size,)
    out = np.full(out_shape, fill_value, dtype=moved.dtype)
    out[..., active] = moved
    return np.moveaxis(out, -1, axis)


def split_coeffs_by_order(coeffs: ArrayLike, *, max_order: int | None = None, axis: int = -1) -> tuple[NDArray, ...]:
    """Split an Ambisonic coefficient array into per-degree blocks."""

    c = np.asarray(coeffs)
    ch_axis, order = validate_ambisonic_channel_axis(c, axis=axis, max_order=max_order)
    moved = np.moveaxis(c, ch_axis, -1)
    blocks = tuple(moved[..., s] for s in order_channel_slices(order))
    return tuple(np.moveaxis(block, -1, ch_axis) if block.ndim > ch_axis else block for block in blocks)


def join_coeffs_by_order(blocks: tuple[ArrayLike, ...], *, axis: int = -1) -> NDArray:
    """Join per-degree coefficient blocks along an Ambisonic channel axis."""

    arrays = [np.asarray(block) for block in blocks]
    if not arrays:
        raise ValueError("blocks must be non-empty")
    moved = [np.moveaxis(a, axis, -1) for a in arrays]
    for n, a in enumerate(moved):
        if a.shape[-1] != 2 * n + 1:
            raise ValueError(f"block {n} must have {2 * n + 1} channels")
        if a.shape[:-1] != moved[0].shape[:-1]:
            raise ValueError("all blocks must have matching non-channel shape")
    joined = np.concatenate(moved, axis=-1)
    return np.moveaxis(joined, -1, axis)


def per_order_rms(coeffs: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """RMS coefficient magnitude per Ambisonic degree."""

    c = np.asarray(coeffs)
    _, order = validate_ambisonic_channel_axis(c, axis=axis)
    moved = np.moveaxis(c, axis, -1)
    values = []
    for s in order_channel_slices(order):
        values.append(float(np.sqrt(np.mean(np.abs(moved[..., s]) ** 2))))
    return np.asarray(values, dtype=float)


def active_channel_report(
    coeffs: ArrayLike,
    *,
    axis: int = -1,
    threshold: float = 1e-12,
) -> dict[str, object]:
    """Compact report of active Ambisonic channels and degrees."""

    c = np.asarray(coeffs)
    ch_axis, order = validate_ambisonic_channel_axis(c, axis=axis)
    moved = np.moveaxis(c, ch_axis, -1)
    channel_energy = np.sum(np.abs(moved) ** 2, axis=tuple(range(moved.ndim - 1)))
    active = channel_energy > float(threshold)
    active_degrees = sorted({int(n) for n in channel_degrees(order)[active]})
    return {
        "max_order": int(order),
        "active_channel_count": int(np.count_nonzero(active)),
        "active_indices": np.nonzero(active)[0].astype(int).tolist(),
        "active_degrees": active_degrees,
        "channel_energy": channel_energy.astype(float),
    }


__all__ = [
    "AmbisonicChannel",
    "ChannelLabelStyle",
    "MixedOrderShape",
    "MixedOrderSpec",
    "acn_to_fuma_permutation",
    "acn_sequence",
    "active_channel_indices",
    "active_channel_mask",
    "active_channel_report",
    "channel_degree_order",
    "channel_degrees",
    "channel_energy",
    "channel_index",
    "channel_label_to_acn",
    "channel_labels",
    "channel_metadata_dicts",
    "channel_name",
    "channel_orders",
    "channel_peak",
    "channel_rms",
    "channel_table",
    "combine_channel_masks",
    "compact_mixed_order_coeffs",
    "degree_channel_indices",
    "degree_mask",
    "drop_channels",
    "expand_mixed_order_coeffs",
    "first_acn_for_degree",
    "first_order_channel_mask",
    "fuma_to_acn_permutation",
    "fuma_channel_labels",
    "horizontal_channel_indices",
    "horizontal_channel_mask",
    "infer_mixed_order",
    "indices_to_mask",
    "invert_channel_mask",
    "join_coeffs_by_order",
    "last_acn_for_degree",
    "mask_to_indices",
    "mixed_order_channel_count",
    "mixed_order_channel_indices",
    "mixed_order_mask",
    "non_omni_channel_mask",
    "omni_channel_mask",
    "order_block_slice",
    "order_block_start",
    "order_block_stop",
    "order_weight_vector",
    "per_order_channel_counts",
    "per_order_energy",
    "per_order_energy_fraction",
    "per_order_peak",
    "per_order_rms",
    "reorder_acn_to_fuma",
    "reorder_fuma_to_acn",
    "sectoral_channel_indices",
    "sectoral_channel_mask",
    "select_channels",
    "signed_order_channel_indices",
    "signed_order_mask",
    "split_coeffs_by_order",
    "truncate_channel_mask",
    "validate_ambisonic_channel_axis",
    "validate_channel_indices",
    "validate_channel_mask",
    "zero_channels",
    "zonal_channel_indices",
    "zonal_channel_mask",
]
