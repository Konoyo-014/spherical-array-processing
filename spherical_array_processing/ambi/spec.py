"""Typed Ambisonics signal containers and convention helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..types import AngleConvention, BasisKind, NormalizationKind, SHBasisSpec
from .format import convert_ambi_normalization


AmbisonicChannelOrder = Literal["acn"]
AmbisonicDomain = Literal["time", "frequency", "stft"]


def channel_count(max_order: int) -> int:
    """Return the full 3-D Ambisonic channel count ``(N + 1)^2``."""
    order = int(max_order)
    if order < 0:
        raise ValueError("max_order must be non-negative")
    return (order + 1) ** 2


def infer_order(n_channels: int) -> int:
    """Infer Ambisonic order from a full 3-D channel count.

    Raises
    ------
    ValueError
        If *n_channels* is not one of ``1, 4, 9, 16, ...``.
    """
    n = int(n_channels)
    if n < 1:
        raise ValueError("n_channels must be positive")
    root = int(round(np.sqrt(n)))
    if root * root != n:
        raise ValueError(
            "n_channels must equal (max_order + 1)^2; "
            f"got {n_channels!r}"
        )
    return root - 1


def order_channel_slices(max_order: int) -> tuple[slice, ...]:
    """Return ACN channel slices for each Ambisonic order.

    Order ``n`` occupies the contiguous ACN block
    ``[n², (n + 1)²)``.  The returned tuple has ``max_order + 1``
    entries, so ``order_channel_slices(2)[1]`` selects the first-order
    ``X/Y/Z`` block.
    """
    order = int(max_order)
    if order < 0:
        raise ValueError("max_order must be non-negative")
    return tuple(slice(n * n, (n + 1) * (n + 1)) for n in range(order + 1))


def order_channel_mask(
    max_order: int,
    *,
    min_order: int = 0,
    max_active_order: int | None = None,
) -> NDArray[np.bool_]:
    """Boolean ACN mask selecting a contiguous Ambisonic order range."""
    order = int(max_order)
    lo = int(min_order)
    hi = order if max_active_order is None else int(max_active_order)
    if order < 0:
        raise ValueError("max_order must be non-negative")
    if lo < 0 or hi < lo or hi > order:
        raise ValueError(
            "order range must satisfy 0 <= min_order <= max_active_order <= max_order"
        )
    mask = np.zeros(channel_count(order), dtype=bool)
    mask[lo * lo : (hi + 1) * (hi + 1)] = True
    return mask


@dataclass(frozen=True)
class AmbisonicSignalReport:
    """Numerical health report for an Ambisonic signal tensor."""

    max_order: int
    n_channels: int
    channel_axis: int
    active_channels: int
    channel_rms: NDArray[np.float64]
    per_order_energy: NDArray[np.float64]
    per_order_energy_fraction: NDArray[np.float64]
    peak_abs: float
    rms: float
    crest_factor_db: float
    has_nan: bool
    has_inf: bool


@dataclass(frozen=True)
class AmbisonicSpec:
    """Convention metadata for a real or complex Ambisonic stream.

    The stable interchange default is **ACN/SN3D real AmbiX**.  The
    package's mathematical internals often use ``"orthonormal"``, but a
    public signal container should preserve the user's file / stream
    convention explicitly so normalization changes are never implicit.
    """

    max_order: int
    basis: BasisKind = "real"
    normalization: NormalizationKind = "sn3d"
    channel_order: AmbisonicChannelOrder = "acn"
    angle_convention: AngleConvention = "az_colat"
    domain: AmbisonicDomain = "time"
    mixed_order_mask: NDArray[np.bool_] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.max_order, (int, np.integer)) or self.max_order < 0:
            raise ValueError("max_order must be a non-negative integer")
        if self.basis not in ("real", "complex"):
            raise ValueError("basis must be 'real' or 'complex'")
        if self.normalization not in ("orthonormal", "n3d", "sn3d"):
            raise ValueError(
                "normalization must be 'orthonormal', 'n3d', or 'sn3d'"
            )
        if self.channel_order != "acn":
            raise ValueError("only ACN channel_order is supported")
        if self.angle_convention not in ("az_el", "az_colat"):
            raise ValueError("angle_convention must be 'az_el' or 'az_colat'")
        if self.domain not in ("time", "frequency", "stft"):
            raise ValueError("domain must be 'time', 'frequency', or 'stft'")
        if self.mixed_order_mask is not None:
            mask = np.asarray(self.mixed_order_mask, dtype=bool).reshape(-1)
            if mask.size != self.n_channels:
                raise ValueError(
                    "mixed_order_mask length must match n_channels"
                )
            object.__setattr__(self, "mixed_order_mask", mask)

    @property
    def n_channels(self) -> int:
        return channel_count(self.max_order)

    def basis_spec(
        self,
        *,
        normalization: NormalizationKind | None = None,
    ) -> SHBasisSpec:
        """Return the matching :class:`~spherical_array_processing.SHBasisSpec`."""
        return SHBasisSpec(
            max_order=int(self.max_order),
            basis=self.basis,
            normalization=self.normalization if normalization is None else normalization,
            angle_convention=self.angle_convention,
            channel_order=self.channel_order,
        )

    def validate_axis(self, data: ArrayLike, axis: int = -1) -> int:
        """Validate and return the normalized channel axis for *data*."""
        arr = np.asarray(data)
        if arr.ndim == 0:
            raise ValueError("Ambisonic data must have at least one dimension")
        ch_axis = int(axis) % arr.ndim
        if arr.shape[ch_axis] != self.n_channels:
            raise ValueError(
                f"channel axis has length {arr.shape[ch_axis]}, expected "
                f"{self.n_channels} for max_order={self.max_order}"
            )
        return ch_axis

    def with_normalization(self, normalization: NormalizationKind) -> "AmbisonicSpec":
        """Return a copy with a different normalization label."""
        return AmbisonicSpec(
            max_order=self.max_order,
            basis=self.basis,
            normalization=normalization,
            channel_order=self.channel_order,
            angle_convention=self.angle_convention,
            domain=self.domain,
            mixed_order_mask=self.mixed_order_mask,
            metadata=dict(self.metadata),
        )

    def with_domain(self, domain: AmbisonicDomain) -> "AmbisonicSpec":
        """Return a copy with a different signal-domain label."""
        return AmbisonicSpec(
            max_order=self.max_order,
            basis=self.basis,
            normalization=self.normalization,
            channel_order=self.channel_order,
            angle_convention=self.angle_convention,
            domain=domain,
            mixed_order_mask=self.mixed_order_mask,
            metadata=dict(self.metadata),
        )


@dataclass
class AmbisonicFrame:
    """Ambisonic data plus convention, rate, and frequency metadata."""

    data: NDArray[Any]
    spec: AmbisonicSpec
    channel_axis: int = -1
    sample_rate_hz: float | None = None
    freqs_hz: NDArray[np.float64] | None = None
    freq_axis: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        self.channel_axis = self.spec.validate_axis(self.data, self.channel_axis)
        if self.sample_rate_hz is not None and float(self.sample_rate_hz) <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")
        if self.freqs_hz is not None:
            freqs = np.asarray(self.freqs_hz, dtype=float).reshape(-1)
            f_axis = 0 if self.freq_axis is None else int(self.freq_axis) % self.data.ndim
            if f_axis == self.channel_axis:
                raise ValueError("freq_axis must be different from channel_axis")
            if self.spec.domain in ("frequency", "stft") and self.data.shape[f_axis] != freqs.size:
                raise ValueError(
                    "freqs_hz length must match the frequency axis "
                    "for frequency/STFT AmbisonicFrame data"
                )
            self.freqs_hz = freqs
            self.freq_axis = f_axis
        elif self.freq_axis is not None:
            self.freq_axis = int(self.freq_axis) % self.data.ndim

    def as_normalization(
        self,
        normalization: NormalizationKind,
    ) -> "AmbisonicFrame":
        """Return a frame with coefficients rescaled to *normalization*."""
        if normalization == self.spec.normalization:
            return AmbisonicFrame(
                self.data.copy(),
                self.spec,
                channel_axis=self.channel_axis,
                sample_rate_hz=self.sample_rate_hz,
                freqs_hz=None if self.freqs_hz is None else self.freqs_hz.copy(),
                freq_axis=self.freq_axis,
                metadata=dict(self.metadata),
            )
        converted = convert_ambi_normalization(
            self.data,
            max_order=self.spec.max_order,
            from_=self.spec.normalization,
            to=normalization,
            axis=self.channel_axis,
        )
        return AmbisonicFrame(
            converted,
            self.spec.with_normalization(normalization),
            channel_axis=self.channel_axis,
            sample_rate_hz=self.sample_rate_hz,
            freqs_hz=None if self.freqs_hz is None else self.freqs_hz.copy(),
            freq_axis=self.freq_axis,
            metadata=dict(self.metadata),
        )

    def with_mixed_order_mask_applied(self) -> "AmbisonicFrame":
        """Return a copy with inactive mixed-order channels zeroed."""
        if self.spec.mixed_order_mask is None:
            return AmbisonicFrame(
                self.data.copy(),
                self.spec,
                channel_axis=self.channel_axis,
                sample_rate_hz=self.sample_rate_hz,
                freqs_hz=None if self.freqs_hz is None else self.freqs_hz.copy(),
                freq_axis=self.freq_axis,
                metadata=dict(self.metadata),
            )
        data = np.array(self.data, copy=True)
        moved = np.moveaxis(data, self.channel_axis, -1)
        moved[..., ~self.spec.mixed_order_mask] = 0
        data = np.moveaxis(moved, -1, self.channel_axis)
        return AmbisonicFrame(
            data,
            self.spec,
            channel_axis=self.channel_axis,
            sample_rate_hz=self.sample_rate_hz,
            freqs_hz=None if self.freqs_hz is None else self.freqs_hz.copy(),
            freq_axis=self.freq_axis,
            metadata=dict(self.metadata),
        )


def per_order_energy(
    data: ArrayLike,
    *,
    max_order: int | None = None,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Total signal energy grouped by Ambisonic order.

    Energy is summed over every non-channel axis and over all channels
    belonging to each ACN order.  Complex inputs use ``|x|²``.
    """
    arr = np.asarray(data)
    if arr.ndim == 0:
        raise ValueError("data must have at least one dimension")
    ch_axis = int(axis) % arr.ndim
    n_channels = int(arr.shape[ch_axis])
    order = infer_order(n_channels) if max_order is None else int(max_order)
    if channel_count(order) != n_channels:
        raise ValueError(
            f"axis has {n_channels} channels, expected {channel_count(order)} "
            f"for max_order={order}"
        )
    moved = np.moveaxis(arr, ch_axis, -1)
    power = np.abs(moved) ** 2
    return np.array(
        [float(np.sum(power[..., sl])) for sl in order_channel_slices(order)],
        dtype=float,
    )


def ambisonic_signal_report(
    data: ArrayLike,
    *,
    spec: AmbisonicSpec | None = None,
    axis: int = -1,
    active_threshold_db: float = -120.0,
) -> AmbisonicSignalReport:
    """Summarise channel and order-level health of an Ambisonic tensor.

    The report is intentionally convention-neutral: it does not try to
    infer SN3D/N3D/orthonormal scaling from amplitudes.  It checks
    shape consistency, finite values, per-channel RMS, per-order
    energy, active-channel count, peak level, and crest factor.
    """
    arr = np.asarray(data)
    if arr.ndim == 0:
        raise ValueError("data must have at least one dimension")
    ch_axis = int(axis) % arr.ndim
    if spec is None:
        order = infer_order(arr.shape[ch_axis])
    else:
        ch_axis = spec.validate_axis(arr, ch_axis)
        order = spec.max_order
    moved = np.moveaxis(arr, ch_axis, -1)
    finite = np.isfinite(moved)
    has_nan = bool(np.isnan(moved).any())
    has_inf = bool(np.isinf(moved).any())
    safe = np.where(finite, moved, 0)
    channel_rms = np.sqrt(np.mean(np.abs(safe) ** 2, axis=tuple(range(safe.ndim - 1))))
    total_rms = float(np.sqrt(np.mean(np.abs(safe) ** 2)))
    peak_abs = float(np.max(np.abs(safe))) if safe.size else 0.0
    if total_rms > 0.0 and peak_abs > 0.0:
        crest = float(20.0 * np.log10(peak_abs / total_rms))
    else:
        crest = -float("inf")
    energies = per_order_energy(safe, max_order=order, axis=-1)
    total_energy = float(np.sum(energies))
    fractions = (
        energies / total_energy
        if total_energy > 0.0
        else np.zeros_like(energies, dtype=float)
    )
    if channel_rms.size:
        threshold = float(np.max(channel_rms)) * 10.0 ** (active_threshold_db / 20.0)
        active = int(np.count_nonzero(channel_rms > threshold))
    else:
        active = 0
    return AmbisonicSignalReport(
        max_order=order,
        n_channels=channel_count(order),
        channel_axis=ch_axis,
        active_channels=active,
        channel_rms=channel_rms.astype(float, copy=False),
        per_order_energy=energies,
        per_order_energy_fraction=fractions,
        peak_abs=peak_abs,
        rms=total_rms,
        crest_factor_db=crest,
        has_nan=has_nan,
        has_inf=has_inf,
    )


__all__ = [
    "AmbisonicChannelOrder",
    "AmbisonicDomain",
    "AmbisonicFrame",
    "AmbisonicSpec",
    "AmbisonicSignalReport",
    "ambisonic_signal_report",
    "channel_count",
    "infer_order",
    "order_channel_mask",
    "order_channel_slices",
    "per_order_energy",
]
