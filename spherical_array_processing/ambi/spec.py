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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        self.channel_axis = self.spec.validate_axis(self.data, self.channel_axis)
        if self.sample_rate_hz is not None and float(self.sample_rate_hz) <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")
        if self.freqs_hz is not None:
            freqs = np.asarray(self.freqs_hz, dtype=float).reshape(-1)
            if self.spec.domain in ("frequency", "stft") and self.data.shape[0] != freqs.size:
                raise ValueError(
                    "freqs_hz length must match the leading frequency axis "
                    "for frequency/STFT AmbisonicFrame data"
                )
            self.freqs_hz = freqs

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
            metadata=dict(self.metadata),
        )


__all__ = [
    "AmbisonicChannelOrder",
    "AmbisonicDomain",
    "AmbisonicFrame",
    "AmbisonicSpec",
    "channel_count",
    "infer_order",
]
