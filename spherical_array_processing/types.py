from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


BasisKind = Literal["complex", "real"]
NormalizationKind = Literal["orthonormal", "n3d", "sn3d"]
AngleConvention = Literal["az_el", "az_colat"]


def _to_1d_float(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


@dataclass
class SHBasisSpec:
    """Specification for a spherical harmonics basis.

    .. note::
       The default ``angle_convention`` is ``"az_colat"`` (azimuth +
       colatitude), which differs from :class:`SphericalGrid`'s default
       ``"az_el"`` (azimuth + elevation).  The SH basis functions
       internally convert grids to ``az_colat`` via
       ``_grid_to_az_colat``, so you do **not** need to match these
       manually — but be aware of the difference when inspecting raw
       angle arrays.
    """

    max_order: int
    basis: BasisKind = "complex"
    normalization: NormalizationKind = "orthonormal"
    angle_convention: AngleConvention = "az_colat"
    channel_order: Literal["acn"] = "acn"

    def __post_init__(self) -> None:
        if not isinstance(self.max_order, (int, np.integer)) or self.max_order < 0:
            raise ValueError(f"max_order must be a non-negative integer, got {self.max_order!r}")

    @property
    def n_coeffs(self) -> int:
        return (self.max_order + 1) ** 2


@dataclass
class SphericalGrid:
    """A set of directions on the unit sphere with optional quadrature weights.

    .. note::
       The default ``convention`` is ``"az_el"`` (azimuth + elevation),
       which differs from :class:`SHBasisSpec`'s default
       ``angle_convention`` of ``"az_colat"`` (azimuth + colatitude).
       The SH basis functions handle this conversion automatically, so
       you normally do **not** need to convert between conventions
       yourself.
    """

    azimuth: NDArray[np.float64]
    angle2: NDArray[np.float64]
    weights: NDArray[np.float64] | None = None
    convention: AngleConvention = "az_el"
    _xyz_cache: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.azimuth = _to_1d_float(self.azimuth)
        self.angle2 = _to_1d_float(self.angle2)
        if self.azimuth.shape != self.angle2.shape:
            raise ValueError("azimuth and angle2 must have the same shape")
        if self.weights is not None:
            self.weights = _to_1d_float(self.weights)
            if self.weights.shape != self.azimuth.shape:
                raise ValueError("weights shape must match grid size")

    @property
    def size(self) -> int:
        return self.azimuth.size

    @property
    def elevation(self) -> NDArray[np.float64]:
        if self.convention == "az_el":
            return self.angle2
        return (np.pi / 2.0) - self.angle2

    @property
    def colatitude(self) -> NDArray[np.float64]:
        if self.convention == "az_colat":
            return self.angle2
        return (np.pi / 2.0) - self.angle2


@dataclass
class ArrayGeometry:
    """Spherical array geometry plus capsule / baffle convention.

    Attributes
    ----------
    radius_m : float
        Array radius in metres.
    sensor_grid : SphericalGrid
        Sensor directions on the sphere.
    array_type : {"open", "rigid", "cardioid", "directional"}
        Baffle / capsule family forwarded to
        :func:`spherical_array_processing.acoustics.plane_wave_radial_bn`.
        ``"directional"`` (added in 0.4.0b15) selects a first-order
        directional capsule ``α + (1-α)·cosθ``; pair it with a
        ``metadata["dir_coeff"]`` entry in ``[0, 1]`` to record the
        capsule coefficient.
    sensor_kind : {"pressure", "directional"} or None
        Capsule family at the array shell — ``"pressure"`` for omni
        capsules (``array_type`` ``"open"`` / ``"rigid"``) or
        ``"directional"`` for first-order capsules (``array_type``
        ``"cardioid"`` / ``"directional"``).  When left as ``None``
        (the default), the value is **auto-derived** from
        ``array_type`` in ``__post_init__``, so the field cannot
        silently drift out of sync with the baffle spec.  An
        explicit value is allowed but must match the value implied
        by ``array_type`` — a mismatch (e.g. ``array_type="rigid"``
        with ``sensor_kind="directional"``) raises ``ValueError``
        immediately, which turns this field from a free-form tag
        into an actual consistency invariant enforced by the
        dataclass.
    metadata : dict
        Free-form metadata.  As of 0.4.0b15, when
        ``array_type == "directional"`` the
        :func:`spherical_array_processing.array.simulate_sh_array_response`
        entry point reads ``metadata["dir_coeff"]`` as the fallback
        for its ``dir_coeff`` kwarg, so an ``ArrayGeometry`` that
        already records its directional coefficient does not need the
        caller to re-state it.  Explicit function kwargs still take
        precedence over the metadata value.  Other modules (e.g. the
        ``encoding.radial_equalizer_*`` family) do not currently
        consume this entry and still require ``dir_coeff`` to be
        passed explicitly.
    """

    radius_m: float
    sensor_grid: SphericalGrid
    array_type: Literal["open", "rigid", "cardioid", "directional"] = "rigid"
    sensor_kind: Literal["pressure", "directional"] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        implied = (
            "pressure" if self.array_type in ("open", "rigid")
            else "directional"
        )
        if self.sensor_kind is None:
            # Default path: derive the capsule family from the baffle
            # spec, so the two fields cannot drift apart.
            self.sensor_kind = implied
        elif self.sensor_kind != implied:
            raise ValueError(
                f"ArrayGeometry: sensor_kind={self.sensor_kind!r} is "
                f"inconsistent with array_type={self.array_type!r} "
                f"(expected sensor_kind={implied!r}).  Open / rigid "
                f"arrays carry pressure (omni) capsules; cardioid / "
                f"directional arrays carry first-order directional "
                f"capsules."
            )

    @property
    def n_sensors(self) -> int:
        return self.sensor_grid.size


@dataclass
class SHSignalFrame:
    data: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64]
    basis: SHBasisSpec


@dataclass
class SHCovariance:
    data: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64] | None
    basis: SHBasisSpec


@dataclass
class SpatialSpectrumResult:
    spectrum: NDArray[np.float64]
    grid: SphericalGrid
    peak_indices: NDArray[np.int64]
    peak_dirs_rad: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureReproConfig:
    dpi: int = 150
    figsize: tuple[float, float] = (8.0, 6.0)
    font_family: str = "DejaVu Sans"
    font_size: float = 12.0
    line_width: float = 1.8
    colormap: str = "viridis"
    image_ssim_threshold: float = 0.95
    max_rel_curve_error: float = 1e-2
