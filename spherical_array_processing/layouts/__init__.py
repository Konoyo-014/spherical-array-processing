"""Loudspeaker layout containers, presets, and geometry diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..coords import cart_to_sph, unit_sph_to_cart
from ..types import SphericalGrid


LayoutConvention = Literal["az_el", "az_colat"]


@dataclass
class LoudspeakerLayout:
    """Named loudspeaker layout with spherical directions and metadata."""

    directions_rad: NDArray[np.float64]
    labels: tuple[str, ...] | None = None
    convention: LayoutConvention = "az_el"
    weights: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dirs = np.asarray(self.directions_rad, dtype=float)
        if dirs.ndim != 2 or dirs.shape[1] != 2:
            raise ValueError("directions_rad must have shape (n_speakers, 2)")
        if self.convention not in ("az_el", "az_colat"):
            raise ValueError("convention must be 'az_el' or 'az_colat'")
        if self.labels is not None and len(self.labels) != dirs.shape[0]:
            raise ValueError("labels must match number of speakers")
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
            if self.weights.shape != (dirs.shape[0],):
                raise ValueError("weights must match number of speakers")
        self.directions_rad = dirs

    @property
    def n_speakers(self) -> int:
        return int(self.directions_rad.shape[0])

    @property
    def azimuth(self) -> NDArray[np.float64]:
        return self.directions_rad[:, 0]

    @property
    def angle2(self) -> NDArray[np.float64]:
        return self.directions_rad[:, 1]

    def as_grid(self) -> SphericalGrid:
        """Return the layout as a :class:`SphericalGrid`."""

        return SphericalGrid(
            azimuth=self.azimuth.copy(),
            angle2=self.angle2.copy(),
            weights=None if self.weights is None else self.weights.copy(),
            convention=self.convention,
        )


def layout_from_degrees(
    directions_deg: ArrayLike,
    *,
    labels: tuple[str, ...] | None = None,
    convention: LayoutConvention = "az_el",
    weights: ArrayLike | None = None,
    name: str | None = None,
) -> LoudspeakerLayout:
    """Create a loudspeaker layout from degree-valued directions."""

    metadata = {} if name is None else {"name": name}
    return LoudspeakerLayout(
        directions_rad=np.deg2rad(np.asarray(directions_deg, dtype=float)),
        labels=labels,
        convention=convention,
        weights=None if weights is None else np.asarray(weights, dtype=float),
        metadata=metadata,
    )


def layout_from_grid(
    grid: SphericalGrid,
    *,
    labels: tuple[str, ...] | None = None,
    name: str | None = None,
) -> LoudspeakerLayout:
    """Create a layout from a spherical grid."""

    metadata = {} if name is None else {"name": name}
    return LoudspeakerLayout(
        directions_rad=np.column_stack([grid.azimuth, grid.angle2]),
        labels=labels,
        convention=grid.convention,
        weights=None if grid.weights is None else np.asarray(grid.weights, dtype=float),
        metadata=metadata,
    )


def layout_to_degrees(layout: LoudspeakerLayout) -> NDArray[np.float64]:
    """Return layout directions in degrees."""

    return np.rad2deg(layout.directions_rad)


def layout_cartesian(layout: LoudspeakerLayout, *, radius: float = 1.0) -> NDArray[np.float64]:
    """Cartesian speaker coordinates on a sphere."""

    return unit_sph_to_cart(
        layout.azimuth,
        layout.angle2,
        convention=layout.convention,
    ) * float(radius)


def layout_pairwise_angles(layout: LoudspeakerLayout) -> NDArray[np.float64]:
    """Pairwise loudspeaker angular distances in radians."""

    xyz = layout_cartesian(layout)
    return np.arccos(np.clip(xyz @ xyz.T, -1.0, 1.0))


def layout_nearest_neighbor_angles(layout: LoudspeakerLayout) -> NDArray[np.float64]:
    """Nearest-speaker angle for each speaker."""

    if layout.n_speakers < 2:
        raise ValueError("layout must contain at least two speakers")
    d = layout_pairwise_angles(layout)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def layout_min_separation(layout: LoudspeakerLayout) -> float:
    """Smallest loudspeaker angular separation."""

    return float(np.min(layout_nearest_neighbor_angles(layout)))


def layout_centroid_vector(layout: LoudspeakerLayout, *, weights: ArrayLike | None = None) -> NDArray[np.float64]:
    """Weighted Cartesian centroid vector."""

    xyz = layout_cartesian(layout)
    if weights is None:
        w = np.ones(layout.n_speakers)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != (layout.n_speakers,):
            raise ValueError("weights must match layout")
    if float(np.sum(w)) == 0.0:
        raise ValueError("weights must have non-zero sum")
    return np.sum(w[:, None] * xyz, axis=0) / float(np.sum(w))


def layout_is_hemispherical(layout: LoudspeakerLayout, *, axis: ArrayLike = (0.0, 0.0, 1.0)) -> bool:
    """Whether all speakers lie in one closed hemisphere."""

    xyz = layout_cartesian(layout)
    a = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(a))
    if norm <= 0.0:
        raise ValueError("axis must be non-zero")
    return bool(np.all(xyz @ (a / norm) >= -1e-12))


def layout_has_upper_hemisphere(layout: LoudspeakerLayout) -> bool:
    """Whether the layout contains any positive-elevation speaker."""

    grid = layout.as_grid()
    return bool(np.any(grid.elevation > 0.0))


def layout_has_lower_hemisphere(layout: LoudspeakerLayout) -> bool:
    """Whether the layout contains any negative-elevation speaker."""

    grid = layout.as_grid()
    return bool(np.any(grid.elevation < 0.0))


def mirror_layout_z(layout: LoudspeakerLayout) -> LoudspeakerLayout:
    """Mirror a layout across the horizontal plane."""

    xyz = layout_cartesian(layout)
    xyz[:, 2] *= -1.0
    az, angle2, _ = cart_to_sph(xyz[:, 0], xyz[:, 1], xyz[:, 2], convention=layout.convention)
    return LoudspeakerLayout(
        directions_rad=np.column_stack([az, angle2]),
        labels=layout.labels,
        convention=layout.convention,
        weights=None if layout.weights is None else layout.weights.copy(),
        metadata=dict(layout.metadata),
    )


def rotate_layout_z(layout: LoudspeakerLayout, yaw_rad: float) -> LoudspeakerLayout:
    """Rotate a layout around the vertical axis."""

    dirs = layout.directions_rad.copy()
    dirs[:, 0] = (dirs[:, 0] + float(yaw_rad) + np.pi) % (2.0 * np.pi) - np.pi
    return LoudspeakerLayout(
        directions_rad=dirs,
        labels=layout.labels,
        convention=layout.convention,
        weights=None if layout.weights is None else layout.weights.copy(),
        metadata=dict(layout.metadata),
    )


def subset_layout(layout: LoudspeakerLayout, indices: ArrayLike) -> LoudspeakerLayout:
    """Return a subset of speakers by index."""

    idx = np.asarray(indices, dtype=int).reshape(-1)
    labels = None if layout.labels is None else tuple(layout.labels[i] for i in idx)
    weights = None if layout.weights is None else layout.weights[idx]
    return LoudspeakerLayout(
        directions_rad=layout.directions_rad[idx],
        labels=labels,
        convention=layout.convention,
        weights=weights,
        metadata=dict(layout.metadata),
    )


def horizontal_layout(n_speakers: int, *, start_deg: float = 0.0) -> LoudspeakerLayout:
    """Evenly spaced horizontal ring layout."""

    n = int(n_speakers)
    if n < 2:
        raise ValueError("n_speakers must be >= 2")
    az = float(start_deg) + np.arange(n) * 360.0 / n
    return layout_from_degrees(
        np.column_stack([az, np.zeros(n)]),
        labels=tuple(f"S{i + 1}" for i in range(n)),
        name=f"horizontal-{n}",
    )


def stereo_layout() -> LoudspeakerLayout:
    """Stereo layout at +/-30 degrees."""

    return layout_from_degrees(
        [[30.0, 0.0], [-30.0, 0.0]],
        labels=("L", "R"),
        name="stereo",
    )


def itu_5_1_layout(*, include_lfe: bool = False) -> LoudspeakerLayout:
    """ITU-style 5.0/5.1 bed layout."""

    dirs = [[30.0, 0.0], [-30.0, 0.0], [0.0, 0.0], [110.0, 0.0], [-110.0, 0.0]]
    labels = ["L", "R", "C", "Ls", "Rs"]
    if include_lfe:
        dirs.append([0.0, -90.0])
        labels.append("LFE")
    return layout_from_degrees(dirs, labels=tuple(labels), name="itu-5.1" if include_lfe else "itu-5.0")


def itu_7_1_layout(*, include_lfe: bool = False) -> LoudspeakerLayout:
    """ITU-style 7.0/7.1 bed layout."""

    dirs = [
        [30.0, 0.0], [-30.0, 0.0], [0.0, 0.0],
        [90.0, 0.0], [-90.0, 0.0], [150.0, 0.0], [-150.0, 0.0],
    ]
    labels = ["L", "R", "C", "Lss", "Rss", "Lrs", "Rrs"]
    if include_lfe:
        dirs.append([0.0, -90.0])
        labels.append("LFE")
    return layout_from_degrees(dirs, labels=tuple(labels), name="itu-7.1" if include_lfe else "itu-7.0")


def itu_5_1_4_layout() -> LoudspeakerLayout:
    """Common 5.1.4 immersive layout without LFE in the decode grid."""

    base = list(layout_to_degrees(itu_5_1_layout()).tolist())
    top = [[45.0, 45.0], [-45.0, 45.0], [135.0, 45.0], [-135.0, 45.0]]
    labels = ("L", "R", "C", "Ls", "Rs", "Ltf", "Rtf", "Ltr", "Rtr")
    return layout_from_degrees(base + top, labels=labels, name="itu-5.1.4")


def itu_7_1_4_layout() -> LoudspeakerLayout:
    """Common 7.1.4 immersive layout without LFE in the decode grid."""

    base = list(layout_to_degrees(itu_7_1_layout()).tolist())
    top = [[45.0, 45.0], [-45.0, 45.0], [135.0, 45.0], [-135.0, 45.0]]
    labels = ("L", "R", "C", "Lss", "Rss", "Lrs", "Rrs", "Ltf", "Rtf", "Ltr", "Rtr")
    return layout_from_degrees(base + top, labels=labels, name="itu-7.1.4")


def cube_layout() -> LoudspeakerLayout:
    """Eight-speaker cube layout at the vertices of a cube."""

    xyz = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    az, el, _ = cart_to_sph(xyz[:, 0], xyz[:, 1], xyz[:, 2], convention="az_el")
    return LoudspeakerLayout(np.column_stack([az, el]), labels=tuple(f"C{i + 1}" for i in range(8)), metadata={"name": "cube"})


def octahedral_layout() -> LoudspeakerLayout:
    """Six-speaker octahedral layout."""

    dirs = [[0.0, 0.0], [90.0, 0.0], [180.0, 0.0], [-90.0, 0.0], [0.0, 90.0], [0.0, -90.0]]
    return layout_from_degrees(dirs, labels=("X+", "Y+", "X-", "Y-", "Z+", "Z-"), name="octahedral")


def tetrahedral_layout() -> LoudspeakerLayout:
    """Four-speaker tetrahedral layout."""

    xyz = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    az, el, _ = cart_to_sph(xyz[:, 0], xyz[:, 1], xyz[:, 2], convention="az_el")
    return LoudspeakerLayout(np.column_stack([az, el]), labels=("T1", "T2", "T3", "T4"), metadata={"name": "tetrahedral"})


def layout_registry() -> dict[str, LoudspeakerLayout]:
    """Return a small registry of common layout presets."""

    return {
        "stereo": stereo_layout(),
        "itu_5_0": itu_5_1_layout(),
        "itu_7_0": itu_7_1_layout(),
        "itu_5_1_4": itu_5_1_4_layout(),
        "itu_7_1_4": itu_7_1_4_layout(),
        "tetrahedral": tetrahedral_layout(),
        "octahedral": octahedral_layout(),
        "cube": cube_layout(),
    }


def get_layout(name: str) -> LoudspeakerLayout:
    """Fetch a common layout preset by name."""

    registry = layout_registry()
    key = str(name).lower().replace("-", "_").replace(".", "_")
    if key not in registry:
        raise ValueError(f"unknown layout preset {name!r}")
    return registry[key]


__all__ = [
    "LayoutConvention",
    "LoudspeakerLayout",
    "cube_layout",
    "get_layout",
    "horizontal_layout",
    "itu_5_1_4_layout",
    "itu_5_1_layout",
    "itu_7_1_4_layout",
    "itu_7_1_layout",
    "layout_cartesian",
    "layout_centroid_vector",
    "layout_from_degrees",
    "layout_from_grid",
    "layout_has_lower_hemisphere",
    "layout_has_upper_hemisphere",
    "layout_is_hemispherical",
    "layout_min_separation",
    "layout_nearest_neighbor_angles",
    "layout_pairwise_angles",
    "layout_registry",
    "layout_to_degrees",
    "mirror_layout_z",
    "octahedral_layout",
    "rotate_layout_z",
    "stereo_layout",
    "subset_layout",
    "tetrahedral_layout",
]
