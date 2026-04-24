"""Pre-defined microphone-array geometries for popular SH arrays.

These helpers return ready-to-use :class:`ArrayGeometry` instances with
canonical radii, so a user can quickly reproduce published results
without re-typing (θ, φ) tables or measuring a custom rig.

All angles are stored in radians and azimuth/elevation follow the
``az_el`` convention (elevation from the x-y plane, ``+π/2`` at the
north pole).  Use :func:`spherical_array_processing.coords.azel_to_az_colat`
if colatitude-based grids are preferred.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..types import ArrayGeometry, SphericalGrid


# --------------------------------------------------------------------------- #
# Eigenmike em32                                                              #
# --------------------------------------------------------------------------- #
# mh acoustics Eigenmike EM32 — nominal microphone positions.
# Values transcribed from the mh acoustics "EigenUnits" documentation, in
# (azimuth_deg, elevation_deg) with the elevation measured from the horizon.
_EM32_POSITIONS_DEG: tuple[tuple[float, float], ...] = (
    (0.0, 21.0),
    (32.0, 0.0),
    (0.0, -21.0),
    (328.0, 0.0),
    (0.0, 58.0),
    (45.0, 35.0),
    (69.0, 0.0),
    (45.0, -35.0),
    (0.0, -58.0),
    (315.0, -35.0),
    (291.0, 0.0),
    (315.0, 35.0),
    (91.0, 69.0),
    (90.0, 32.0),
    (90.0, -31.0),
    (89.0, -69.0),
    (180.0, 21.0),
    (212.0, 0.0),
    (180.0, -21.0),
    (148.0, 0.0),
    (180.0, 58.0),
    (225.0, 35.0),
    (249.0, 0.0),
    (225.0, -35.0),
    (180.0, -58.0),
    (135.0, -35.0),
    (111.0, 0.0),
    (135.0, 35.0),
    (269.0, 69.0),
    (270.0, 32.0),
    (270.0, -32.0),
    (271.0, -69.0),
)


def em32_eigenmike(radius_m: float = 0.042) -> ArrayGeometry:
    """Eigenmike EM32 — 32 microphones on a rigid sphere of radius ``4.2 cm``.

    Parameters
    ----------
    radius_m : float, optional
        Rigid-sphere radius.  Default ``0.042`` m matches the physical
        device; pass a custom value to model a scaled prototype.
    """
    az = np.radians([p[0] for p in _EM32_POSITIONS_DEG])
    el = np.radians([p[1] for p in _EM32_POSITIONS_DEG])
    # Uniform unit weights over the 32 grid points — Eigenmike is not a
    # quadrature grid, so downstream code should use the weighted
    # least-squares SHT (which happens to be the default in
    # :func:`direct_sht` without quadrature weights).
    grid = SphericalGrid(
        azimuth=az,
        angle2=el,
        weights=np.full(32, 4.0 * np.pi / 32),
        convention="az_el",
    )
    return ArrayGeometry(radius_m=float(radius_m), sensor_grid=grid, array_type="rigid")


# --------------------------------------------------------------------------- #
# Tetrahedral A-format array                                                  #
# --------------------------------------------------------------------------- #
def tetrahedral_array(
    radius_m: float = 0.015,
    orientation: Literal["front", "upright"] = "front",
    array_type: Literal["open", "cardioid"] = "cardioid",
) -> ArrayGeometry:
    """4-microphone tetrahedral array used for 1st-order Ambisonics (A-format).

    The four capsules are placed at the vertices of a regular tetrahedron
    centred on the origin.  Two orientations are supported:

    * ``"front"`` — the tetrahedron is rotated so that one vertex points
      toward the ``+y`` axis (front in typical listening coordinates)
      and the remaining three form a down-facing equilateral triangle.
    * ``"upright"`` — one vertex points toward ``+z`` (up), with the
      other three capsules equally spaced around the equator.  Matches
      the SoundField-style "LFD/RFU/LBU/RBD" layout after the usual
      45° rotations.

    Parameters
    ----------
    radius_m : float, optional
        Tetrahedral radius (capsule distance from the acoustic centre).
        Default ``1.5 cm`` matches typical production tetrahedral
        microphones (e.g. SoundField SPS200, TetraMic).
    orientation : {"front", "upright"}, optional
        Orientation of the tetrahedron.
    array_type : {"open", "cardioid"}, optional
        Modelling assumption.  Default ``"cardioid"`` matches physical
        tetrahedral mics.

    Returns
    -------
    ArrayGeometry
    """
    # Canonical tetrahedron vertices (on unit sphere).  Use integer
    # coordinates proportional to (±1, ±1, ±1) with only even sign flips.
    verts = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    if orientation == "front":
        target = np.array([0.0, 1.0, 0.0])
    elif orientation == "upright":
        target = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError(f"orientation must be 'front' or 'upright', got {orientation!r}")

    # Rotate so the first capsule lies on the requested cardinal axis via
    # the Rodrigues rotation formula.
    v0 = verts[0]
    axis = np.cross(v0, target)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        rot = np.eye(3)
    else:
        axis = axis / s
        angle = np.arccos(np.clip(np.dot(v0, target), -1.0, 1.0))
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rot = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    verts = verts @ rot.T
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    azimuth = np.arctan2(y, x) % (2 * np.pi)
    elevation = np.arcsin(np.clip(z, -1.0, 1.0))

    grid = SphericalGrid(
        azimuth=azimuth,
        angle2=elevation,
        weights=np.full(4, np.pi),
        convention="az_el",
    )
    return ArrayGeometry(
        radius_m=float(radius_m), sensor_grid=grid, array_type=array_type
    )


# --------------------------------------------------------------------------- #
# Cubic array                                                                 #
# --------------------------------------------------------------------------- #
def cubic_array(
    radius_m: float = 0.035,
    array_type: Literal["open", "rigid"] = "open",
) -> ArrayGeometry:
    """8-microphone array on the vertices of a cube inscribed in a sphere.

    Useful as a lightweight second-order-adjacent test rig.  By default
    the cube is oriented with faces normal to the coordinate axes.
    """
    verts = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    azimuth = np.arctan2(y, x) % (2 * np.pi)
    elevation = np.arcsin(np.clip(z, -1.0, 1.0))
    grid = SphericalGrid(
        azimuth=azimuth,
        angle2=elevation,
        weights=np.full(8, 4.0 * np.pi / 8),
        convention="az_el",
    )
    return ArrayGeometry(
        radius_m=float(radius_m), sensor_grid=grid, array_type=array_type
    )


# --------------------------------------------------------------------------- #
# Uniform circular array                                                      #
# --------------------------------------------------------------------------- #
def circular_array(
    n_mics: int,
    radius_m: float,
    elevation_rad: float = 0.0,
    array_type: Literal["open", "rigid"] = "open",
) -> ArrayGeometry:
    """Evenly-spaced microphones on a horizontal ring at given elevation."""
    if n_mics < 2:
        raise ValueError("n_mics must be >= 2")
    az = np.linspace(0.0, 2 * np.pi, int(n_mics), endpoint=False)
    el = np.full(int(n_mics), float(elevation_rad))
    grid = SphericalGrid(
        azimuth=az,
        angle2=el,
        weights=np.full(int(n_mics), 4.0 * np.pi / int(n_mics)),
        convention="az_el",
    )
    return ArrayGeometry(
        radius_m=float(radius_m), sensor_grid=grid, array_type=array_type
    )


__all__ = [
    "circular_array",
    "cubic_array",
    "em32_eigenmike",
    "tetrahedral_array",
]
