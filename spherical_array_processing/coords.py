from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "sph_to_cart",
    "cart_to_sph",
    "azel_to_az_colat",
    "az_colat_to_azel",
    "unit_sph_to_cart",
]


def _a(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=float)


def sph_to_cart(
    azimuth: ArrayLike,
    angle2: ArrayLike,
    radius: ArrayLike | float = 1.0,
    convention: str = "az_el",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert spherical coordinates to Cartesian (x, y, z).

    Parameters
    ----------
    azimuth : array_like
        Azimuth angle(s) in radians.
    angle2 : array_like
        Second angle in radians. Interpreted as elevation when
        *convention* is ``"az_el"`` or colatitude when ``"az_colat"``.
    radius : array_like or float, optional
        Radial distance. Default is 1.0.
    convention : str, optional
        ``"az_el"`` (default) treats *angle2* as elevation;
        ``"az_colat"`` treats it as colatitude (inclination from +z).

    Returns
    -------
    x, y, z : tuple of ndarray
        Cartesian coordinates with the same broadcast shape as the inputs.

    Examples
    --------
    >>> import numpy as np
    >>> x, y, z = sph_to_cart(0.0, 0.0, 1.0)
    >>> np.allclose([x, y, z], [1.0, 0.0, 0.0])
    True
    >>> x, y, z = sph_to_cart(0.0, np.pi / 2, 1.0, convention="az_colat")
    >>> np.allclose([x, y, z], [1.0, 0.0, 0.0], atol=1e-15)
    True
    """
    az = _a(azimuth)
    a2 = _a(angle2)
    r = _a(radius)
    if convention == "az_el":
        el = a2
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z
    if convention == "az_colat":
        th = a2
        x = r * np.sin(th) * np.cos(az)
        y = r * np.sin(th) * np.sin(az)
        z = r * np.cos(th)
        return x, y, z
    raise ValueError(f"unsupported convention: {convention}")


def cart_to_sph(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    convention: str = "az_el",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert Cartesian (x, y, z) coordinates to spherical.

    Parameters
    ----------
    x, y, z : array_like
        Cartesian coordinates.
    convention : str, optional
        ``"az_el"`` (default) returns elevation as the second angle;
        ``"az_colat"`` returns colatitude instead.

    Returns
    -------
    azimuth : ndarray
        Azimuth angle in radians, in (-pi, pi].
    angle2 : ndarray
        Elevation (``"az_el"``) or colatitude (``"az_colat"``) in radians.
    radius : ndarray
        Radial distance.

    Examples
    --------
    >>> import numpy as np
    >>> az, el, r = cart_to_sph(1.0, 0.0, 0.0)
    >>> np.allclose([az, el, r], [0.0, 0.0, 1.0])
    True
    >>> az, colat, r = cart_to_sph(0.0, 0.0, 1.0, convention="az_colat")
    >>> np.allclose(colat, 0.0)
    True
    """
    x = _a(x)
    y = _a(y)
    z = _a(z)
    r = np.sqrt(x * x + y * y + z * z)
    az = np.arctan2(y, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        if convention == "az_el":
            el = np.arcsin(np.where(r == 0, 0.0, z / r))
            return az, el, r
        if convention == "az_colat":
            th = np.arccos(np.clip(np.where(r == 0, 1.0, z / r), -1.0, 1.0))
            return az, th, r
    raise ValueError(f"unsupported convention: {convention}")


def azel_to_az_colat(azimuth: ArrayLike, elevation: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert (azimuth, elevation) to (azimuth, colatitude).

    Parameters
    ----------
    azimuth : array_like
        Azimuth angle(s) in radians.
    elevation : array_like
        Elevation angle(s) in radians.

    Returns
    -------
    azimuth : ndarray
        Azimuth (unchanged).
    colatitude : ndarray
        Colatitude, computed as ``pi/2 - elevation``.

    Examples
    --------
    >>> import numpy as np
    >>> az, colat = azel_to_az_colat(0.0, np.pi / 2)
    >>> np.allclose(colat, 0.0)
    True
    """
    az = _a(azimuth)
    el = _a(elevation)
    return az, (np.pi / 2.0) - el


def az_colat_to_azel(azimuth: ArrayLike, colatitude: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert (azimuth, colatitude) to (azimuth, elevation).

    Parameters
    ----------
    azimuth : array_like
        Azimuth angle(s) in radians.
    colatitude : array_like
        Colatitude angle(s) in radians.

    Returns
    -------
    azimuth : ndarray
        Azimuth (unchanged).
    elevation : ndarray
        Elevation, computed as ``pi/2 - colatitude``.

    Examples
    --------
    >>> import numpy as np
    >>> az, el = az_colat_to_azel(0.0, 0.0)
    >>> np.allclose(el, np.pi / 2)
    True
    """
    az = _a(azimuth)
    th = _a(colatitude)
    return az, (np.pi / 2.0) - th


def unit_sph_to_cart(azimuth: ArrayLike, angle2: ArrayLike, convention: str = "az_el") -> NDArray[np.float64]:
    """Convert spherical angles to unit-length Cartesian vectors as an [..., 3] array.

    Parameters
    ----------
    azimuth : array_like
        Azimuth angle(s) in radians.
    angle2 : array_like
        Elevation or colatitude in radians (see *convention*).
    convention : str, optional
        ``"az_el"`` (default) or ``"az_colat"``.

    Returns
    -------
    ndarray, shape (..., 3)
        Unit Cartesian vectors stacked along the last axis.

    Examples
    --------
    >>> import numpy as np
    >>> v = unit_sph_to_cart(0.0, 0.0)
    >>> np.allclose(v, [1.0, 0.0, 0.0])
    True
    """
    x, y, z = sph_to_cart(azimuth, angle2, radius=1.0, convention=convention)
    return np.stack([x, y, z], axis=-1)

