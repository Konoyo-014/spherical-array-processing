"""Minimal end-to-end plane-wave DOA demo.

Encodes a single synthetic plane-wave source directly in the
spherical-harmonic domain, builds a rank-1 SH covariance from those
encoded coefficients, then scans a dense direction grid with both
:func:`spherical_array_processing.doa.pwd_spectrum` (Plane-Wave
Decomposition / conventional beamforming in the SH domain) and
:func:`spherical_array_processing.doa.music_spectrum` (subspace MUSIC
with the noise subspace inferred from the rank-1 structure).

The script uses only the stable public API — nothing from
``spherical_array_processing.repro`` / ``regression`` /
``experimental`` — and has no external data dependency.

Run after installing the wheel::

    python -m spherical_array_processing.examples.plane_wave_doa

Programmatic use::

    from spherical_array_processing.examples.plane_wave_doa import (
        run_example,
    )
    out = run_example(az_deg=73.0, col_deg=62.0)
    print(out["pwd_error_deg"], out["music_error_deg"])

The returned dictionary carries the exact recovered directions (in
degrees) for both estimators plus the scan-grid angular-resolution
limit, so it is easy to assert on either correctness or the
estimator-vs-grid error floor from a notebook or a test.
"""

from __future__ import annotations

import numpy as np

from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.coords import unit_sph_to_cart
from spherical_array_processing.doa.spectra import (
    music_spectrum,
    pwd_spectrum,
)
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


def _az_col_to_unit(az: float, col: float) -> np.ndarray:
    return unit_sph_to_cart(
        np.array([az]), np.array([col]), convention="az_colat",
    )[0]


def _angular_error_deg(u_true: np.ndarray, u_est: np.ndarray) -> float:
    dot = float(np.clip(np.dot(u_true, u_est), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def run_example(
    az_deg: float = 73.0,
    col_deg: float = 62.0,
    max_order: int = 4,
    n_scan: int = 2562,
) -> dict:
    """Round-trip a single plane-wave DOA through PWD and MUSIC.

    Parameters
    ----------
    az_deg, col_deg : float
        True source direction in degrees (azimuth + colatitude).
    max_order : int
        Maximum SH order used for the rank-1 SH covariance.  The
        defaults are deliberately small so the demo runs in <100 ms
        on a laptop while still resolving DOA well below 1°.
    n_scan : int
        Number of Fibonacci scan directions; the intrinsic angular
        resolution is ≈ ``sqrt(4π / n_scan)`` radians.  The default
        yields ≈ 2.5° resolution, which is enough to demonstrate
        sub-degree-residual recovery but fast enough for an example.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``az_deg``, ``col_deg`` — the input (true) DOA.
        - ``pwd_az_deg``, ``pwd_col_deg`` — PWD peak (degrees).
        - ``music_az_deg``, ``music_col_deg`` — MUSIC peak (degrees).
        - ``pwd_error_deg``, ``music_error_deg`` — angular error vs
          the true DOA (degrees).
        - ``grid_resolution_deg`` — expected scan-grid error floor.
    """
    az_true = np.radians(az_deg)
    col_true = np.radians(col_deg)

    # --- 1.  Build the rank-1 SH covariance of a plane wave.
    spec = SHBasisSpec(
        max_order=max_order, basis="complex", normalization="orthonormal",
    )
    true_grid = SphericalGrid(
        azimuth=np.array([az_true]),
        angle2=np.array([col_true]),
        convention="az_colat",
    )
    # Per the package's plane-wave SH encoding convention, the SH
    # coefficients of a unit plane wave from (az, colat) are Y(az, colat)*.
    y_true = sh_matrix(spec, true_grid).ravel()  # (N+1)^2
    sh_cov = np.conj(y_true)[:, None] * y_true[None, :]

    # --- 2.  Dense scan grid (Fibonacci) shared by both estimators.
    scan_grid = fibonacci_grid(n_scan)
    grid_resolution_deg = float(np.degrees(np.sqrt(4.0 * np.pi / n_scan)))

    # --- 3.  PWD.
    pwd = pwd_spectrum(sh_cov, scan_grid, spec)
    idx_pwd = int(pwd.peak_indices[0])
    pwd_az = float(scan_grid.azimuth[idx_pwd])
    pwd_col = float(scan_grid.angle2[idx_pwd])

    # --- 4.  MUSIC with the known rank-1 source structure.
    music = music_spectrum(sh_cov, scan_grid, spec, n_sources=1)
    idx_music = int(music.peak_indices[0])
    music_az = float(scan_grid.azimuth[idx_music])
    music_col = float(scan_grid.angle2[idx_music])

    u_true = _az_col_to_unit(az_true, col_true)
    u_pwd = _az_col_to_unit(pwd_az, pwd_col)
    u_music = _az_col_to_unit(music_az, music_col)

    return {
        "az_deg": az_deg,
        "col_deg": col_deg,
        "pwd_az_deg": float(np.degrees(pwd_az)),
        "pwd_col_deg": float(np.degrees(pwd_col)),
        "music_az_deg": float(np.degrees(music_az)),
        "music_col_deg": float(np.degrees(music_col)),
        "pwd_error_deg": _angular_error_deg(u_true, u_pwd),
        "music_error_deg": _angular_error_deg(u_true, u_music),
        "grid_resolution_deg": grid_resolution_deg,
    }


def main() -> None:
    """CLI entry point used by ``python -m`` invocations."""
    out = run_example()
    print(
        "plane-wave DOA recovery demo\n"
        f"  true     : az={out['az_deg']:6.2f}°, colat={out['col_deg']:6.2f}°\n"
        f"  PWD      : az={out['pwd_az_deg']:6.2f}°, colat={out['pwd_col_deg']:6.2f}°"
        f"  (error {out['pwd_error_deg']:.2f}°)\n"
        f"  MUSIC    : az={out['music_az_deg']:6.2f}°, colat={out['music_col_deg']:6.2f}°"
        f"  (error {out['music_error_deg']:.2f}°)\n"
        f"  grid step ≈ {out['grid_resolution_deg']:.2f}°"
    )


if __name__ == "__main__":  # pragma: no cover — manual invocation
    main()
