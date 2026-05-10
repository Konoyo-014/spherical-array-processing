"""Direction-of-arrival (DOA) estimation examples.

Demonstrates:
- Building a rank-1 SH covariance for a known source direction
- PWD spatial spectrum and peak picking
- MUSIC spatial spectrum (sharper peaks)
- Full simulation pipeline: plane-wave → SHT → covariance → DOA

Run with::

    python examples/core/doa_estimation.py
"""

from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap_repo_import

bootstrap_repo_import()

import spherical_array_processing as sap
from spherical_array_processing.doa import music_spectrum, pwd_spectrum


def angular_distance_deg(az1: float, el1: float, az2: float, el2: float) -> float:
    """Great-circle distance in degrees between two (az, el) points."""
    col1 = np.pi / 2 - el1
    col2 = np.pi / 2 - el2
    cos_sep = (
        np.sin(col1) * np.sin(col2) * np.cos(az1 - az2)
        + np.cos(col1) * np.cos(col2)
    )
    return float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))))


def main() -> None:
    N      = 3
    spec   = sap.SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
    search = sap.array.fibonacci_grid(500)
    Q      = spec.n_coeffs

    # ── 1. Exact rank-1 covariance → PWD ─────────────────────────────────────
    print("=== PWD with exact rank-1 covariance ===")
    src_idx  = 100
    Y        = sap.sh.matrix(spec, search)    # (G, Q)
    y_src    = Y[src_idx]                     # steering vector at source
    R        = np.outer(y_src, y_src.conj())  # rank-1, exact
    result   = pwd_spectrum(R, search, spec, n_peaks=1)
    found    = result.peak_indices[0]
    print(f"  Source at grid index {src_idx}, PWD peak at index {found}  ({'PASS' if found == src_idx else 'FAIL'})")

    # ── 2. MUSIC vs PWD sharpness ─────────────────────────────────────────────
    print("\n=== MUSIC vs PWD sharpness (rank-1 + small noise) ===")
    R_noisy  = R + 0.01 * np.eye(Q)
    pwd_res  = pwd_spectrum(R_noisy, search, spec, n_peaks=1)
    music_res = music_spectrum(R_noisy, search, spec, n_sources=1, n_peaks=1)

    s_pwd   = pwd_res.spectrum
    s_music = music_res.spectrum
    ptm_pwd   = s_pwd.max()   / s_pwd.mean()
    ptm_music = s_music.max() / s_music.mean()
    print(f"  PWD   peak-to-mean ratio: {ptm_pwd:.1f}")
    print(f"  MUSIC peak-to-mean ratio: {ptm_music:.1f}  (should be >> PWD)")

    # ── 3. Full simulation pipeline ───────────────────────────────────────────
    print("\n=== Full pipeline: simulation → SHT → PWD ===")
    N_pipe   = 2
    spec_p   = sap.SHBasisSpec(max_order=N_pipe, basis="complex", angle_convention="az_colat")
    sensors  = sap.array.fibonacci_grid(200)
    geometry = sap.ArrayGeometry(radius_m=0.042, sensor_grid=sensors)

    # Place source at search grid index 42
    src_search_idx = 42
    search_p       = sap.array.fibonacci_grid(300)
    src_grid       = sap.SphericalGrid(
        azimuth   = search_p.azimuth[src_search_idx : src_search_idx + 1],
        angle2    = search_p.angle2[src_search_idx : src_search_idx + 1],
        weights   = np.array([4.0 * np.pi]),
        convention="az_colat",
    )

    _, H   = sap.array.simulate_plane_wave_array_response(64, 8000.0, geometry, src_grid)
    p_mic  = H[5, :, 0]                           # frequency bin 5

    Ys   = sap.sh.matrix(spec_p, sensors)
    nm   = sap.sh.direct_sht(p_mic, Ys, sensors)
    R_p  = np.outer(nm, nm.conj())
    res  = pwd_spectrum(R_p, search_p, spec_p, n_peaks=1)

    az_true  = float(search_p.azimuth[src_search_idx])
    el_true  = float(search_p.elevation[src_search_idx])
    az_found = float(search_p.azimuth[res.peak_indices[0]])
    el_found = float(search_p.elevation[res.peak_indices[0]])
    sep      = angular_distance_deg(az_true, el_true, az_found, el_found)
    print(f"  Source at (az={np.degrees(az_true):.1f}°, el={np.degrees(el_true):.1f}°)")
    print(f"  PWD peak  (az={np.degrees(az_found):.1f}°, el={np.degrees(el_found):.1f}°)")
    print(f"  Angular separation: {sep:.1f}°  (< 45° expected with 200 sensors, N=2)")

    print("\nDone.")


if __name__ == "__main__":
    main()
