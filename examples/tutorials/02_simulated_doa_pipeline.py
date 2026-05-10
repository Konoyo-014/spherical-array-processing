"""End-to-end tutorial: simulate a source covariance and estimate DOA.

Run from the repository root with:

    python examples/tutorials/02_simulated_doa_pipeline.py
"""

from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap_repo_import

bootstrap_repo_import()

import spherical_array_processing as sap


def main() -> None:
    order = 3
    spec = sap.SHBasisSpec(max_order=order, basis="complex", angle_convention="az_colat")
    search = sap.array.fibonacci_grid(600)
    y = sap.sh.matrix(spec, search)

    source_index = 137
    steering = np.conj(y[source_index])
    covariance = np.outer(steering, steering.conj()) + 0.01 * np.eye(spec.n_coeffs)

    pwd = sap.doa.pwd_spectrum(covariance, search, spec, n_peaks=1)
    music = sap.doa.music_spectrum(covariance, search, spec, n_sources=1, n_peaks=1)

    pwd_index = int(pwd.peak_indices[0])
    music_index = int(music.peak_indices[0])
    pwd_error = float(
        sap.coords.angular_distance_deg(
            search.azimuth[source_index],
            search.angle2[source_index],
            search.azimuth[pwd_index],
            search.angle2[pwd_index],
            convention="az_colat",
        )
    )
    music_error = float(
        sap.coords.angular_distance_deg(
            search.azimuth[source_index],
            search.angle2[source_index],
            search.azimuth[music_index],
            search.angle2[music_index],
            convention="az_colat",
        )
    )

    print("=== Simulated DOA pipeline ===")
    print(f"source grid index: {source_index}")
    print(f"PWD peak index: {pwd_index}, angular error: {pwd_error:.2f} deg")
    print(f"MUSIC peak index: {music_index}, angular error: {music_error:.2f} deg")

    if pwd_error > 10.0:
        raise RuntimeError("PWD did not recover the simulated source direction")
    if music_error > 10.0:
        raise RuntimeError("MUSIC did not recover the simulated source direction")


if __name__ == "__main__":
    main()
