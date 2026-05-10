"""End-to-end tutorial: radial modal response and equalization.

Run from the repository root with:

    python examples/tutorials/03_modal_equalization_pipeline.py
"""

from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap_repo_import

bootstrap_repo_import()

import spherical_array_processing as sap


def main() -> None:
    order = 3
    n_bins = 24
    k_radius = np.linspace(1.0, 6.0, n_bins)
    n_coeffs = (order + 1) ** 2

    rng = np.random.default_rng(12)
    true_plane_wave_sh = rng.standard_normal((n_bins, n_coeffs)) + 1j * rng.standard_normal(
        (n_bins, n_coeffs)
    )

    modal_response = sap.acoustics.bn_matrix(order, k_radius, sphere="rigid")
    recorded_pressure_sh = modal_response * true_plane_wave_sh
    recovered = sap.acoustics.equalize_modal_coeffs(
        recorded_pressure_sh,
        modal_response,
        reg_param=1e-12,
        reg_type="tikhonov",
    )
    relative_error = float(np.linalg.norm(recovered - true_plane_wave_sh) / np.linalg.norm(true_plane_wave_sh))

    print("=== Modal equalization pipeline ===")
    print(f"frequency bins: {n_bins}")
    print(f"SH channels: {n_coeffs}")
    print(f"minimum modal magnitude: {np.min(np.abs(modal_response)):.3f}")
    print(f"relative recovery error: {relative_error:.2e}")

    if relative_error > 1e-6:
        raise RuntimeError("modal equalization recovery error is unexpectedly large")


if __name__ == "__main__":
    main()
