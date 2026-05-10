"""End-to-end tutorial: SH transform plus fixed beamforming.

Run from the repository root with:

    python examples/tutorials/01_sht_and_beamforming.py
"""

from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap_repo_import

bootstrap_repo_import()

import spherical_array_processing as sap
from spherical_array_processing.sh import acn_index


def main() -> None:
    order = 3
    grid = sap.array.fibonacci_grid(5000)
    spec = sap.SHBasisSpec(max_order=order, basis="complex", angle_convention="az_colat")
    y = sap.sh.matrix(spec, grid)

    target = np.zeros(spec.n_coeffs, dtype=np.complex128)
    target[acn_index(2, 1)] = 1.0
    field = sap.sh.inverse_sht(target, y)
    recovered = sap.sh.direct_sht(field, y, grid)
    coeff_error = float(np.max(np.abs(recovered - target)))

    weights = sap.beamforming.beam_weights_hypercardioid(order)
    theta = np.array([0.0, np.pi / 2.0, np.pi])
    pattern = sap.beamforming.axisymmetric_pattern(theta, weights)

    print("=== SH transform plus fixed beamforming ===")
    print(f"grid directions: {grid.size}")
    print(f"SH coefficient count: {spec.n_coeffs}")
    print(f"max coefficient recovery error: {coeff_error:.2e}")
    print(
        "hypercardioid pattern: "
        f"B(0 deg)={pattern[0]:.6f}, "
        f"B(90 deg)={pattern[1]:.6f}, "
        f"B(180 deg)={pattern[2]:.6f}"
    )

    if coeff_error > 1e-3:
        raise RuntimeError("SHT recovery error is unexpectedly large")
    if abs(pattern[0] - 1.0) > 1e-10:
        raise RuntimeError("beamformer front gain is not normalized to one")


if __name__ == "__main__":
    main()
