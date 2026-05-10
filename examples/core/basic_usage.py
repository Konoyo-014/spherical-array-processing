"""Basic usage demo for spherical-array-processing.

Demonstrates the most common entry points:
- Building a Fibonacci grid
- Computing a complex SH matrix
- Running a forward/inverse SHT round-trip
- Computing a max-DI beamformer pattern

Run with::

    python examples/core/basic_usage.py
"""

from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap_repo_import

bootstrap_repo_import()

import spherical_array_processing as sap


def main() -> None:
    # ── Spatial sampling ──────────────────────────────────────────────────────
    grid = sap.array.fibonacci_grid(100)
    print(f"Fibonacci grid: {grid.size} directions, weights sum = {grid.weights.sum():.4f} (≈ 4π = {4*np.pi:.4f})")

    # ── SH basis matrix ───────────────────────────────────────────────────────
    spec = sap.SHBasisSpec(max_order=3, basis="complex", angle_convention="az_colat")
    Y = sap.sh.matrix(spec, grid)     # (100, 16)
    print(f"SH matrix shape: {Y.shape}  (grid_size × (N+1)²)")

    # ── Forward / inverse SHT round-trip ─────────────────────────────────────
    # Note: the SHT projects f onto the N=3 SH subspace; the reconstruction
    # error is dominated by the high-frequency content of f, not quadrature.
    rng = np.random.default_rng(42)
    f = rng.standard_normal(grid.size)
    nm    = sap.sh.direct_sht(f, Y, grid)   # (16,) coefficients
    f_rec = sap.sh.inverse_sht(nm, Y).real  # (100,) reconstruction
    rel_err = np.linalg.norm(f - f_rec) / np.linalg.norm(f)
    print(f"SHT round-trip relative error: {rel_err:.4f}  (random signal, large error expected)")

    # ── Hypercardioid beam pattern ────────────────────────────────────────────
    b       = sap.beamforming.beam_weights_hypercardioid(3)
    thetas  = np.array([0.0, np.pi / 2, np.pi])
    pattern = sap.beamforming.axisymmetric_pattern(thetas, b)
    print(f"Hypercardioid N=3:  B(0°)={pattern[0]:.4f}  B(90°)={pattern[1]:.4f}  B(180°)={pattern[2]:.4f}")


if __name__ == "__main__":
    main()
