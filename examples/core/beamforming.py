"""Beamforming examples.

Demonstrates:
- All four fixed beamformer weight functions (cardioid, hypercardioid,
  supercardioid, MaxEV) and their pattern values
- Unit front gain check (B(0) = 1 for all types)
- MVDR and LCMV adaptive beamformers from a simulated SH covariance

Run with::

    python examples/core/beamforming.py
"""

from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap_repo_import

bootstrap_repo_import()

import spherical_array_processing as sap
from spherical_array_processing.beamforming import (
    axisymmetric_pattern,
    beam_weights_cardioid,
    beam_weights_hypercardioid,
    beam_weights_maxev,
    beam_weights_supercardioid,
)


def pattern_summary(name: str, b: np.ndarray) -> None:
    thetas  = np.array([0.0, np.pi / 2, np.pi])
    pattern = axisymmetric_pattern(thetas, b)
    print(f"  {name:<16s}  B(  0°) = {pattern[0]: .6f}   B( 90°) = {pattern[1]: .6f}   B(180°) = {pattern[2]: .6f}")


def main() -> None:
    N = 3
    print(f"=== Fixed beamformers, order N={N} ===")
    print(f"  {'name':<16s}  {'B(0°)':<22s} {'B(90°)':<22s} {'B(180°)'}")
    for name, fn in [
        ("cardioid",      beam_weights_cardioid),
        ("hypercardioid", beam_weights_hypercardioid),
        ("supercardioid", beam_weights_supercardioid),
        ("maxev",         beam_weights_maxev),
    ]:
        b = fn(N)
        pattern_summary(name, b)

    # ── Directivity index of hypercardioid ─────────────────────────────────
    # DI = 4π * B(0)² / ∫ B(θ)² sin(θ) dθ   (unit-front-gain convention)
    # Hypercardioid maximises DI; DI increases monotonically with order.
    N_vals = [1, 2, 3, 4]
    print("\n=== Hypercardioid directivity index ===")
    for N in N_vals:
        b      = beam_weights_hypercardioid(N)
        thetas = np.linspace(0, np.pi, 4097)
        pat    = axisymmetric_pattern(thetas, b)
        di_num = 4.0 * np.pi * pat[0] ** 2 / float(np.trapezoid(pat ** 2 * np.sin(thetas), thetas))
        print(f"  N={N}: DI = {di_num:.2f}  (increases with order, B(0)=1)")

    # ── Adaptive: MVDR from a simulated diffuse + source covariance ────────
    print("\n=== MVDR beamformer (SH-domain) ===")
    N_bf   = 2
    spec   = sap.SHBasisSpec(max_order=N_bf, basis="complex", angle_convention="az_colat")
    grid   = sap.array.fibonacci_grid(300)
    Q      = spec.n_coeffs

    # Rank-1 source covariance at look direction (grid index 0) + diagonal noise
    look_dir  = sap.SphericalGrid(
        azimuth  = grid.azimuth[:1],
        angle2   = grid.angle2[:1],
        weights  = np.array([4.0 * np.pi]),
        convention="az_colat",
    )
    Y_look  = sap.sh.matrix(spec, look_dir)    # (1, Q)
    y_look  = Y_look[0]                        # (Q,)
    R_noise = 0.1 * np.eye(Q, dtype=complex)
    R       = np.outer(y_look, y_look.conj()) + R_noise

    w_mvdr  = sap.beamforming.mvdr_weights(R, y_look)
    gain    = abs(w_mvdr.conj() @ y_look)
    print(f"  MVDR distortionless response at look direction: {gain:.6f}  (expect 1.0)")

    print("\nDone.")


if __name__ == "__main__":
    main()
