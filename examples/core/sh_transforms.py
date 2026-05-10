"""Spherical Harmonic transform examples.

Demonstrates:
- Complex and real SH basis matrices
- SH orthonormality check on a Fibonacci grid
- Forward/inverse SHT for a known function (Y_2^0 = 5th-order monopole harmonic)
- Coefficient conversion between complex and real (tesseral) representations

Run with::

    python examples/core/sh_transforms.py
"""

from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap_repo_import

bootstrap_repo_import()

import spherical_array_processing as sap
from spherical_array_processing.sh import acn_index


def main() -> None:
    # ── Grid ─────────────────────────────────────────────────────────────────
    N = 4
    # Use a large Fibonacci grid for good quadrature approximation
    M = max(300 * (N + 1) ** 2, 500)
    grid = sap.array.fibonacci_grid(M)
    print(f"Using {M}-point Fibonacci grid for order N={N}")

    # ── Complex SH matrix and approximate orthonormality ─────────────────────
    spec_c = sap.SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
    Yc = sap.sh.matrix(spec_c, grid)   # (M, (N+1)²)
    Q  = spec_c.n_coeffs
    # G ≈ Y^H diag(w) Y  should be close to identity
    G = (Yc.conj().T * grid.weights) @ Yc
    err = np.max(np.abs(G - np.eye(Q)))
    print(f"Complex SH Gram matrix max off-diagonal error: {err:.4e}  (< 0.01 expected)")

    # ── Real SH matrix ────────────────────────────────────────────────────────
    spec_r = sap.SHBasisSpec(max_order=N, basis="real", angle_convention="az_colat")
    Yr = sap.sh.matrix(spec_r, grid)
    Gr = (Yr.T * grid.weights) @ Yr
    err_r = np.max(np.abs(Gr - np.eye(Q)))
    print(f"Real SH Gram matrix max off-diagonal error:    {err_r:.4e}  (< 0.01 expected)")

    # ── SHT of a known function: f = Y_2^1 (complex) ─────────────────────────
    q21 = acn_index(2, 1)
    f   = Yc[:, q21]              # evaluate Y_2^1 on the grid
    nm  = sap.sh.direct_sht(f, Yc, grid)
    # Expect nm[q21] ≈ 1, all others ≈ 0
    print(f"\nSHT of Y_2^1:  coeff at (n=2,m=1) = {nm[q21]:.6f}  (expect 1+0j)")
    max_other = np.max(np.abs(np.concatenate([nm[:q21], nm[q21 + 1:]])))
    print(f"  max |other coefficients| = {max_other:.2e}  (expect < 0.01)")

    # ── Coefficient roundtrip: real → complex → real ──────────────────────────
    rng    = np.random.default_rng(7)
    b_real = rng.standard_normal(Q)
    b_cplx = sap.sh.real_to_complex_coeffs(b_real, max_order=N)
    b_back = sap.sh.complex_to_real_coeffs(b_cplx, max_order=N)
    err_rt = np.max(np.abs(b_back - b_real))
    print(f"\nCoeff roundtrip (real→complex→real) max error: {err_rt:.2e}  (expect < 1e-14)")

    print("\nDone.")


if __name__ == "__main__":
    main()
