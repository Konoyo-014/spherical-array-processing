"""Cross-module consistency tests for the v0.4.0 beta surface.

Each test wires together at least two modules (simulation / acoustics / sh
/ rotation / doa) so that sign-convention and normalization regressions
surface as a numerical failure rather than as silent drift.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.acoustics import bn_matrix, plane_wave_radial_bn
from spherical_array_processing.array import (
    fibonacci_grid,
    gauss_legendre_sampling,
    simulate_plane_wave_array_response,
    simulate_sh_array_response,
)
from spherical_array_processing.sh import (
    direct_sht,
    matrix as sh_matrix,
    rotate_sh_coeffs,
    sh_rotation_matrix_complex,
    sh_rotation_matrix_real,
)
from spherical_array_processing.sh.basis import acn_index, complex_matrix, real_matrix
from spherical_array_processing.types import ArrayGeometry, SHBasisSpec, SphericalGrid


# ---------------------------------------------------------------------------
# simulator convention & reduction to free-field
# ---------------------------------------------------------------------------

class TestSimulatorConventions:
    def _geometry(self, n_mics: int = 24, radius_m: float = 0.042) -> ArrayGeometry:
        grid = fibonacci_grid(n_mics)
        return ArrayGeometry(sensor_grid=grid, radius_m=radius_m)

    def _sources(self):
        return SphericalGrid(
            azimuth=np.array([0.0, 1.7, 2.5]),
            angle2=np.array([1.2, 0.6, 2.0]),
            convention="az_colat",
        )

    def test_open_sh_simulator_matches_free_field_at_high_order(self):
        """simulate_sh_array_response(open, N=40) reproduces free-field phase.

        The two APIs are independent code paths: the free-field form uses
        `exp(+j k · û_s · r_m)` directly, while the SH form sums the
        Jacobi–Anger expansion with modal coefficients.  Matching to
        machine precision is a strong correctness signal for the DOA
        phase convention.
        """
        geom = self._geometry()
        src = self._sources()
        _, h_free = simulate_plane_wave_array_response(256, 16000.0, geom, src)
        _, h_sh = simulate_sh_array_response(
            256, 16000.0, geom, src, max_order=40, array_type="open"
        )
        assert_allclose(h_sh, h_free, atol=1e-12)

    def test_open_and_rigid_have_unity_dc(self):
        geom = self._geometry()
        src = self._sources()
        for array_type in ("open", "rigid"):
            _, h = simulate_sh_array_response(
                128, 8000.0, geom, src, max_order=10, array_type=array_type
            )
            assert_allclose(h[0, :, :], 1.0, atol=1e-12)

    def test_cardioid_dc_is_capsule_directional_response(self):
        """At DC the cardioid modal sum must reproduce ``0.5·(1 + cos γ)``
        rather than collapse to a uniform ``1``.  Guards against the DC
        override regression fixed in v0.4.0b1.
        """
        from spherical_array_processing.coords import unit_sph_to_cart

        geom = self._geometry()
        src = self._sources()
        _, h = simulate_sh_array_response(
            128, 8000.0, geom, src, max_order=10, array_type="cardioid"
        )
        mic_u = unit_sph_to_cart(
            geom.sensor_grid.azimuth,
            geom.sensor_grid.angle2,
            convention=geom.sensor_grid.convention,
        )
        src_u = unit_sph_to_cart(
            src.azimuth, src.angle2, convention=src.convention
        )
        cos_gamma = np.clip(mic_u @ src_u.T, -1.0, 1.0)
        expected = 0.5 * (1.0 + cos_gamma)
        assert_allclose(h[0, :, :].real, expected, atol=1e-12)
        assert_allclose(h[0, :, :].imag, 0.0, atol=1e-12)

    def test_pwd_locates_source_after_radial_compensation(self):
        """Full simulate → SHT → radial compensation → PWD chain.

        Under the package's ``y(q̂)ᵀ R y(q̂)*`` PWD convention the
        physical SHT output can be squared into ``R = E[c cᴴ]`` and fed
        into :func:`pwd_spectrum` directly; no user-side conjugation is
        required.
        """
        from spherical_array_processing.doa import pwd_spectrum

        n_mics = 200
        mic_grid = fibonacci_grid(n_mics)
        geom = ArrayGeometry(sensor_grid=mic_grid, radius_m=0.042)
        az_true = np.radians(60.0)
        colat_true = np.radians(50.0)
        src_grid = SphericalGrid(
            azimuth=[az_true], angle2=[colat_true], convention="az_colat"
        )
        fft_len, fs = 128, 16000.0
        max_order = 4
        spec = SHBasisSpec(max_order=max_order, basis="complex")

        _, H = simulate_sh_array_response(
            fft_len, fs, geom, src_grid, max_order=12, array_type="open"
        )
        bin_idx = 25
        p_mic = H[bin_idx, :, 0]

        Y = sh_matrix(spec, mic_grid)
        nm = direct_sht(p_mic, Y, mic_grid)

        freqs = np.arange(fft_len // 2 + 1) * fs / fft_len
        kR = 2 * np.pi * freqs[bin_idx] / 343.0 * geom.radius_m
        bn = bn_matrix(
            max_order, kr=np.array([kR]), sphere="open", repeat_per_order=True
        )[0]
        nm_eq = nm / np.where(np.abs(bn) > 1e-10, bn, 1.0)

        R = np.outer(nm_eq, nm_eq.conj())
        scan = fibonacci_grid(4000)
        result = pwd_spectrum(R, scan, spec, n_peaks=1)
        peak_idx = int(result.peak_indices[0])

        az_found = scan.azimuth[peak_idx]
        colat_found = scan.angle2[peak_idx]
        cos_sep = (
            np.sin(colat_true) * np.sin(colat_found) * np.cos(az_true - az_found)
            + np.cos(colat_true) * np.cos(colat_found)
        )
        sep_deg = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
        assert sep_deg < 3.0, (
            f"PWD peak off by {sep_deg:.2f}° — expected ≲ grid resolution"
        )


# ---------------------------------------------------------------------------
# Wigner-D rotation consistency
# ---------------------------------------------------------------------------

class TestWignerDRotation:
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 7])
    def test_identity(self, n: int):
        from spherical_array_processing.sh import wigner_D

        D = wigner_D(n, 0.0, 0.0, 0.0)
        assert_allclose(D, np.eye(2 * n + 1), atol=1e-14)

    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_unitarity(self, n: int):
        from spherical_array_processing.sh import wigner_D

        D = wigner_D(n, 0.7, 1.2, 2.3)
        assert_allclose(D @ D.conj().T, np.eye(2 * n + 1), atol=1e-10)

    def test_z_rotation_composition(self):
        from spherical_array_processing.sh import wigner_D

        a1, a2 = 0.3, 1.1
        D1 = wigner_D(6, a1, 0.0, 0.0)
        D2 = wigner_D(6, a2, 0.0, 0.0)
        D_sum = wigner_D(6, a1 + a2, 0.0, 0.0)
        assert_allclose(D1 @ D2, D_sum, atol=1e-12)

    def test_sh_rotation_matches_direct_evaluation(self):
        """Rotating SH of a unit-amplitude plane-wave direction via Wigner-D
        matches re-evaluating the SH basis at the rotated direction.
        """
        n_order = 5
        spec = SHBasisSpec(max_order=n_order, basis="complex")
        az0, colat0 = np.radians(60.0), np.radians(50.0)
        k_u0 = np.array(
            [
                np.sin(colat0) * np.cos(az0),
                np.sin(colat0) * np.sin(az0),
                np.cos(colat0),
            ]
        )
        alpha, beta, gamma = np.radians(30.0), np.radians(20.0), np.radians(10.0)
        Rz_a = np.array(
            [[np.cos(alpha), -np.sin(alpha), 0.0], [np.sin(alpha), np.cos(alpha), 0.0], [0.0, 0.0, 1.0]]
        )
        Ry_b = np.array(
            [[np.cos(beta), 0.0, np.sin(beta)], [0.0, 1.0, 0.0], [-np.sin(beta), 0.0, np.cos(beta)]]
        )
        Rz_g = np.array(
            [[np.cos(gamma), -np.sin(gamma), 0.0], [np.sin(gamma), np.cos(gamma), 0.0], [0.0, 0.0, 1.0]]
        )
        R_active = Rz_a @ Ry_b @ Rz_g
        k_u_rot = R_active @ k_u0
        colat_rot = np.arccos(np.clip(k_u_rot[2], -1.0, 1.0))
        az_rot = np.arctan2(k_u_rot[1], k_u_rot[0]) % (2 * np.pi)

        grid0 = SphericalGrid(azimuth=np.array([az0]), angle2=np.array([colat0]), convention="az_colat")
        grid_rot = SphericalGrid(
            azimuth=np.array([az_rot]), angle2=np.array([colat_rot]), convention="az_colat"
        )
        c_orig = complex_matrix(spec, grid0)[0].conj()
        c_direct = complex_matrix(spec, grid_rot)[0].conj()
        R_mat = sh_rotation_matrix_complex(n_order, alpha, beta, gamma)
        c_rotated = R_mat @ c_orig
        assert_allclose(c_rotated, c_direct, atol=1e-12)

    def test_real_rotation_is_orthogonal(self):
        for n in (1, 3, 5, 7):
            R = sh_rotation_matrix_real(n, 0.7, 1.2, 0.3)
            assert_allclose(R @ R.T, np.eye((n + 1) ** 2), atol=1e-10)

    def test_rotate_sh_coeffs_roundtrip(self):
        rng = np.random.default_rng(0)
        n_order = 4
        c = rng.normal(size=(3, (n_order + 1) ** 2)) + 1j * rng.normal(
            size=(3, (n_order + 1) ** 2)
        )
        alpha, beta, gamma = 0.5, 0.3, 0.1
        c_rot = rotate_sh_coeffs(c, n_order, alpha, beta, gamma, basis="complex")
        c_back = rotate_sh_coeffs(
            c_rot, n_order, -gamma, -beta, -alpha, basis="complex"
        )
        assert_allclose(c_back, c, atol=1e-10)

    @pytest.mark.parametrize("n", [1, 3, 7, 15])
    def test_jy_method_matches_sakurai(self, n: int):
        """The Jy-eigendecomposition backend must agree with the Sakurai
        direct-sum reference to machine precision for ``n ≤ 15`` (the
        regime where Sakurai itself is still accurate).
        """
        from spherical_array_processing.sh.rotation import (
            _wigner_small_d_sakurai, _wigner_small_d_jy,
        )

        d_sak = _wigner_small_d_sakurai(n, 0.7)
        d_jy = _wigner_small_d_jy(n, 0.7)
        assert_allclose(d_jy, d_sak, atol=1e-12)

    @pytest.mark.parametrize("n", [30, 50, 80])
    def test_jy_method_stays_unitary_at_high_order(self, n: int):
        """Jy backend must preserve unitarity well past the order where
        the Sakurai direct sum loses precision.
        """
        from spherical_array_processing.sh.rotation import _wigner_small_d_jy

        d = _wigner_small_d_jy(n, 1.2)
        assert_allclose(d @ d.T, np.eye(2 * n + 1), atol=1e-12)


# ---------------------------------------------------------------------------
# bn_matrix / plane_wave_radial_bn direct relationships
# ---------------------------------------------------------------------------

class TestBnMatrixRelationships:
    def test_open_bn_matches_4pi_i_n_jn(self):
        from scipy.special import spherical_jn

        kr = np.linspace(0.1, 5.0, 20)
        for n in (0, 1, 2, 5, 10):
            bn = plane_wave_radial_bn(n, kr, sphere="open")
            expected = 4 * np.pi * (1j ** n) * spherical_jn(n, kr)
            assert_allclose(bn, expected, atol=1e-14)

    def test_bn_matrix_repeat_per_order_layout(self):
        """Per-mode expansion repeats each B_n value (2n+1) times in ACN order."""
        kr = np.array([1.0, 3.0])
        N = 3
        bn_short = bn_matrix(N, kr, sphere="rigid", repeat_per_order=False)
        bn_full = bn_matrix(N, kr, sphere="rigid", repeat_per_order=True)
        assert bn_full.shape == (kr.size, (N + 1) ** 2)
        for n in range(N + 1):
            count = 2 * n + 1
            start = n * n  # ACN of (n, -n) = n²
            expected = np.broadcast_to(bn_short[:, n : n + 1], (kr.size, count))
            assert_allclose(bn_full[:, start : start + count], expected, atol=0.0)

    def test_cardioid_is_unit_front_gain(self):
        """Half-factor normalisation: sphere=cardioid corresponds to 0.5·(1+cos θ)."""
        from scipy.special import spherical_jn

        kr = np.linspace(0.1, 5.0, 8)
        for n in range(5):
            bn_card = plane_wave_radial_bn(n, kr, sphere="cardioid")
            jn = spherical_jn(n, kr)
            jnd = spherical_jn(n, kr, derivative=True)
            # α = 0.5 cardioid: 4π iⁿ (0.5 j_n − j·0.5 j_n') = 2π iⁿ (j_n − j j_n')
            expected = 2 * np.pi * (1j ** n) * (jn - 1j * jnd)
            assert_allclose(bn_card, expected, atol=1e-14)

    def test_directional_reduces_to_canonical_limits(self):
        """``sphere="directional"`` must match ``open`` at α=1, ``cardioid``
        at α=0.5, and the pure dipole ``-4π iⁿ⁺¹ j_n'`` at α=0.
        """
        from scipy.special import spherical_jn

        kr = np.linspace(0.2, 4.0, 6)
        for n in range(5):
            b_open = plane_wave_radial_bn(n, kr, sphere="open")
            b_card = plane_wave_radial_bn(n, kr, sphere="cardioid")
            b_dip = -4 * np.pi * (1j ** (n + 1)) * spherical_jn(n, kr, derivative=True)

            assert_allclose(
                plane_wave_radial_bn(n, kr, sphere="directional", dir_coeff=1.0),
                b_open, atol=1e-14,
            )
            assert_allclose(
                plane_wave_radial_bn(n, kr, sphere="directional", dir_coeff=0.5),
                b_card, atol=1e-14,
            )
            assert_allclose(
                plane_wave_radial_bn(n, kr, sphere="directional", dir_coeff=0.0),
                b_dip, atol=1e-14,
            )

    def test_directional_requires_dir_coeff_and_range(self):
        import pytest

        with pytest.raises(ValueError, match="dir_coeff"):
            plane_wave_radial_bn(1, np.array([1.0]), sphere="directional")
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            plane_wave_radial_bn(
                1, np.array([1.0]), sphere="directional", dir_coeff=1.5
            )
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            plane_wave_radial_bn(
                1, np.array([1.0]), sphere="directional", dir_coeff=-0.1
            )

    def test_esprit_recovers_single_source_exactly(self):
        """ESPRIT on a synthetic rank-1 SH covariance must recover the
        source direction to numerical precision (beyond grid
        resolution, since ESPRIT is grid-free)."""
        from spherical_array_processing.doa import esprit_doa
        from spherical_array_processing.sh.basis import complex_matrix

        N = 4
        spec = SHBasisSpec(max_order=N, basis="complex")
        for az_deg, col_deg in [(45.0, 60.0), (120.0, 30.0), (285.0, 110.0)]:
            az = np.radians(az_deg)
            col = np.radians(col_deg)
            src_grid = SphericalGrid(
                azimuth=[az], angle2=[col], convention="az_colat"
            )
            Y_src = complex_matrix(spec, src_grid)
            # physical SHT: c ∝ Y*(k̂)
            c = np.conj(Y_src[0])
            R = np.outer(c, c.conj()) + 1e-3 * np.eye(spec.n_coeffs, dtype=complex)
            R = 0.5 * (R + R.conj().T)
            result = esprit_doa(R, n_sources=1, convention="az_colat")
            az_f = result.grid.azimuth[0]
            col_f = result.grid.angle2[0]
            cos_sep = (
                np.sin(col) * np.sin(col_f) * np.cos(az - az_f)
                + np.cos(col) * np.cos(col_f)
            )
            sep = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
            assert sep < 0.1, (
                f"ESPRIT off by {sep:.4f}° at az={az_deg}, colat={col_deg}"
            )

    def test_esprit_recovers_two_sources(self):
        from spherical_array_processing.doa import esprit_doa
        from spherical_array_processing.sh.basis import complex_matrix

        N = 4
        spec = SHBasisSpec(max_order=N, basis="complex")
        az_true = np.array([np.radians(45.0), np.radians(200.0)])
        col_true = np.array([np.radians(60.0), np.radians(70.0)])
        src_grid = SphericalGrid(
            azimuth=az_true, angle2=col_true, convention="az_colat"
        )
        Y_src = complex_matrix(spec, src_grid)
        c = np.conj(Y_src)
        R = c.T @ np.diag([1.0, 1.5]) @ c.conj()
        R = R + 5e-3 * np.eye(spec.n_coeffs, dtype=complex)
        R = 0.5 * (R + R.conj().T)
        result = esprit_doa(R, n_sources=2, convention="az_colat")
        # match-up regardless of permutation
        for (az_t, col_t) in zip(az_true, col_true):
            seps = []
            for (az_f, col_f) in zip(result.grid.azimuth, result.grid.angle2):
                cos_sep = (
                    np.sin(col_t) * np.sin(col_f) * np.cos(az_t - az_f)
                    + np.cos(col_t) * np.cos(col_f)
                )
                seps.append(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))))
            assert min(seps) < 0.5, (
                f"source at ({np.degrees(az_t):.1f}, {np.degrees(col_t):.1f}) "
                f"not matched within 0.5°: {seps}"
            )

    def test_aic_mdl_recover_source_count_at_high_snr(self):
        """Both AIC and MDL should return ``K_true`` for a high-SNR,
        many-snapshot synthetic scenario where they are known to be
        consistent estimators.
        """
        from spherical_array_processing.doa import estimate_n_sources
        from spherical_array_processing.sh.basis import complex_matrix

        N = 4
        q = (N + 1) ** 2
        spec = SHBasisSpec(max_order=N, basis="complex")
        rng = np.random.default_rng(0)
        for K_true in (1, 2, 3):
            az = rng.uniform(0, 2 * np.pi, K_true)
            col = rng.uniform(0.3, np.pi - 0.3, K_true)
            src_grid = SphericalGrid(
                azimuth=az, angle2=col, convention="az_colat"
            )
            Y_src = complex_matrix(spec, src_grid)
            c_src = np.conj(Y_src)
            T = 300
            noise_var = 0.02
            snapshots = np.zeros((T, q), dtype=complex)
            for t in range(T):
                s = np.exp(1j * rng.uniform(0, 2 * np.pi, K_true))
                snapshots[t] = s @ c_src + (
                    rng.normal(size=q) + 1j * rng.normal(size=q)
                ) * np.sqrt(noise_var / 2)
            r = snapshots.T @ snapshots.conj() / T
            r = 0.5 * (r + r.conj().T)
            assert estimate_n_sources(r, n_snapshots=T, criterion="aic") == K_true
            assert estimate_n_sources(r, n_snapshots=T, criterion="mdl") == K_true

    def test_estimate_n_sources_accepts_eigvals_directly(self):
        from spherical_array_processing.doa import estimate_n_sources

        # Simulated eigenvalue spectrum with 2 dominant + 7 noise eigvals
        eigvals = np.array([10.0, 8.0, 0.05, 0.06, 0.04, 0.055, 0.045, 0.05, 0.048])
        assert estimate_n_sources(eigvals, n_snapshots=500, criterion="mdl") == 2

    def test_esprit_az_el_vs_az_colat(self):
        from spherical_array_processing.doa import esprit_doa
        from spherical_array_processing.sh.basis import complex_matrix

        N = 3
        spec = SHBasisSpec(max_order=N, basis="complex")
        src_grid = SphericalGrid(
            azimuth=[np.radians(30.0)], angle2=[np.radians(50.0)],
            convention="az_colat",
        )
        Y_src = complex_matrix(spec, src_grid)
        c = np.conj(Y_src[0])
        R = np.outer(c, c.conj()) + 1e-4 * np.eye(spec.n_coeffs, dtype=complex)
        R = 0.5 * (R + R.conj().T)
        r_el = esprit_doa(R, n_sources=1, convention="az_el")
        r_col = esprit_doa(R, n_sources=1, convention="az_colat")
        # elevation + colatitude = π/2
        assert_allclose(
            r_el.grid.angle2[0] + r_col.grid.angle2[0], np.pi / 2.0, atol=1e-12
        )
        assert_allclose(r_el.grid.azimuth[0], r_col.grid.azimuth[0], atol=1e-12)

    def test_directional_dc_matches_capsule_pattern_in_simulator(self):
        """``simulate_sh_array_response(array_type="directional")`` must give
        ``H(0) = α + (1-α)·cos γ`` at DC for any valid α.
        """
        from spherical_array_processing.coords import unit_sph_to_cart

        mic_grid = fibonacci_grid(24)
        geom = ArrayGeometry(sensor_grid=mic_grid, radius_m=0.05)
        src = SphericalGrid(
            azimuth=[0.3, 1.2], angle2=[0.7, 1.9], convention="az_colat"
        )
        mic_u = unit_sph_to_cart(
            mic_grid.azimuth, mic_grid.angle2, convention=mic_grid.convention
        )
        src_u = unit_sph_to_cart(
            src.azimuth, src.angle2, convention=src.convention
        )
        cos_gamma = mic_u @ src_u.T
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            _, h = simulate_sh_array_response(
                128, 16000.0, geom, src, max_order=8,
                array_type="directional", dir_coeff=alpha,
            )
            expected = alpha + (1.0 - alpha) * cos_gamma
            assert_allclose(h[0, :, :].real, expected, atol=1e-12)
            assert_allclose(h[0, :, :].imag, 0.0, atol=1e-12)
