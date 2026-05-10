"""Tests for recently added public API features.

Covers:
- coords: angular_distance, angular_distance_deg
- array: spatial_aliasing_frequency, max_sh_order
- acoustics: equalize_modal_coeffs (Tikhonov and softlimit)
- beamforming: steer_sh_weights, beamform_sh
- doa: estimate_sh_cov, forward_backward_cov, diagonal_loading
"""

from __future__ import annotations

import numpy as np
import pytest

from spherical_array_processing.array.sampling import (
    fibonacci_grid,
    max_sh_order,
    spatial_aliasing_frequency,
)
from spherical_array_processing.acoustics import (
    bn_matrix,
    equalize_modal_coeffs,
    sph_modal_coeffs,
)
from spherical_array_processing.beamforming import (
    axisymmetric_pattern,
    beam_weights_hypercardioid,
    beamform_sh,
    steer_sh_weights,
)
from spherical_array_processing.coords import (
    angular_distance,
    angular_distance_deg,
)
from spherical_array_processing.doa import (
    diagonal_loading,
    estimate_sh_cov,
    forward_backward_cov,
    pwd_spectrum,
)
from spherical_array_processing.types import SHBasisSpec


# ---------------------------------------------------------------------------
# coords: angular_distance
# ---------------------------------------------------------------------------

class TestAngularDistance:
    def test_zero_distance(self):
        """Same direction → 0°."""
        d = angular_distance(1.0, 0.5, 1.0, 0.5, convention="az_el")
        assert float(d) == pytest.approx(0.0, abs=1e-12)

    def test_antipodal_el(self):
        """Poles of az_el sphere → 180°."""
        d = angular_distance_deg(0.0, np.pi / 2, 0.0, -np.pi / 2)
        assert float(d) == pytest.approx(180.0, abs=1e-10)

    def test_antipodal_az(self):
        """Opposite azimuths on equator → 180°."""
        d = angular_distance_deg(0.0, 0.0, np.pi, 0.0, convention="az_el")
        assert float(d) == pytest.approx(180.0, abs=1e-10)

    def test_quarter_circle(self):
        """Equatorial to pole → 90°."""
        d = angular_distance_deg(0.0, 0.0, 0.0, np.pi / 2)
        assert float(d) == pytest.approx(90.0, abs=1e-10)

    def test_az_colat_convention(self):
        """az_colat: colat=0 (north pole) to equator (colat=π/2) → 90°."""
        d = angular_distance_deg(0.0, 0.0, 0.0, np.pi / 2, convention="az_colat")
        assert float(d) == pytest.approx(90.0, abs=1e-10)

    def test_symmetry(self):
        """d(A, B) == d(B, A)."""
        d1 = angular_distance(0.3, 0.4, 1.2, -0.2)
        d2 = angular_distance(1.2, -0.2, 0.3, 0.4)
        assert float(d1) == pytest.approx(float(d2), abs=1e-12)

    def test_bounded(self):
        """Result is always in [0, π]."""
        rng = np.random.default_rng(0)
        az1 = rng.uniform(0, 2 * np.pi, 50)
        el1 = rng.uniform(-np.pi / 2, np.pi / 2, 50)
        az2 = rng.uniform(0, 2 * np.pi, 50)
        el2 = rng.uniform(-np.pi / 2, np.pi / 2, 50)
        d = angular_distance(az1, el1, az2, el2)
        assert np.all(d >= 0.0)
        assert np.all(d <= np.pi + 1e-12)

    def test_vectorised(self):
        """Broadcasting: arrays of directions."""
        az = np.array([0.0, np.pi / 2, np.pi])
        el = np.zeros(3)
        d = angular_distance(az, el, np.zeros(3), np.zeros(3))
        assert d.shape == (3,)
        assert d[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# array: spatial_aliasing_frequency, max_sh_order
# ---------------------------------------------------------------------------

class TestAliasingUtilities:
    def test_f_alias_formula(self):
        """f_alias = N*c / (2π R)."""
        R = 0.042
        N = 4
        c = 343.0
        expected = N * c / (2.0 * np.pi * R)
        assert spatial_aliasing_frequency(R, N, c) == pytest.approx(expected, rel=1e-10)

    def test_f_alias_increases_with_order(self):
        """Higher order → higher aliasing frequency."""
        R = 0.042
        freqs = [spatial_aliasing_frequency(R, N) for N in range(1, 6)]
        assert all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1))

    def test_max_sh_order_formula(self):
        """N = floor(kR) = floor(2π f R / c)."""
        R, f, c = 0.042, 4000.0, 343.0
        kR = 2.0 * np.pi * f * R / c
        expected = int(np.floor(kR))
        assert max_sh_order(R, f, c) == expected

    def test_max_sh_order_consistency(self):
        """max_sh_order at f_alias returns N (i.e. floor(N) = N)."""
        R, N = 0.042, 3
        f = spatial_aliasing_frequency(R, N)
        # At exactly the aliasing frequency, kR = N exactly
        assert max_sh_order(R, f) == N

    def test_max_sh_order_zero(self):
        """Very low frequency → order 0."""
        assert max_sh_order(0.042, 1.0) == 0

    def test_aliasing_frequency_default_c(self):
        """Default speed of sound is 343 m/s."""
        val_default = spatial_aliasing_frequency(0.05, 2)
        val_explicit = spatial_aliasing_frequency(0.05, 2, c=343.0)
        assert val_default == pytest.approx(val_explicit, rel=1e-10)


# ---------------------------------------------------------------------------
# acoustics: equalize_modal_coeffs
# ---------------------------------------------------------------------------

class TestEqualizationModalCoeffs:
    @pytest.fixture(scope="class")
    def setup(self):
        N, K = 3, 64
        kR = np.linspace(0.1, 4.0, K)
        bn = bn_matrix(N, kR)            # (K, 16)
        rng = np.random.default_rng(7)
        sh_in = rng.standard_normal((K, 16)) + 1j * rng.standard_normal((K, 16))
        return {"N": N, "K": K, "kR": kR, "bn": bn, "sh_in": sh_in}

    def test_output_shape(self, setup):
        out = equalize_modal_coeffs(setup["sh_in"], setup["bn"])
        assert out.shape == (setup["K"], 16)

    def test_output_complex(self, setup):
        out = equalize_modal_coeffs(setup["sh_in"], setup["bn"])
        assert out.dtype == np.complex128

    def test_tikhonov_finite(self, setup):
        """Tikhonov equalization produces finite values everywhere."""
        out = equalize_modal_coeffs(setup["sh_in"], setup["bn"], reg_param=1e-4)
        assert np.all(np.isfinite(out))

    def test_softlimit_finite(self, setup):
        """Softlimit equalization produces finite values."""
        out = equalize_modal_coeffs(
            setup["sh_in"], setup["bn"], reg_param=1e-3, reg_type="softlimit"
        )
        assert np.all(np.isfinite(out))

    def test_per_order_bn_input(self, setup):
        """Accepts bn of shape (K, N+1) and expands it."""
        N, K = setup["N"], setup["K"]
        kR = setup["kR"]
        bn_per_order = sph_modal_coeffs(N, kR)   # (K, N+1)
        out = equalize_modal_coeffs(setup["sh_in"], bn_per_order, reg_param=1e-4)
        assert out.shape == (K, 16)
        assert np.all(np.isfinite(out))

    def test_large_bn_no_regularization_needed(self):
        """When |b_n| >> reg_param, equalization ≈ element-wise division."""
        K = 10
        Q = 9
        # Use a simple identity b_n
        bn = np.ones((K, Q), dtype=complex) * 2.0
        sh_in = np.ones((K, Q), dtype=complex) * 3.0
        out = equalize_modal_coeffs(sh_in, bn, reg_param=1e-10)
        # At zero reg, out ≈ sh_in / bn = 3/2 = 1.5
        assert np.allclose(np.abs(out), 1.5, atol=1e-5)

    def test_invalid_reg_type(self, setup):
        with pytest.raises(ValueError, match="reg_type"):
            equalize_modal_coeffs(setup["sh_in"], setup["bn"], reg_type="unknown")


# ---------------------------------------------------------------------------
# beamforming: steer_sh_weights, beamform_sh
# ---------------------------------------------------------------------------

class TestSHBeamSteering:
    @pytest.fixture(scope="class")
    def basis(self):
        return SHBasisSpec(max_order=3, basis="complex", angle_convention="az_colat")

    def test_steer_weight_shape(self, basis):
        b = beam_weights_hypercardioid(3)
        w = steer_sh_weights(b, 0.0, 0.0, basis)
        assert w.shape == ((3 + 1) ** 2,)

    def test_steer_weight_dtype(self, basis):
        b = beam_weights_hypercardioid(3)
        w = steer_sh_weights(b, 0.0, 0.0, basis)
        assert w.dtype == np.complex128

    def test_steer_distortionless_constraint(self, basis):
        """w^H d = 1 (distortionless constraint at look direction).

        The SH steering vector for the look direction is d = Y(Ω_0) (the
        row of the SH matrix). Weights satisfy w^H d = Σ conj(w_nm) d_nm = 1.
        """
        N = 3
        b_n = beam_weights_hypercardioid(N)
        look_az, look_colat = 0.5, 0.8
        w = steer_sh_weights(b_n, look_az, look_colat, basis)

        from spherical_array_processing.types import SphericalGrid
        from spherical_array_processing.sh import matrix as sh_matrix
        look = SphericalGrid(
            azimuth=np.array([look_az]),
            angle2=np.array([look_colat]),
            weights=np.array([4.0 * np.pi]),
            convention="az_colat",
        )
        Y_look = sh_matrix(basis, look)
        d = Y_look[0]  # steering vector at look direction, shape (Q,)
        # Distortionless constraint: w^H d = conj(w) · d = 1
        gain = float(abs(np.dot(w.conj(), d)))
        # gain = Σ_nm b_n |Y_nm|^2 = Σ_n b_n (2n+1)/(4π) = 1
        assert gain == pytest.approx(1.0, abs=1e-8)

    def test_steer_front_gain_plane_wave(self, basis):
        """Beamform a plane-wave SH signal: output ≈ 1 at look direction.

        The SH signal for a unit plane wave from Ω is conj(Y(Ω)) — the
        conjugated SH steering vector.
        """
        N = 3
        b_n = beam_weights_hypercardioid(N)
        look_az, look_colat = 0.5, 0.8
        w = steer_sh_weights(b_n, look_az, look_colat, basis)

        from spherical_array_processing.types import SphericalGrid
        from spherical_array_processing.sh import matrix as sh_matrix
        look = SphericalGrid(
            azimuth=np.array([look_az]),
            angle2=np.array([look_colat]),
            weights=np.array([4.0 * np.pi]),
            convention="az_colat",
        )
        Y_look = sh_matrix(basis, look)
        # Plane-wave SH signal from look direction = conj(Y(Ω_0))
        a = Y_look[0].conj()  # shape (Q,)
        y_out = beamform_sh(a, w)
        # y = Σ_nm Y_nm^*(Ω_0) * b_n * Y_nm(Ω_0)
        #   = Σ_n b_n Σ_m |Y_nm|^2 = Σ_n b_n (2n+1)/(4π) = 1
        assert abs(float(abs(y_out)) - 1.0) < 1e-8

    def test_order_mismatch_raises(self, basis):
        """b_n of wrong length raises ValueError."""
        b_wrong = np.ones(5)  # wrong length for N=3 (should be 4)
        with pytest.raises(ValueError, match="max_order"):
            steer_sh_weights(b_wrong, 0.0, 0.0, basis)

    def test_beamform_sh_shape(self):
        """beamform_sh contracts the last axis."""
        sh = np.random.randn(64, 9) + 0j
        w = np.ones(9, dtype=complex)
        y = beamform_sh(sh, w)
        assert y.shape == (64,)

    def test_beamform_sh_batch(self):
        """beamform_sh works on 3-D input."""
        sh = np.random.randn(8, 64, 9) + 0j
        w = np.ones(9, dtype=complex)
        y = beamform_sh(sh, w)
        assert y.shape == (8, 64)

    def test_beamform_sh_values(self):
        """Simple unit weight: output = sum over SH channels."""
        sh = np.array([[1.0, 2.0, 3.0]], dtype=complex)  # 1 row, 3 channels
        w = np.ones(3, dtype=complex)
        y = beamform_sh(sh, w)
        # y = sh @ conj(w) = 1+2+3 = 6
        assert y[0] == pytest.approx(6.0 + 0j, abs=1e-12)


# ---------------------------------------------------------------------------
# doa: estimate_sh_cov, forward_backward_cov, diagonal_loading
# ---------------------------------------------------------------------------

class TestCovarianceUtils:
    @pytest.fixture(scope="class")
    def sample_R(self):
        rng = np.random.default_rng(42)
        Q = 9
        L = 200
        snaps = rng.standard_normal((L, Q)) + 1j * rng.standard_normal((L, Q))
        return estimate_sh_cov(snaps)

    def test_estimate_sh_cov_shape(self, sample_R):
        assert sample_R.shape == (9, 9)

    def test_estimate_sh_cov_hermitian(self, sample_R):
        assert np.allclose(sample_R, sample_R.conj().T, atol=1e-14)

    def test_estimate_sh_cov_psd(self, sample_R):
        """Sample covariance is positive semi-definite."""
        evals = np.linalg.eigvalsh(sample_R)
        assert np.all(evals >= -1e-10)

    def test_estimate_sh_cov_transposed_input(self):
        """Accepts (Q, L) shaped input (transposed from default)."""
        rng = np.random.default_rng(1)
        Q, L = 9, 500
        snaps_T = rng.standard_normal((Q, L)) + 1j * rng.standard_normal((Q, L))
        R = estimate_sh_cov(snaps_T)
        assert R.shape == (Q, Q)
        assert np.allclose(R, R.conj().T)

    def test_forward_backward_hermitian(self, sample_R):
        R_fb = forward_backward_cov(sample_R)
        assert np.allclose(R_fb, R_fb.conj().T, atol=1e-12)

    def test_forward_backward_trace_preserved(self, sample_R):
        """Forward-backward averaging preserves trace."""
        R_fb = forward_backward_cov(sample_R)
        assert np.trace(R_fb).real == pytest.approx(np.trace(sample_R).real, rel=1e-10)

    def test_forward_backward_idempotent(self, sample_R):
        """Applying fb twice gives the same result as once."""
        R_fb1 = forward_backward_cov(sample_R)
        R_fb2 = forward_backward_cov(R_fb1)
        assert np.allclose(R_fb1, R_fb2, atol=1e-12)

    def test_diagonal_loading_shape(self, sample_R):
        R_dl = diagonal_loading(sample_R, 0.1)
        assert R_dl.shape == sample_R.shape

    def test_diagonal_loading_increases_diagonal(self, sample_R):
        """Loading strictly increases diagonal entries."""
        R_dl = diagonal_loading(sample_R, 0.1)
        assert np.all(R_dl.diagonal().real > sample_R.diagonal().real)

    def test_diagonal_loading_off_diagonal_unchanged(self, sample_R):
        """Off-diagonal entries are unchanged."""
        R_dl = diagonal_loading(sample_R, 0.1)
        Q = sample_R.shape[0]
        mask = ~np.eye(Q, dtype=bool)
        assert np.allclose(R_dl[mask], sample_R[mask])

    def test_diagonal_loading_absolute(self, sample_R):
        """Absolute loading adds exactly load to each diagonal entry."""
        delta = 0.5
        R_dl = diagonal_loading(sample_R, delta, relative=False)
        diff = (R_dl - sample_R).diagonal().real
        assert np.allclose(diff, delta, atol=1e-12)

    def test_diagonal_loading_hermitian(self, sample_R):
        R_dl = diagonal_loading(sample_R, 0.01)
        assert np.allclose(R_dl, R_dl.conj().T, atol=1e-14)

    def test_forward_backward_plus_loading_improves_music(self):
        """With fb averaging + diagonal loading, MUSIC peak is near the true source."""
        N = 2
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        search = fibonacci_grid(300)
        from spherical_array_processing.sh import matrix as sh_matrix
        Y = sh_matrix(spec, search)
        src_idx = 77
        y_src = Y[src_idx, :].conj()
        # Build rank-1 covariance + small noise
        R = np.outer(y_src, y_src.conj()) + 0.01 * np.eye(spec.n_coeffs)
        R_fb = forward_backward_cov(R)
        R_dl = diagonal_loading(R_fb, 1e-3)
        from spherical_array_processing.doa import music_spectrum
        result = music_spectrum(R_dl, search, spec, n_sources=1)
        found_idx = result.peak_indices[0]

        # Check angular separation < 15°
        az_t = search.azimuth[src_idx];   col_t = search.angle2[src_idx]
        az_f = search.azimuth[found_idx]; col_f = search.angle2[found_idx]
        cos_sep = (np.sin(col_t) * np.sin(col_f) * np.cos(az_t - az_f)
                   + np.cos(col_t) * np.cos(col_f))
        sep_deg = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
        assert sep_deg < 15.0, (
            f"MUSIC peak {sep_deg:.1f}° from source; expected < 15°"
        )
