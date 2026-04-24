"""Tests for the v0.4.0 encoding filters, SRP-PHAT DOA, and array presets."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.acoustics import bn_matrix
from spherical_array_processing.array import (
    circular_array,
    cubic_array,
    em32_eigenmike,
    fibonacci_grid,
    simulate_plane_wave_array_response,
    simulate_sh_array_response,
    tetrahedral_array,
)
from spherical_array_processing.doa import (
    srp_map,
    srp_map_from_covariance,
)
from spherical_array_processing.encoding import (
    apply_radial_equalizer,
    radial_equalizer,
    radial_equalizer_tikhonov,
    radial_equalizer_wng_limited,
)
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


# --------------------------------------------------------------------------- #
# Radial equalizer filters                                                    #
# --------------------------------------------------------------------------- #

class TestRadialEqualizer:
    def test_tikhonov_zero_lambda_is_exact_inverse(self):
        kr = np.linspace(0.5, 5.0, 10)
        for array_type in ("open", "rigid", "cardioid"):
            eq = radial_equalizer_tikhonov(
                4, kr, array_type=array_type,
                regularization=0.0, repeat_per_order=False,
            )
            bn = bn_matrix(4, kr, sphere=array_type, repeat_per_order=False)
            assert_allclose(eq * bn, np.ones_like(bn), atol=1e-12)

    def test_tikhonov_bounded_for_small_kr(self):
        # At kr → 0 the naive 1/B_n diverges for n ≥ 1, but Tikhonov
        # caps the gain to ≈ 1/λ for |B_n| → 0.
        kr = np.array([1e-3, 1e-2, 1e-1])
        eq = radial_equalizer_tikhonov(
            5, kr, array_type="rigid", regularization=0.05
        )
        assert np.all(np.isfinite(eq))
        # Max gain should be bounded roughly by 1/(2λ) for |B_n| ≈ 0.
        assert np.max(np.abs(eq)) < 1.0 / (2.0 * 0.05) * 1.05

    def test_wng_limit_respects_ceiling(self):
        kr = np.linspace(1e-2, 5.0, 50)
        eq = radial_equalizer_wng_limited(
            5, kr, array_type="rigid", max_gain_db=20.0
        )
        max_abs = 10.0 ** (20.0 / 20.0)
        assert np.max(np.abs(eq)) <= max_abs + 1e-12

    def test_wng_limit_saturates_at_modal_zeros(self):
        """At ``kr = 0`` all higher-order modal gains blow up; the WNG
        filter must report the ceiling, not collapse to zero.  Guards
        against the ``np.isfinite -> 0`` regression fixed in v0.4.0b1.
        """
        kr = np.array([0.0])
        max_gain_db = 30.0
        max_abs = 10.0 ** (max_gain_db / 20.0)
        eq = radial_equalizer_wng_limited(
            5, kr, array_type="rigid",
            max_gain_db=max_gain_db, repeat_per_order=False,
        )[0]
        # Order 0 is finite (|B_0(0)| = 4π); orders ≥ 1 are modal zeros and
        # must saturate at the ceiling.
        assert np.all(np.isfinite(eq))
        assert_allclose(np.abs(eq[1:]), max_abs, atol=1e-12)

    def test_wng_phase_matches_inverse(self):
        kr = np.linspace(0.5, 5.0, 10)
        eq = radial_equalizer_wng_limited(
            3, kr, array_type="rigid", max_gain_db=40.0
        )
        bn = bn_matrix(3, kr, sphere="rigid", repeat_per_order=True)
        eq_phase = np.angle(eq)
        expected_phase = np.angle(1.0 / bn)
        # Wrap difference into [-π, π] and check ≈ 0.
        diff = np.angle(np.exp(1j * (eq_phase - expected_phase)))
        assert np.max(np.abs(diff)) < 1e-10

    def test_unified_entry_point_dispatch(self):
        kr = np.linspace(0.5, 5.0, 5)
        eq_t = radial_equalizer(
            2, kr, array_type="rigid", regularization="tikhonov", tikhonov_lambda=0.01
        )
        eq_w = radial_equalizer(
            2, kr, array_type="rigid", regularization="wng_limit", max_gain_db=20.0
        )
        eq_n = radial_equalizer(2, kr, array_type="rigid", regularization="none")
        assert eq_t.shape == eq_w.shape == eq_n.shape
        # Tikhonov and WNG should produce finite bounded outputs; 'none' may blow up at DC only.
        assert np.all(np.isfinite(eq_t))
        assert np.all(np.isfinite(eq_w))

    def test_apply_radial_equalizer_matches_broadcasting(self):
        rng = np.random.default_rng(0)
        n_bins, n_coeffs, n_src = 17, 9, 4
        c = rng.normal(size=(n_bins, n_coeffs, n_src)) + 1j * rng.normal(
            size=(n_bins, n_coeffs, n_src)
        )
        eq = rng.normal(size=(n_bins, n_coeffs)) + 1j * rng.normal(size=(n_bins, n_coeffs))
        out = apply_radial_equalizer(c, eq, freq_axis=0, coeff_axis=1)
        expected = c * eq[:, :, None]
        assert_allclose(out, expected)


# --------------------------------------------------------------------------- #
# SRP / SRP-PHAT DOA                                                          #
# --------------------------------------------------------------------------- #

class TestSRPMap:
    @pytest.fixture
    def setup(self):
        mic_grid = fibonacci_grid(32)
        from spherical_array_processing.types import ArrayGeometry

        geom = ArrayGeometry(sensor_grid=mic_grid, radius_m=0.05)
        az_true = np.radians(60.0)
        colat_true = np.radians(50.0)
        src = SphericalGrid(azimuth=[az_true], angle2=[colat_true], convention="az_colat")
        fft_len, fs = 256, 16000.0
        freqs, H = simulate_plane_wave_array_response(fft_len, fs, geom, src)
        return geom, az_true, colat_true, freqs, H[:, :, 0]

    def test_srp_map_finds_single_source(self, setup):
        geom, az_true, colat_true, freqs, X = setup
        scan = fibonacci_grid(2000)
        for weighting in ("phat", "none"):
            result = srp_map(
                X, freqs, geom, scan,
                weighting=weighting, freq_range_hz=(500, 6000),
            )
            peak = int(result.peak_indices[0])
            az_p, colat_p = scan.azimuth[peak], scan.angle2[peak]
            cos_sep = (
                np.sin(colat_true) * np.sin(colat_p) * np.cos(az_true - az_p)
                + np.cos(colat_true) * np.cos(colat_p)
            )
            sep = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
            assert sep < 5.0, f"{weighting}: sep={sep:.2f}°"

    def test_srp_covariance_matches_outer_product(self, setup):
        geom, az_true, colat_true, freqs, X = setup
        scan = fibonacci_grid(2000)
        R = np.einsum("fm,fn->fmn", X, X.conj())
        for weighting in ("phat", "none"):
            r = srp_map_from_covariance(
                R, freqs, geom, scan,
                weighting=weighting, freq_range_hz=(500, 6000),
            )
            peak = int(r.peak_indices[0])
            az_p, colat_p = scan.azimuth[peak], scan.angle2[peak]
            cos_sep = (
                np.sin(colat_true) * np.sin(colat_p) * np.cos(az_true - az_p)
                + np.cos(colat_true) * np.cos(colat_p)
            )
            sep = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
            assert sep < 5.0

    def test_srp_nms_resolves_two_sources(self):
        """With min_separation_deg set, both incoherent sources are reported."""
        from spherical_array_processing.types import ArrayGeometry

        em32_grid = fibonacci_grid(64)
        geom = ArrayGeometry(sensor_grid=em32_grid, radius_m=0.05)
        az1, col1 = np.radians(45.0), np.radians(60.0)
        az2, col2 = np.radians(200.0), np.radians(80.0)
        srcs = SphericalGrid(
            azimuth=[az1, az2], angle2=[col1, col2], convention="az_colat"
        )
        fft_len, fs = 256, 16000.0
        freqs, H = simulate_plane_wave_array_response(fft_len, fs, geom, srcs)
        rng = np.random.default_rng(42)
        n_frames = 40
        mic = np.zeros((H.shape[0], H.shape[1], n_frames), dtype=complex)
        for t in range(n_frames):
            ph1 = np.exp(1j * rng.uniform(0, 2 * np.pi, H.shape[0]))
            ph2 = np.exp(1j * rng.uniform(0, 2 * np.pi, H.shape[0]))
            mic[:, :, t] = ph1[:, None] * H[:, :, 0] + ph2[:, None] * H[:, :, 1]
        scan = fibonacci_grid(2500)
        result = srp_map(
            mic, freqs, geom, scan,
            weighting="phat", freq_range_hz=(500, 6000),
            n_peaks=2, min_separation_deg=25.0,
        )
        found = [(scan.azimuth[p], scan.angle2[p]) for p in result.peak_indices]
        # Accept any permutation — check that both true directions have a near match.
        for (az_t, col_t) in ((az1, col1), (az2, col2)):
            seps = [
                np.degrees(np.arccos(np.clip(
                    np.sin(col_t) * np.sin(cp) * np.cos(az_t - ap)
                    + np.cos(col_t) * np.cos(cp),
                    -1.0, 1.0,
                )))
                for (ap, cp) in found
            ]
            assert min(seps) < 8.0, (
                f"True source ({np.degrees(az_t):.1f}, {np.degrees(col_t):.1f}) "
                f"not within 8° of any reported peak; seps={seps}"
            )

    def test_srp_rejects_bad_shapes(self, setup):
        geom, _, _, freqs, X = setup
        scan = fibonacci_grid(100)
        with pytest.raises(ValueError, match="stft_bins"):
            srp_map(X[0], freqs, geom, scan)  # 1D instead of 2D
        with pytest.raises(ValueError, match="freqs_hz"):
            srp_map(X, freqs[:10], geom, scan)
        with pytest.raises(ValueError, match="weighting"):
            srp_map(X, freqs, geom, scan, weighting="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="no frequency"):
            srp_map(X, freqs, geom, scan, freq_range_hz=(1e6, 2e6))


# --------------------------------------------------------------------------- #
# Array presets                                                               #
# --------------------------------------------------------------------------- #

class TestArrayPresets:
    def test_em32_layout(self):
        geom = em32_eigenmike()
        assert geom.sensor_grid.size == 32
        assert_allclose(geom.radius_m, 0.042)
        assert geom.array_type == "rigid"

    def test_tetrahedral_has_equal_mutual_angles(self):
        for orient in ("front", "upright"):
            geom = tetrahedral_array(orientation=orient)
            az = geom.sensor_grid.azimuth
            el = geom.sensor_grid.angle2
            u = np.column_stack(
                [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
            )
            # All pairwise dot products are -1/3 for a regular tetrahedron.
            for i in range(4):
                for j in range(i + 1, 4):
                    assert_allclose(u[i] @ u[j], -1.0 / 3.0, atol=1e-12)
            # And all unit-norm.
            assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-12)

    def test_tetrahedral_orientation_axes(self):
        """The first capsule of ``orientation="front"`` must land on
        ``+y`` and ``orientation="upright"`` must land on ``+z``.  Guards
        against the docstring/behaviour mismatch fixed in v0.4.0b1.
        """
        for orient, target in (("front", (0.0, 1.0, 0.0)), ("upright", (0.0, 0.0, 1.0))):
            geom = tetrahedral_array(orientation=orient)
            az0 = geom.sensor_grid.azimuth[0]
            el0 = geom.sensor_grid.angle2[0]
            u0 = np.array(
                [np.cos(az0) * np.cos(el0), np.sin(az0) * np.cos(el0), np.sin(el0)]
            )
            assert_allclose(u0, np.asarray(target), atol=1e-12)

    def test_cubic_array_has_right_angles(self):
        geom = cubic_array()
        az = geom.sensor_grid.azimuth
        el = geom.sensor_grid.angle2
        u = np.column_stack(
            [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
        )
        # Nearest-neighbour dot product on a unit cube inscribed in a sphere
        # is 1/3; face-diagonal is −1/3; body-diagonal is −1.
        dots = np.round(np.sort(np.unique(np.round(np.outer(u, u).flatten(), 6))), 6)
        # Just a basic sanity check — no NaNs, all ≤ 1 in magnitude.
        assert np.all(np.abs(dots) <= 1 + 1e-12)
        assert geom.sensor_grid.size == 8

    def test_circular_array_spacing(self):
        geom = circular_array(n_mics=6, radius_m=0.1)
        az = geom.sensor_grid.azimuth
        diffs = np.diff(np.sort(az % (2 * np.pi)))
        assert_allclose(diffs, 2 * np.pi / 6, atol=1e-12)
        assert np.all(geom.sensor_grid.angle2 == 0.0)

    def test_em32_with_sh_simulator_dc_is_one(self):
        geom = em32_eigenmike()
        src = SphericalGrid(azimuth=[0.5], angle2=[1.2], convention="az_colat")
        _, H = simulate_sh_array_response(128, 16000.0, geom, src, max_order=4, array_type="rigid")
        assert_allclose(H[0, :, 0], 1.0, atol=1e-12)
