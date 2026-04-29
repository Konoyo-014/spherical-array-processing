"""Tests for the v0.4.0b2 `spherical_array_processing.decoding` module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.array import (
    fibonacci_grid,
    get_tdesign_fallback,
)
from spherical_array_processing.decoding import (
    allrad_decoder,
    apply_decoder,
    decoder_matrix,
    epad_decoder,
    mmd_decoder,
    sad_decoder,
    vbap_gains,
)
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


# --------------------------------------------------------------------------- #
# Single-method shape / energy / orthogonality properties
# --------------------------------------------------------------------------- #


class TestDecoderProperties:
    @pytest.fixture
    def n_order(self) -> int:
        return 3

    @pytest.fixture
    def tdesign(self, n_order: int) -> SphericalGrid:
        return get_tdesign_fallback(2 * n_order + 2)

    @pytest.fixture
    def irregular_grid(self) -> SphericalGrid:
        # 16 Fibonacci points — regular-ish but not a t-design.
        return fibonacci_grid(16)

    def test_shapes(self, n_order: int, tdesign: SphericalGrid):
        q = (n_order + 1) ** 2
        for method in ("sad", "mmd", "epad", "allrad"):
            d = decoder_matrix(tdesign, n_order, method=method, basis="real")
            assert d.shape == (tdesign.size, q)

    def test_epad_is_energy_preserving(
        self, n_order: int, irregular_grid: SphericalGrid
    ):
        """EPAD's defining property: ``DᵀD = (4π / L) · I`` regardless of
        whether the speaker grid is regular."""
        d = epad_decoder(irregular_grid, n_order, basis="real")
        expected = (4.0 * np.pi / irregular_grid.size) * np.eye((n_order + 1) ** 2)
        assert_allclose(d.T @ d, expected, atol=1e-12)

    def test_sad_equals_mmd_on_tdesign(
        self, n_order: int, tdesign: SphericalGrid
    ):
        """For a sufficient-order t-design SAD and MMD coincide."""
        d_sad = sad_decoder(tdesign, n_order, basis="real")
        d_mmd = mmd_decoder(tdesign, n_order, basis="real")
        # Tolerance reflects the practical t-design orthogonality error.
        assert_allclose(d_sad, d_mmd, atol=1e-3)

    def test_sad_equals_epad_on_tdesign(
        self, n_order: int, tdesign: SphericalGrid
    ):
        d_sad = sad_decoder(tdesign, n_order, basis="real")
        d_epad = epad_decoder(tdesign, n_order, basis="real")
        assert_allclose(d_sad, d_epad, atol=1e-3)

    def test_sad_round_trip_on_tdesign(
        self, n_order: int, tdesign: SphericalGrid
    ):
        """SAD-decoded loudspeaker samples re-encoded via ``Y_spkᵀ``
        (not the quadrature-weighted variant) recover the original
        ambisonic signal.  Specifically ``Y_spkᵀ · D_sad @ b = b`` when
        the loudspeaker grid is SH-orthogonal to the required degree.
        """
        spec = SHBasisSpec(max_order=n_order, basis="real")
        y_spk = sh_matrix(spec, tdesign)
        # Grid orthogonality residual: (4π/L) Y^T Y − I.
        identity_err = np.max(
            np.abs(
                (4 * np.pi / tdesign.size) * y_spk.T @ y_spk
                - np.eye(y_spk.shape[1])
            )
        )
        d = sad_decoder(tdesign, n_order, basis="real")
        rng = np.random.default_rng(0)
        b = rng.normal(size=(n_order + 1) ** 2)
        s = d @ b
        b_back = y_spk.T @ s
        tolerance = np.linalg.norm(b) * identity_err * 2.0
        assert_allclose(b_back, b, atol=max(tolerance, 1e-10))

    def test_decoder_concentrates_energy_at_nearest_speaker(
        self, n_order: int, irregular_grid: SphericalGrid
    ):
        """All four decoders must peak at the speaker closest to the
        encoded plane-wave direction."""
        spec = SHBasisSpec(max_order=n_order, basis="real")
        az, col = np.radians(40.0), np.radians(55.0)
        src = SphericalGrid(azimuth=[az], angle2=[col], convention="az_colat")
        b = sh_matrix(spec, src)[0]
        spk_cos = (
            np.sin(col) * np.sin(irregular_grid.angle2)
            * np.cos(az - irregular_grid.azimuth)
            + np.cos(col) * np.cos(irregular_grid.angle2)
        )
        nearest = int(np.argmax(spk_cos))
        for method in ("sad", "mmd", "epad", "allrad"):
            d = decoder_matrix(irregular_grid, n_order, method=method)
            s = d @ b
            assert int(np.argmax(s)) == nearest, (
                f"{method}: peak at {int(np.argmax(s))}, want {nearest}"
            )


# --------------------------------------------------------------------------- #
# VBAP properties
# --------------------------------------------------------------------------- #


class TestVBAP:
    def test_gains_sum_of_squares_is_unit(self):
        """VBAP amplitude-normalises to ``Σ g² = 1`` per virtual source."""
        spk = fibonacci_grid(20)
        virt = fibonacci_grid(100)
        from spherical_array_processing.coords import unit_sph_to_cart

        virt_xyz = unit_sph_to_cart(
            virt.azimuth, virt.angle2, convention=virt.convention
        )
        g = vbap_gains(spk, virt_xyz)
        energy = np.sum(g ** 2, axis=0)
        assert_allclose(energy, 1.0, atol=1e-9)

    def test_gains_are_nonnegative(self):
        spk = fibonacci_grid(20)
        virt = fibonacci_grid(80)
        from spherical_array_processing.coords import unit_sph_to_cart

        g = vbap_gains(
            spk,
            unit_sph_to_cart(virt.azimuth, virt.angle2, convention=virt.convention),
        )
        assert np.all(g >= -1e-12)

    def test_speaker_direction_maps_to_itself(self):
        """A virtual source pointing exactly at speaker ``k`` collapses
        to ``g_k = 1`` with all other gains zero."""
        spk = fibonacci_grid(24)
        from spherical_array_processing.coords import unit_sph_to_cart

        spk_xyz = unit_sph_to_cart(
            spk.azimuth, spk.angle2, convention=spk.convention
        )
        g = vbap_gains(spk, spk_xyz)
        # Diagonal should be ≈ 1, off-diagonal essentially zero.
        assert_allclose(np.diag(g), 1.0, atol=1e-6)
        off_max = np.max(np.abs(g - np.diag(np.diag(g))))
        assert off_max < 1e-6


# --------------------------------------------------------------------------- #
# apply_decoder tensor handling
# --------------------------------------------------------------------------- #


class TestApplyDecoder:
    def test_apply_decoder_shape(self):
        n_order = 2
        q = (n_order + 1) ** 2
        spk = fibonacci_grid(12)
        d = decoder_matrix(spk, n_order, method="allrad")
        rng = np.random.default_rng(0)
        ambi = rng.normal(size=(64, 4, q))
        out = apply_decoder(ambi, d, coeff_axis=-1)
        assert out.shape == (64, 4, spk.size)

    def test_apply_decoder_coeff_axis(self):
        n_order = 2
        q = (n_order + 1) ** 2
        spk = fibonacci_grid(12)
        d = decoder_matrix(spk, n_order, method="allrad")
        rng = np.random.default_rng(1)
        ambi = rng.normal(size=(q, 32))
        out = apply_decoder(ambi, d, coeff_axis=0)
        assert out.shape == (spk.size, 32)

    def test_apply_decoder_rejects_mismatched_shape(self):
        spk = fibonacci_grid(10)
        d = decoder_matrix(spk, 2, method="sad")
        with pytest.raises(ValueError, match="coefficients"):
            apply_decoder(np.zeros((16, 5)), d, coeff_axis=-1)

    def test_decoder_matrix_rejects_bad_method(self):
        spk = fibonacci_grid(10)
        with pytest.raises(ValueError, match="method"):
            decoder_matrix(spk, 2, method="bogus")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Edge cases surfaced by release validation.
# --------------------------------------------------------------------------- #


class TestDegenerateLayouts:
    def test_coplanar_ring_raises_informative_error(self):
        """A pure horizontal ring cannot produce a 3-D convex hull —
        VBAP (and hence AllRAD) must reject it explicitly instead of
        leaking a raw ``scipy.spatial.QhullError``.
        """
        from spherical_array_processing.array import circular_array
        from spherical_array_processing.coords import unit_sph_to_cart

        ring_geom = circular_array(8, radius_m=0.5, elevation_rad=0.0)
        spk_grid = ring_geom.sensor_grid
        virt = fibonacci_grid(50)
        virt_xyz = unit_sph_to_cart(
            virt.azimuth, virt.angle2, convention=virt.convention
        )
        with pytest.raises(ValueError, match="3-D convex hull"):
            vbap_gains(spk_grid, virt_xyz)
        with pytest.raises(ValueError, match="3-D convex hull"):
            allrad_decoder(spk_grid, max_order=2, basis="real")

    def test_uncovered_source_warns_or_raises(self):
        """A hemispherical layout cannot cover virtual sources below
        the speaker horizon — default behaviour emits a warning,
        ``strict=True`` raises.
        """
        from spherical_array_processing.coords import unit_sph_to_cart

        # Upper-hemisphere-only layout: speakers at positive elevations only.
        az = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        el = np.full(az.size, np.radians(30.0))
        az = np.concatenate([az, np.array([0.0])])
        el = np.concatenate([el, np.array([np.radians(89.9)])])  # top cap
        spk = SphericalGrid(azimuth=az, angle2=el, convention="az_el")
        virt = fibonacci_grid(120)
        virt_xyz = unit_sph_to_cart(
            virt.azimuth, virt.angle2, convention=virt.convention
        )
        with pytest.warns(RuntimeWarning, match="outside the loudspeaker convex hull"):
            vbap_gains(spk, virt_xyz)
        with pytest.raises(ValueError, match="outside the loudspeaker convex hull"):
            vbap_gains(spk, virt_xyz, strict=True)


class TestImaginaryLoudspeakerAllRAD:
    def _upper_dome(self) -> SphericalGrid:
        az = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        el = np.radians([15.0, 45.0])
        speakers_az = np.tile(az, len(el))
        speakers_el = np.repeat(el, len(az))
        speakers_az = np.concatenate([speakers_az, [0.0]])
        speakers_el = np.concatenate([speakers_el, [np.radians(89.0)]])
        return SphericalGrid(
            azimuth=speakers_az, angle2=speakers_el, convention="az_el"
        )

    def test_plain_vbap_reports_uncovered(self):
        """Sanity: the upper dome without imaginary speakers leaves
        many virtual-source directions outside the hull."""
        from spherical_array_processing.decoding import vbap_gains
        from spherical_array_processing.coords import unit_sph_to_cart

        spk = self._upper_dome()
        virt = fibonacci_grid(120)
        virt_xyz = unit_sph_to_cart(
            virt.azimuth, virt.angle2, convention=virt.convention
        )
        with pytest.warns(RuntimeWarning, match="outside the loudspeaker"):
            vbap_gains(spk, virt_xyz)

    def test_suggest_imaginary_loudspeakers_closes_hull(self):
        """After appending the suggested imaginary speakers, VBAP must
        succeed without warnings on the dense t-design virtual grid."""
        import warnings
        from spherical_array_processing.decoding import (
            vbap_gains,
            suggest_imaginary_loudspeakers,
        )
        from spherical_array_processing.coords import unit_sph_to_cart

        spk = self._upper_dome()
        imag = suggest_imaginary_loudspeakers(spk, min_cap_half_width_deg=25.0)
        assert imag.size > 0
        virt = fibonacci_grid(120)
        virt_xyz = unit_sph_to_cart(
            virt.azimuth, virt.angle2, convention=virt.convention
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            gains = vbap_gains(spk, virt_xyz, imaginary_loudspeakers=imag)
        assert gains.shape == (spk.size, virt.size)
        # Only real loudspeakers remain in the returned matrix.

    def test_check_layout_coverage_fibonacci_is_tight(self):
        from spherical_array_processing.decoding import check_layout_coverage

        dense = fibonacci_grid(60)
        report = check_layout_coverage(dense)
        assert report["max_gap_deg"] < 25.0
        assert report["mean_gap_deg"] < 12.0
        assert report["uncovered_fraction_above_30deg"] == 0.0

    def test_check_layout_coverage_hemisphere_reports_large_gap(self):
        from spherical_array_processing.decoding import check_layout_coverage

        spk = self._upper_dome()
        report = check_layout_coverage(spk)
        # Nadir is uncovered in an upper dome.
        assert report["max_gap_deg"] > 60.0
        assert report["uncovered_fraction_above_30deg"] > 0.1

    def test_suggest_imaginary_loudspeakers_strict_raises_when_cannot_close(self):
        from spherical_array_processing.decoding import suggest_imaginary_loudspeakers

        spk = self._upper_dome()
        # Ask for an aggressive gap target that 1 imaginary speaker can't hit.
        with pytest.raises(ValueError, match="residual gap"):
            suggest_imaginary_loudspeakers(
                spk,
                min_cap_half_width_deg=10.0,
                max_imaginary=1,
                strict=True,
            )

    def test_suggest_imaginary_loudspeakers_respects_max_imaginary(self):
        from spherical_array_processing.decoding import suggest_imaginary_loudspeakers

        spk = self._upper_dome()
        imag_2 = suggest_imaginary_loudspeakers(spk, max_imaginary=2)
        imag_6 = suggest_imaginary_loudspeakers(spk, max_imaginary=6)
        assert imag_2.size <= 2
        assert imag_6.size >= imag_2.size

    def test_allrad_auto_close_hull(self):
        """``auto_close_hull=True`` must silence the "outside hull"
        warning for a hemispherical layout."""
        import warnings

        spk = self._upper_dome()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            D = allrad_decoder(spk, max_order=3, auto_close_hull=True)
        assert D.shape == (spk.size, (3 + 1) ** 2)


class TestDualBandDecoder:
    def test_max_re_weights_are_nonincreasing(self):
        from spherical_array_processing.decoding import max_re_sh_weights

        for N in (1, 3, 5, 7):
            w = max_re_sh_weights(N)
            per_order = np.asarray([w[n ** 2] for n in range(N + 1)])
            # Monotone decrease from order 0 (= 1) outwards.
            assert per_order[0] == 1.0
            assert np.all(np.diff(per_order) < 0.0)

    def test_hf_decoder_preserves_average_energy(self):
        """Under a uniform-direction prior, applying the HF decoder
        preserves expected loudspeaker energy relative to the LF
        decoder.
        """
        from spherical_array_processing.decoding import dual_band_decoder_matrix
        from spherical_array_processing.array import get_tdesign_fallback

        n_order = 3
        spk = fibonacci_grid(20)
        d_lf, d_hf = dual_band_decoder_matrix(spk, n_order, method="sad")
        test_grid = get_tdesign_fallback(2 * n_order + 2)
        y = np.asarray(
            sh_matrix(
                SHBasisSpec(
                    max_order=n_order, basis="real",
                    angle_convention=test_grid.convention,
                ),
                test_grid,
            )
        )
        e_lf = np.mean(np.sum((y @ d_lf.T) ** 2, axis=1))
        e_hf = np.mean(np.sum((y @ d_hf.T) ** 2, axis=1))
        assert_allclose(e_hf / e_lf, 1.0, atol=1e-3)

    def test_apply_dual_band_edges_match_matrices(self):
        from spherical_array_processing.decoding import (
            apply_dual_band_decoder,
            dual_band_decoder_matrix,
        )

        n_order = 2
        spk = fibonacci_grid(12)
        d_lf, d_hf = dual_band_decoder_matrix(spk, n_order, method="sad")
        rng = np.random.default_rng(0)
        f_bins = 64
        freqs = np.linspace(0.0, 8000.0, f_bins)
        ambi = rng.normal(size=(f_bins, (n_order + 1) ** 2)) + 1j * rng.normal(
            size=(f_bins, (n_order + 1) ** 2)
        )
        out = apply_dual_band_decoder(
            ambi, freqs, d_lf, d_hf, crossover_hz=700.0, crossover_order=4,
            coeff_axis=-1,
        )
        # At f=0 the crossover weight is 1 → LF only.
        assert_allclose(out[0], ambi[0] @ d_lf.T, atol=1e-10)
        # Push one bin to very-high-frequency via a very high ratio so
        # the HF weight converges to 1.  Use a bin at 100·fc.
        high_ambi = ambi[-1:]
        high_freqs = np.array([700.0 * 100.0])
        out_high = apply_dual_band_decoder(
            high_ambi, high_freqs, d_lf, d_hf, crossover_hz=700.0,
            crossover_order=4, coeff_axis=-1,
        )
        # 100·fc with an order-4 crossover leaves residual LF gain
        # ≈ 1e-4; relax the tolerance accordingly.
        assert_allclose(
            out_high[0], high_ambi[0] @ d_hf.T, atol=5e-4, rtol=5e-3
        )

    def test_apply_dual_band_rejects_mismatched_shape(self):
        from spherical_array_processing.decoding import (
            apply_dual_band_decoder,
            dual_band_decoder_matrix,
        )

        spk = fibonacci_grid(12)
        d_lf, d_hf = dual_band_decoder_matrix(spk, 2, method="sad")
        with pytest.raises(ValueError, match="freqs_hz"):
            apply_dual_band_decoder(
                np.zeros((10, 9), dtype=complex),
                np.linspace(0, 1, 5),  # wrong length
                d_lf, d_hf,
            )

    def test_apply_dual_band_rejects_freq_axis_as_coeff(self):
        from spherical_array_processing.decoding import (
            apply_dual_band_decoder,
            dual_band_decoder_matrix,
        )

        spk = fibonacci_grid(12)
        d_lf, d_hf = dual_band_decoder_matrix(spk, 2, method="sad")
        with pytest.raises(ValueError, match="frequency axis"):
            apply_dual_band_decoder(
                np.zeros((9, 10, 3), dtype=complex),
                np.linspace(0, 1, 10),
                d_lf,
                d_hf,
                coeff_axis=0,
            )

    def test_apply_dual_band_validates_crossover_params(self):
        from spherical_array_processing.decoding import (
            apply_dual_band_decoder,
            dual_band_decoder_matrix,
        )

        spk = fibonacci_grid(12)
        d_lf, d_hf = dual_band_decoder_matrix(spk, 2, method="sad")
        ambi = np.zeros((10, 9), dtype=complex)
        freqs = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="crossover_hz"):
            apply_dual_band_decoder(ambi, freqs, d_lf, d_hf, crossover_hz=0.0)
        with pytest.raises(ValueError, match="crossover_order"):
            apply_dual_band_decoder(ambi, freqs, d_lf, d_hf, crossover_order=0)


class TestRadialTypoValidation:
    def test_sph_modal_coeffs_rejects_typo(self):
        """Unknown ``array_type`` must raise instead of silently falling
        back to rigid-sphere behaviour."""
        from spherical_array_processing.acoustics import sph_modal_coeffs

        with pytest.raises(ValueError, match="array_type"):
            sph_modal_coeffs(2, np.array([1.0]), array_type="rigidd")
        with pytest.raises(ValueError, match="array_type"):
            sph_modal_coeffs(2, np.array([1.0]), array_type="direction")
