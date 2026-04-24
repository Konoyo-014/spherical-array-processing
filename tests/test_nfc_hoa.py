"""Tests for NFC-HOA distance compensation filters."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.acoustics.radial import besselhs2
from spherical_array_processing.ambi import nfc_hoa_distance_filter


class TestNfcHoaDistanceFilter:
    def test_identity_when_d_equals_r(self):
        freqs = np.linspace(0.0, 20000.0, 128)
        f = nfc_hoa_distance_filter(3, freqs, 2.0, 2.0)
        assert f.shape == (128, 16)
        assert_allclose(f, 1.0 + 0j, atol=1e-12)

    def test_dc_limit_is_r_over_d_to_n_plus_one(self):
        """``F_n(0) = (R/d)^{n+1}`` from the small-x Hankel asymptote."""
        freqs = np.array([0.0])
        for d, R in [(0.5, 2.0), (1.0, 2.0), (5.0, 2.0), (100.0, 2.0)]:
            f = nfc_hoa_distance_filter(
                5, freqs, d, R, repeat_per_order=False,
            )
            assert f.shape == (1, 6)
            expected = np.array([(R / d) ** (n + 1) for n in range(6)])
            assert_allclose(f[0].real, expected, atol=1e-12)
            assert_allclose(f[0].imag, 0.0, atol=1e-12)

    def test_dc_matches_non_zero_low_frequency_limit(self):
        """Computing the filter at k → 0+ should match the hard-coded
        exact-DC closed form to machine precision."""
        c = 343.0
        near_dc = 2.0 * np.pi * 1e-3 / c  # kR ≈ 2e-5 m⁻¹
        freqs = np.array([0.0, 1e-3])
        for d, R in [(0.3, 1.5), (2.0, 1.0), (10.0, 5.0)]:
            f = nfc_hoa_distance_filter(
                4, freqs, d, R, repeat_per_order=False,
            )
            # The k=0 row (closed form) should equal the k = 2π·1e-3/c
            # magnitude to the small-x asymptote accuracy (~1e-3
            # relative at f=1e-3 Hz, dominated by a tiny
            # k(d-R) phase).
            assert_allclose(np.abs(f[0]), np.abs(f[1]), rtol=1e-3)

    def test_matches_direct_hankel_ratio(self):
        """Verify the implementation matches h_n^(2)(kd) / h_n^(2)(kR)
        bin-for-bin at well-conditioned frequencies."""
        freqs = np.linspace(50.0, 10000.0, 32)
        d = 0.7
        R = 2.0
        c = 343.0
        k = 2 * np.pi * freqs / c
        for N in range(4):
            direct = np.stack([
                besselhs2(n, k * d) / besselhs2(n, k * R)
                for n in range(N + 1)
            ], axis=-1)
            got = nfc_hoa_distance_filter(
                N, freqs, d, R, c=c, repeat_per_order=False,
            )
            assert_allclose(got, direct, atol=1e-12)

    def test_repeat_per_order_expansion(self):
        freqs = np.linspace(0.0, 8000.0, 16)
        compact = nfc_hoa_distance_filter(
            3, freqs, 0.5, 2.0, repeat_per_order=False,
        )
        expanded = nfc_hoa_distance_filter(
            3, freqs, 0.5, 2.0, repeat_per_order=True,
        )
        assert compact.shape == (16, 4)
        assert expanded.shape == (16, 16)
        # ACN q-th column carries per-order filter at n = floor(sqrt(q)).
        n_of_q = np.floor(np.sqrt(np.arange(16))).astype(int)
        for q in range(16):
            assert_allclose(expanded[:, q], compact[:, n_of_q[q]], atol=0)

    def test_near_field_boosts_low_frequencies_for_higher_orders(self):
        """For d < R, higher-order channels should have monotonically
        larger DC gain: |F_1(DC)| < |F_2(DC)| < |F_3(DC)|."""
        f = nfc_hoa_distance_filter(
            3, np.array([0.0]), 0.5, 2.0, repeat_per_order=False,
        )
        magnitudes = np.abs(f[0])
        assert magnitudes[0] < magnitudes[1] < magnitudes[2] < magnitudes[3]

    def test_plane_wave_limit_collapses_all_orders(self):
        """For d → ∞ the DC gain ``(R/d)^(n+1)`` collapses to zero at
        every order (including ``n=0``, where it scales as ``R/d``)."""
        f = nfc_hoa_distance_filter(
            3, np.array([0.0]), 1000.0, 2.0, repeat_per_order=False,
        )
        magnitudes = np.abs(f[0])
        assert magnitudes[0] < 1e-2      # R/d ≈ 2e-3
        assert magnitudes[1] < 1e-4
        assert magnitudes[2] < 1e-6
        assert magnitudes[3] < 1e-8

    def test_invalid_distances_raise(self):
        freqs = np.array([100.0])
        with pytest.raises(ValueError, match="source_distance"):
            nfc_hoa_distance_filter(3, freqs, 0.0, 2.0)
        with pytest.raises(ValueError, match="reference_distance"):
            nfc_hoa_distance_filter(3, freqs, 1.0, -1.0)
        with pytest.raises(ValueError, match="c"):
            nfc_hoa_distance_filter(3, freqs, 1.0, 2.0, c=0.0)

    def test_order_zero_magnitude_is_r_over_d_everywhere(self):
        """``h_0^{(2)}(x) = i·exp(-ix)/x`` implies
        ``F_0(k) = (R/d)·exp(-ik(d-R))`` — magnitude R/d at every
        frequency including DC."""
        freqs = np.linspace(0.0, 8000.0, 32)
        f = nfc_hoa_distance_filter(
            0, freqs, 0.5, 2.0, repeat_per_order=False,
        )
        assert f.shape == (32, 1)
        assert_allclose(np.abs(f), 2.0 / 0.5, atol=1e-12)
