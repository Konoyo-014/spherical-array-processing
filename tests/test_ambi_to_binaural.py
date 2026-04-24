"""Tests for the end-to-end SH → binaural pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.binaural import ambi_to_binaural_time_domain
from spherical_array_processing.coords import unit_sph_to_cart
from spherical_array_processing.hrtf import HRTFDataset
from spherical_array_processing.sh import matrix as sh_matrix, real_to_complex_coeffs
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


def _synthetic_hrtf(fs: float = 16000.0, n_taps: int = 256) -> HRTFDataset:
    """Toy HRTF: ±9 cm ITD + gentle x-axis head shadow."""
    grid = fibonacci_grid(200)
    u = unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention)
    ear_pos = np.array([[-0.09, 0.0, 0.0], [0.09, 0.0, 0.0]])
    tau = u @ ear_pos.T / 343.0
    shadow = np.stack(
        [
            1.0 - 0.5 * np.clip(u[:, 0], 0.0, 1.0),   # left shadowed by +x source
            1.0 - 0.5 * np.clip(-u[:, 0], 0.0, 1.0),  # right shadowed by -x source
        ],
        axis=1,
    )
    freqs = np.fft.rfftfreq(n_taps, d=1.0 / fs)
    H = shadow[None, :, :] * np.exp(
        -2j * np.pi * freqs[:, None, None] * tau[None, :, :]
    )
    hrirs = np.fft.irfft(H, n=n_taps, axis=0).transpose(1, 2, 0)
    return HRTFDataset(
        hrirs=hrirs, fs=fs, source_grid=grid, ear_positions_m=ear_pos
    )


def _plane_wave_ambi(
    azimuth_rad: float, colat_rad: float, *, max_order: int, n_samples: int,
    seed: int = 0,
):
    grid = SphericalGrid(
        azimuth=[azimuth_rad], angle2=[colat_rad], convention="az_colat",
    )
    y = np.asarray(
        sh_matrix(SHBasisSpec(max_order=max_order, basis="real"), grid)
    )[0]  # (Q,)
    env = np.random.default_rng(seed).standard_normal(n_samples) * np.hanning(n_samples) * 0.2
    return y[:, None] * env[None, :]  # (Q, T)


class TestAmbiToBinaural:
    def test_output_shape_and_dtype(self):
        ds = _synthetic_hrtf(n_taps=256)
        sig = _plane_wave_ambi(0.0, np.pi / 2, max_order=3, n_samples=4000)
        out = ambi_to_binaural_time_domain(
            sig, ds, max_order=3, f_cut_hz=1500, n_iterations=5,
        )
        assert out.shape == (2, 4000 + 256 - 1)
        assert out.dtype == np.float64

    def test_left_right_asymmetry_matches_source_direction(self):
        ds = _synthetic_hrtf()
        sig_right = _plane_wave_ambi(0.0, np.pi / 2, max_order=3, n_samples=4000)
        sig_left = _plane_wave_ambi(np.pi, np.pi / 2, max_order=3, n_samples=4000)
        out_right = ambi_to_binaural_time_domain(
            sig_right, ds, max_order=3, n_iterations=5,
        )
        out_left = ambi_to_binaural_time_domain(
            sig_left, ds, max_order=3, n_iterations=5,
        )
        er = float(np.sum(out_right ** 2, axis=1)[1])
        el = float(np.sum(out_right ** 2, axis=1)[0])
        assert er > 2.0 * el  # right ear favored for +x source
        er2 = float(np.sum(out_left ** 2, axis=1)[1])
        el2 = float(np.sum(out_left ** 2, axis=1)[0])
        assert el2 > 2.0 * er2  # left ear favored for -x source

    def test_time_major_input_accepted(self):
        ds = _synthetic_hrtf()
        sig_qt = _plane_wave_ambi(0.0, np.pi / 2, max_order=3, n_samples=4000)
        sig_tq = sig_qt.T
        out_qt = ambi_to_binaural_time_domain(
            sig_qt, ds, max_order=3, n_iterations=3,
        )
        out_tq = ambi_to_binaural_time_domain(
            sig_tq, ds, max_order=3, n_iterations=3,
        )
        np.testing.assert_allclose(out_qt, out_tq, atol=1e-12)

    def test_head_tracking_swaps_left_right(self):
        """Source at +x rotated by π about z should now excite the left ear."""
        ds = _synthetic_hrtf()
        n = 4000
        sig = _plane_wave_ambi(0.0, np.pi / 2, max_order=3, n_samples=n)
        # Constant 180° yaw rotation throughout the signal.
        static_yaw = np.array([np.pi, 0.0, 0.0])
        out = ambi_to_binaural_time_domain(
            sig, ds, max_order=3, n_iterations=5,
            head_orientations_zyz=static_yaw,
        )
        el = float(np.sum(out[0] ** 2))
        er = float(np.sum(out[1] ** 2))
        # After 180° yaw the source from +x appears at -x, so left ear
        # should dominate.
        assert el > 2.0 * er

    def test_bad_shape_raises(self):
        ds = _synthetic_hrtf()
        bad = np.zeros((7, 1000))  # 7 is not (N+1)² for N=3
        with pytest.raises(ValueError, match="max_order"):
            ambi_to_binaural_time_domain(bad, ds, max_order=3)

    def test_fft_len_override(self):
        ds = _synthetic_hrtf(n_taps=128)
        sig = _plane_wave_ambi(0.0, np.pi / 2, max_order=2, n_samples=3000)
        out = ambi_to_binaural_time_domain(
            sig, ds, max_order=2, fft_len=512, n_iterations=3,
        )
        assert out.shape == (2, 3000 + 512 - 1)

    def test_complex_basis_matches_real_basis_for_same_field(self):
        ds = _synthetic_hrtf(n_taps=128)
        sig_real = _plane_wave_ambi(0.0, np.pi / 2, max_order=3, n_samples=2000)
        sig_complex = real_to_complex_coeffs(sig_real, max_order=3, axis=0)

        out_real = ambi_to_binaural_time_domain(
            sig_real, ds, max_order=3, basis="real", n_iterations=3,
        )
        out_complex = ambi_to_binaural_time_domain(
            sig_complex, ds, max_order=3, basis="complex", n_iterations=3,
        )

        np.testing.assert_allclose(out_complex, out_real, atol=1e-10, rtol=1e-10)
