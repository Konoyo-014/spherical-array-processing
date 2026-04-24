"""Tests for UHJ-2 stereo ↔ B-format codec (Gerzon)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.ambi import (
    convert_ambi_normalization,
    encode_plane_wave,
    uhj_decode,
    uhj_encode,
)
from spherical_array_processing.types import SphericalGrid


def _plane_wave_foa(az_rad: float, T: int = 2048, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(T) * np.hanning(T) * 0.25
    grid = SphericalGrid(
        azimuth=np.array([float(az_rad)]),
        angle2=np.array([np.pi / 2.0]),
        convention="az_colat",
    )
    return encode_plane_wave(
        src, grid, max_order=1, basis="real", normalization="orthonormal",
    )


class TestUhjEncode:
    def test_shape(self):
        foa = _plane_wave_foa(0.0)
        stereo = uhj_encode(foa)
        assert stereo.shape == (2, foa.shape[1])

    def test_left_source_favors_left_channel(self):
        foa = _plane_wave_foa(np.pi / 4)  # front-left
        stereo = uhj_encode(foa)
        L_rms = float(np.sqrt(np.mean(stereo[0] ** 2)))
        R_rms = float(np.sqrt(np.mean(stereo[1] ** 2)))
        assert L_rms > 1.5 * R_rms

    def test_right_source_favors_right_channel(self):
        foa = _plane_wave_foa(-np.pi / 4)  # front-right
        stereo = uhj_encode(foa)
        L_rms = float(np.sqrt(np.mean(stereo[0] ** 2)))
        R_rms = float(np.sqrt(np.mean(stereo[1] ** 2)))
        assert R_rms > 1.5 * L_rms

    def test_center_source_is_balanced(self):
        """A source at azimuth 0 (front) should give equal L and R RMS."""
        foa = _plane_wave_foa(0.0)
        stereo = uhj_encode(foa)
        L_rms = float(np.sqrt(np.mean(stereo[0] ** 2)))
        R_rms = float(np.sqrt(np.mean(stereo[1] ** 2)))
        assert abs(L_rms - R_rms) < 0.02 * max(L_rms, R_rms)

    def test_y_only_uses_classical_half_gain(self):
        T = 128
        y = np.linspace(-1.0, 1.0, T)
        foa = np.zeros((4, T))
        foa[1] = y  # ACN Y, already identical to FuMa scaling in SN3D.
        stereo = uhj_encode(foa, normalization="sn3d")
        expected = 0.5 * 0.6554516 * y
        assert_allclose(stereo[0], expected, atol=1e-12)
        assert_allclose(stereo[1], -expected, atol=1e-12)

    def test_matches_classical_reference_formula_on_tone(self):
        T = 4096
        k = 17
        phase = 2 * np.pi * k * np.arange(T) / T
        carrier_cos = np.cos(phase)
        carrier_sin = np.sin(phase)

        w = 0.7 * carrier_cos
        x = -0.2 * carrier_cos
        y = 0.4 * carrier_cos
        foa = np.zeros((4, T))
        foa[0] = w
        foa[1] = y
        foa[3] = x

        stereo = uhj_encode(foa, normalization="sn3d")
        w_fuma = w / np.sqrt(2.0)
        s = 0.9396926 * w_fuma + 0.1855740 * x
        quadrature_amp = -0.3420201 * (0.7 / np.sqrt(2.0)) + 0.5098604 * (-0.2)
        d = quadrature_amp * carrier_sin + 0.6554516 * y
        expected_l = 0.5 * (s + d)
        expected_r = 0.5 * (s - d)
        assert_allclose(stereo[0], expected_l, atol=2e-4)
        assert_allclose(stereo[1], expected_r, atol=2e-4)

    def test_channels_last_layout(self):
        foa = _plane_wave_foa(np.pi / 3)
        stereo_cf = uhj_encode(foa, axis="channels_first")
        stereo_cl = uhj_encode(foa.T, axis="channels_last")
        assert stereo_cl.shape == (foa.shape[1], 2)
        assert_allclose(stereo_cl, stereo_cf.T, atol=1e-12)

    def test_normalization_invariance(self):
        """UHJ output should be the same whether you feed orthonormal,
        n3d, or sn3d as long as you declare it correctly."""
        foa_ortho = _plane_wave_foa(np.pi / 5)
        foa_sn3d = convert_ambi_normalization(
            foa_ortho, max_order=1,
            from_="orthonormal", to="sn3d", axis=0,
        )
        foa_n3d = convert_ambi_normalization(
            foa_ortho, max_order=1,
            from_="orthonormal", to="n3d", axis=0,
        )
        out_ortho = uhj_encode(foa_ortho, normalization="orthonormal")
        out_sn3d = uhj_encode(foa_sn3d, normalization="sn3d")
        out_n3d = uhj_encode(foa_n3d, normalization="n3d")
        assert_allclose(out_sn3d, out_ortho, atol=1e-10)
        assert_allclose(out_n3d, out_ortho, atol=1e-10)

    def test_fir_hilbert_matches_fft_at_interior(self):
        """The FIR-based Hilbert and FFT-based one should agree on
        interior samples away from the edges."""
        foa = _plane_wave_foa(np.pi / 5)
        stereo_fft = uhj_encode(foa, hilbert_method="fft")
        stereo_fir = uhj_encode(foa, hilbert_method="fir", fir_taps=513)
        margin = 300
        diff = np.max(
            np.abs(
                stereo_fft[:, margin:-margin]
                - stereo_fir[:, margin:-margin]
            )
        )
        assert diff < 5e-3

    def test_fir_hilbert_output_shape_matches_input(self):
        foa = _plane_wave_foa(0.0, T=3333)
        stereo = uhj_encode(foa, hilbert_method="fir", fir_taps=129)
        assert stereo.shape == (2, 3333)

    def test_fir_hilbert_short_signal_still_matches_input_length(self):
        foa = _plane_wave_foa(0.0, T=64)
        stereo = uhj_encode(foa, hilbert_method="fir", fir_taps=129)
        decoded = uhj_decode(stereo, hilbert_method="fir", fir_taps=129)
        assert stereo.shape == (2, 64)
        assert decoded.shape == (4, 64)

    def test_fir_method_rejects_tiny_taps(self):
        foa = _plane_wave_foa(0.0, T=256)
        with pytest.raises(ValueError, match="numtaps"):
            uhj_encode(foa, hilbert_method="fir", fir_taps=1)

    def test_invalid_hilbert_method_raises(self):
        foa = _plane_wave_foa(0.0, T=128)
        with pytest.raises(ValueError, match="hilbert_method"):
            uhj_encode(foa, hilbert_method="bogus")

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            uhj_encode(np.zeros((4, 2, 3)))

    def test_wrong_channel_count_raises(self):
        with pytest.raises(ValueError, match="channels_first"):
            uhj_encode(np.zeros((3, 100)))


class TestUhjDecode:
    def test_shape(self):
        stereo = np.zeros((2, 1024))
        foa = uhj_decode(stereo)
        assert foa.shape == (4, 1024)

    def test_z_channel_is_zero(self):
        rng = np.random.default_rng(0)
        stereo = rng.standard_normal((2, 512)) * 0.1
        foa = uhj_decode(stereo)
        assert_allclose(foa[2], 0.0, atol=1e-14)

    def test_encode_decode_preserves_direction_sign(self):
        """A right-biased UHJ stereo should decode to an FOA whose Y
        channel is negative at steady-state."""
        foa_in = _plane_wave_foa(-np.pi / 4)  # right
        stereo = uhj_encode(foa_in)
        foa_out = uhj_decode(stereo)
        # Y channel integral should have the same sign as Y of input.
        y_in = float(np.sum(foa_in[1]))
        y_out = float(np.sum(foa_out[1]))
        assert np.sign(y_in) == np.sign(y_out) or abs(y_in) < 1e-6

    def test_channels_last_layout(self):
        rng = np.random.default_rng(5)
        stereo = rng.standard_normal((2, 800)) * 0.1
        foa_cf = uhj_decode(stereo, axis="channels_first")
        foa_cl = uhj_decode(stereo.T, axis="channels_last")
        assert foa_cl.shape == (800, 4)
        assert_allclose(foa_cl, foa_cf.T, atol=1e-12)

    def test_bad_channel_count_raises(self):
        with pytest.raises(ValueError, match="channels_first"):
            uhj_decode(np.zeros((3, 100)))
