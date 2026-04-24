"""Tests for FOA scene translation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import correlate

from spherical_array_processing.ambi import (
    convert_ambi_normalization,
    encode_plane_wave,
    translate_foa,
)
from spherical_array_processing.types import SphericalGrid


def _plane_wave(az_deg: float, freq_hz: float = 200.0, T: int = 8192):
    fs = 48000.0
    t = np.arange(T) / fs
    sig = np.sin(2 * np.pi * freq_hz * t) * np.hanning(T) * 0.3
    grid = SphericalGrid(
        azimuth=np.array([np.radians(az_deg)]),
        angle2=np.array([np.pi / 2.0]),
        convention="az_colat",
    )
    foa = encode_plane_wave(sig, grid, max_order=1, basis="real")
    return foa, fs


class TestTranslateFoa:
    def test_zero_translation_is_identity(self):
        foa, fs = _plane_wave(45.0)
        out = translate_foa(foa, np.zeros(3), fs=fs)
        # The interior samples should match exactly (the FFT bookkeeping
        # is bit-symmetric when translation = 0).
        assert_allclose(out[:, 50:-50], foa[:, 50:-50], atol=1e-10)

    @pytest.mark.parametrize(
        "src_az_deg,r_x,sign",
        [
            (0.0, +0.10, +1),  # source at +x, move to +x → advance
            (0.0, -0.10, -1),  # source at +x, move to -x → delay
            (180.0, +0.10, -1),  # source at -x, move to +x → delay
            (180.0, -0.10, +1),  # source at -x, move to -x → advance
        ],
    )
    def test_small_translation_gives_correct_delay_sign(
        self, src_az_deg, r_x, sign,
    ):
        foa, fs = _plane_wave(src_az_deg)
        foa_t = translate_foa(foa, np.array([r_x, 0.0, 0.0]), fs=fs)
        cc = correlate(foa_t[0], foa[0], mode="full")
        lag = int(np.argmax(np.abs(cc)) - (foa.shape[1] - 1))
        # Positive lag = translated signal is delayed (arrives later).
        # Sign convention: we expect "advance" for sign=+1 → lag<0.
        if sign > 0:
            assert lag < 0
        else:
            assert lag > 0

    def test_small_translation_matches_geometric_prediction(self):
        """For ``|r|·k ≪ 1``, the FOA translation must match the
        geometric advance ``(û·r)/c`` to sub-sample precision."""
        foa, fs = _plane_wave(0.0, freq_hz=200.0)  # source at +x
        r_x = 0.1
        foa_t = translate_foa(foa, np.array([r_x, 0.0, 0.0]), fs=fs)
        cc = correlate(foa_t[0], foa[0], mode="full")
        lag = int(np.argmax(np.abs(cc)) - (foa.shape[1] - 1))
        measured_advance_s = -lag / fs
        expected_advance_s = r_x / 343.0
        # Sample resolution at 48 kHz is 21 µs → allow 1 sample error.
        assert abs(measured_advance_s - expected_advance_s) < 1.5 / fs

    def test_transverse_translation_has_modest_effect_on_colinear_source(self):
        """Moving in +y while source is at +x keeps the waveform mostly
        intact — the exact geometric shift would be zero (``û·r = 0``),
        but the first-order PWD point-spread function introduces a
        bounded amount of smearing (< ~15% RMS for 10 cm translation)."""
        foa, fs = _plane_wave(0.0)
        foa_t = translate_foa(foa, np.array([0.0, 0.1, 0.0]), fs=fs)
        rms_ratio = (
            float(np.sqrt(np.mean((foa_t - foa)[:, 200:-200] ** 2)))
            / float(np.sqrt(np.mean(foa[:, 200:-200] ** 2)))
        )
        assert rms_ratio < 0.15

    def test_normalization_round_trip(self):
        """Translating an SN3D signal should still return SN3D."""
        foa, fs = _plane_wave(30.0)
        foa_sn3d = convert_ambi_normalization(
            foa, max_order=1,
            from_="orthonormal", to="sn3d", axis=0,
        )
        out_sn3d = translate_foa(
            foa_sn3d, np.array([0.05, 0.0, 0.0]),
            fs=fs, normalization="sn3d",
        )
        out_ortho = translate_foa(
            foa, np.array([0.05, 0.0, 0.0]),
            fs=fs, normalization="orthonormal",
        )
        expected_ortho_from_sn3d = convert_ambi_normalization(
            out_sn3d, max_order=1,
            from_="sn3d", to="orthonormal", axis=0,
        )
        assert_allclose(
            expected_ortho_from_sn3d, out_ortho, atol=1e-8,
        )

    def test_channels_last_layout(self):
        foa, fs = _plane_wave(30.0)
        out_cf = translate_foa(
            foa, np.array([0.05, 0.0, 0.0]), fs=fs,
            axis="channels_first",
        )
        out_cl = translate_foa(
            foa.T, np.array([0.05, 0.0, 0.0]), fs=fs,
            axis="channels_last",
        )
        assert out_cl.shape == (foa.shape[1], 4)
        assert_allclose(out_cl, out_cf.T, atol=1e-10)

    def test_rejects_bad_translation_shape(self):
        foa, fs = _plane_wave(0.0)
        with pytest.raises(ValueError, match="translation_m"):
            translate_foa(foa, np.zeros(2), fs=fs)

    def test_rejects_bad_fs(self):
        foa, _ = _plane_wave(0.0)
        with pytest.raises(ValueError, match="fs"):
            translate_foa(foa, np.zeros(3), fs=0.0)

    def test_rejects_low_decomposition_dirs(self):
        foa, fs = _plane_wave(0.0)
        with pytest.raises(ValueError, match="decomposition"):
            translate_foa(
                foa, np.zeros(3), fs=fs, n_decomposition_dirs=3,
            )

    def test_rejects_wrong_shape_input(self):
        with pytest.raises(ValueError, match="2-D"):
            translate_foa(np.zeros((4, 10, 2)), np.zeros(3), fs=48000.0)
        with pytest.raises(ValueError, match="channels_first"):
            translate_foa(np.zeros((3, 100)), np.zeros(3), fs=48000.0)
