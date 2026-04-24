"""Tests for frequency-dependent shoebox absorption."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import freqz

from spherical_array_processing.room import shoebox_rir_banded
from spherical_array_processing.room.banded import _fir_from_bands


def _octave_edges(fs: float) -> np.ndarray:
    return np.array([0.0, 88.0, 177.0, 355.0, 710.0, 1420.0, 2840.0, 5680.0, fs / 2.0])


class TestShoeboxBanded:
    def test_band_fir_respects_piecewise_constant_target(self):
        """A two-band [1, 0] target should stay near zero in the stopband
        rather than ramping across the entire upper band."""
        fs = 16000.0
        fir = _fir_from_bands(
            np.array([1.0, 0.0]),
            np.array([0.0, 2000.0, fs / 2.0]),
            fs,
            513,
        )
        freqs_hz, response = freqz(fir, worN=16384, fs=fs)
        passband = (freqs_hz >= 0.0) & (freqs_hz <= 1800.0)
        stopband = (freqs_hz >= 2200.0) & (freqs_hz <= 7800.0)
        assert float(np.mean(np.abs(response[passband]))) > 0.95
        assert float(np.mean(np.abs(response[stopband]))) < 0.1

    def test_output_shape_and_finite(self):
        fs = 16000.0
        refl = np.full((6, 8), 0.7)
        edges = _octave_edges(fs)
        h = shoebox_rir_banded(
            (5.0, 4.0, 3.0), (1.0, 1.0, 1.5), (4.0, 3.0, 1.5),
            refl, edges,
            fs=fs, ir_length=int(fs), max_reflection_order=6,
        )
        assert h.shape == (int(fs),)
        assert np.all(np.isfinite(h))
        assert float(np.max(np.abs(h))) > 0.0

    def test_higher_absorption_reduces_rt60(self):
        """More absorption → shorter RT60 (classical result)."""
        from spherical_array_processing.room import reverberation_time
        fs = 16000.0
        ir_len = 4 * int(fs)
        edges = _octave_edges(fs)
        # Uniform-band reflection: 0.85 vs 0.5 (scalar per band).
        refl_high = np.full((6, 8), 0.85)
        refl_low = np.full((6, 8), 0.5)
        h_high = shoebox_rir_banded(
            (6.0, 5.0, 3.0), (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            refl_high, edges,
            fs=fs, ir_length=ir_len, max_reflection_order=14,
        )
        h_low = shoebox_rir_banded(
            (6.0, 5.0, 3.0), (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            refl_low, edges,
            fs=fs, ir_length=ir_len, max_reflection_order=14,
        )
        rt_high = reverberation_time(h_high, fs, method="T20")
        rt_low = reverberation_time(h_low, fs, method="T20")
        assert rt_high > rt_low

    def test_rejects_bad_reflection_shape(self):
        with pytest.raises(ValueError, match="reflection_bands"):
            shoebox_rir_banded(
                (5.0, 4.0, 3.0), (1.0, 1.0, 1.5), (4.0, 3.0, 1.5),
                np.full((5, 8), 0.7), _octave_edges(16000.0),
                fs=16000.0, ir_length=1000,
            )

    def test_rejects_out_of_range_reflection(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            shoebox_rir_banded(
                (5.0, 4.0, 3.0), (1.0, 1.0, 1.5), (4.0, 3.0, 1.5),
                np.full((6, 8), 1.1), _octave_edges(16000.0),
                fs=16000.0, ir_length=1000,
            )

    def test_rejects_non_monotonic_bands(self):
        bad_edges = np.array([0.0, 100.0, 50.0, 500.0])
        with pytest.raises(ValueError, match="increasing"):
            shoebox_rir_banded(
                (5.0, 4.0, 3.0), (1.0, 1.0, 1.5), (4.0, 3.0, 1.5),
                np.full((6, 3), 0.7), bad_edges,
                fs=16000.0, ir_length=1000,
            )

    def test_rejects_edges_above_nyquist(self):
        with pytest.raises(ValueError, match="Nyquist"):
            shoebox_rir_banded(
                (5.0, 4.0, 3.0), (1.0, 1.0, 1.5), (4.0, 3.0, 1.5),
                np.full((6, 2), 0.7), np.array([0.0, 100.0, 20000.0]),
                fs=16000.0, ir_length=1000,
            )

    def test_rejects_out_of_room_source(self):
        with pytest.raises(ValueError, match="source"):
            shoebox_rir_banded(
                (5.0, 4.0, 3.0), (6.0, 1.0, 1.5), (4.0, 3.0, 1.5),
                np.full((6, 2), 0.7),
                np.array([0.0, 1000.0, 8000.0]),
                fs=16000.0, ir_length=1000,
            )

    def test_high_frequency_rolloff(self):
        """Reducing top-band reflection magnitude should attenuate the
        top-band late-tail energy by at least a factor of two relative
        to the same geometry with full reflection everywhere."""
        fs = 16000.0
        edges = _octave_edges(fs)
        # Base: uniform high reflection in every band.
        base = np.tile(
            np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]),
            (6, 1),
        )
        # Top-band-attenuated variant.
        atten = np.tile(
            np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.05]),
            (6, 1),
        )
        h_base = shoebox_rir_banded(
            (6.0, 5.0, 3.0), (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            base, edges,
            fs=fs, ir_length=2 * int(fs), max_reflection_order=10,
        )
        h_atten = shoebox_rir_banded(
            (6.0, 5.0, 3.0), (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            atten, edges,
            fs=fs, ir_length=2 * int(fs), max_reflection_order=10,
        )
        # Late-tail power spectrum in the top band.
        tail_base = h_base[int(0.1 * fs):]
        tail_atten = h_atten[int(0.1 * fs):]
        freqs = np.fft.rfftfreq(tail_base.size, d=1.0 / fs)
        mask = (freqs > 5800) & (freqs < 7500)
        e_base = float(np.sum(np.abs(np.fft.rfft(tail_base))[mask] ** 2))
        e_atten = float(np.sum(np.abs(np.fft.rfft(tail_atten))[mask] ** 2))
        assert e_atten < 0.5 * e_base
