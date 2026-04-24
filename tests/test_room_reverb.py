"""Tests for `spherical_array_processing.room.reverb`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import convolve

from spherical_array_processing.room import (
    ShoeboxRoom,
    convolve_mono_to_ambi,
    convolve_sh_to_sh,
    shoebox_sh_rir,
)


class TestConvolveMonoToAmbi:
    def test_single_channel_matches_scipy(self):
        rng = np.random.default_rng(0)
        T = 512
        rir = rng.standard_normal((4, 128))
        dry = rng.standard_normal(T) * 0.1
        out = convolve_mono_to_ambi(dry, rir)
        assert out.shape == (4, T + 128 - 1)
        for q in range(4):
            expected = convolve(dry, rir[q], mode="full")
            assert_allclose(out[q], expected, atol=1e-10)

    def test_batched_dry_signal(self):
        rng = np.random.default_rng(1)
        T = 256
        rir = rng.standard_normal((4, 64))
        dry = rng.standard_normal((3, T)) * 0.1
        out = convolve_mono_to_ambi(dry, rir)
        assert out.shape == (3, 4, T + 64 - 1)
        for b in range(3):
            for q in range(4):
                expected = convolve(dry[b], rir[q], mode="full")
                assert_allclose(out[b, q], expected, atol=1e-10)

    def test_integrates_with_shoebox_sh_rir(self):
        rng = np.random.default_rng(2)
        room = ShoeboxRoom(dimensions_m=(4.0, 4.0, 3.0), reflection=0.5)
        rir = shoebox_sh_rir(
            room, (1.0, 1.0, 1.5), (3.0, 3.0, 1.5),
            fs=16000.0, ir_length=2048, max_order=2, max_reflection_order=6,
        )
        dry = rng.standard_normal(8000) * 0.1
        wet = convolve_mono_to_ambi(dry, rir)
        assert wet.shape == (9, 8000 + 2048 - 1)
        # W-channel energy must be non-zero (direct path + reflections).
        assert float(np.sum(wet[0] ** 2)) > 0

    def test_rejects_non_2d_rir(self):
        with pytest.raises(ValueError, match="sh_rir"):
            convolve_mono_to_ambi(np.zeros(100), np.zeros((4, 64, 1)))

    def test_axis_zero_for_1d_signal_keeps_q_before_time(self):
        rng = np.random.default_rng(3)
        dry = rng.standard_normal(128) * 0.1
        rir = rng.standard_normal((4, 32))
        out = convolve_mono_to_ambi(dry, rir, axis=0)
        assert out.shape == (4, 128 + 32 - 1)
        for q in range(4):
            expected = convolve(dry, rir[q], mode="full")
            assert_allclose(out[q], expected, atol=1e-10)

    def test_non_last_time_axis_inserts_sh_axis_before_time(self):
        rng = np.random.default_rng(4)
        dry = rng.standard_normal((2, 64, 3)) * 0.1
        rir = rng.standard_normal((4, 16))
        out = convolve_mono_to_ambi(dry, rir, axis=1)
        assert out.shape == (2, 4, 64 + 16 - 1, 3)
        for batch in range(2):
            for q in range(4):
                for tail in range(3):
                    expected = convolve(dry[batch, :, tail], rir[q], mode="full")
                    assert_allclose(out[batch, q, :, tail], expected, atol=1e-10)


class TestConvolveShToSh:
    def test_channel_diagonal_match(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((4, 500))
        rir = rng.standard_normal((4, 80))
        out = convolve_sh_to_sh(sig, rir)
        assert out.shape == (4, 500 + 80 - 1)
        for q in range(4):
            expected = convolve(sig[q], rir[q], mode="full")
            assert_allclose(out[q], expected, atol=1e-10)

    def test_axis_override(self):
        rng = np.random.default_rng(1)
        sig_tq = rng.standard_normal((500, 4))  # (T, Q)
        rir = rng.standard_normal((4, 80))
        out = convolve_sh_to_sh(
            sig_tq, rir,
            signal_axis_channels=1, signal_axis_time=0,
        )
        assert out.shape == (579, 4)
        for q in range(4):
            expected = convolve(sig_tq[:, q], rir[q], mode="full")
            assert_allclose(out[:, q], expected, atol=1e-10)

    def test_batched_signal_with_non_default_axes(self):
        rng = np.random.default_rng(2)
        sig = rng.standard_normal((2, 500, 4))
        rir = rng.standard_normal((4, 80))
        out = convolve_sh_to_sh(
            sig, rir,
            signal_axis_channels=2, signal_axis_time=1,
        )
        assert out.shape == (2, 579, 4)
        for batch in range(2):
            for q in range(4):
                expected = convolve(sig[batch, :, q], rir[q], mode="full")
                assert_allclose(out[batch, :, q], expected, atol=1e-10)

    def test_channel_count_mismatch(self):
        with pytest.raises(ValueError, match="channel-count"):
            convolve_sh_to_sh(np.zeros((4, 100)), np.zeros((9, 32)))

    def test_rejects_non_2d_rir(self):
        with pytest.raises(ValueError, match="sh_rir"):
            convolve_sh_to_sh(np.zeros((4, 100)), np.zeros((4,)))

    @pytest.mark.parametrize("sig", [np.array(1.0), np.zeros(100)])
    def test_rejects_signal_with_fewer_than_two_axes(self, sig):
        with pytest.raises(ValueError, match="at least two axes"):
            convolve_sh_to_sh(sig, np.zeros((1, 32)))
