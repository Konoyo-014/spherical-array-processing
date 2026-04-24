"""Tests for the public measured-array encoding filters."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.acoustics.radial import bn_matrix
from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.encoding import (
    apply_measured_equalizer,
    measured_array_equalizer,
)
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec


def _make_ideal_rigid_array(
    *, max_array_order: int, n_mics: int, n_grid: int, n_fft: int, fs: float,
    radius_m: float,
):
    """Return H_array and grid_dirs for an idealised rigid sphere array.

    Useful as a deterministic reference: the measured equalizer should
    recover a near-ideal plane-wave-steering filter when handed this
    noise-free input.
    """
    mic_grid = fibonacci_grid(n_mics)
    meas = fibonacci_grid(n_grid)
    n_bins = n_fft // 2 + 1
    freqs = np.arange(n_bins) * fs / n_fft
    kr = 2 * np.pi * freqs * radius_m / 343.0
    Y_mic = np.asarray(
        sh_matrix(
            SHBasisSpec(max_order=max_array_order, basis="complex"), mic_grid,
        )
    )
    Y_grid = np.asarray(
        sh_matrix(
            SHBasisSpec(max_order=max_array_order, basis="complex"), meas,
        )
    )
    bn = bn_matrix(max_array_order, kr=kr, sphere="rigid", repeat_per_order=True)
    H = np.einsum("fk,mk,gk->fmg", bn, Y_mic, np.conj(Y_grid)) * 4 * np.pi
    dirs = np.stack(
        [meas.azimuth, np.pi / 2 - meas.angle2], axis=-1
    )  # (G, 2) az/el
    return H.astype(np.complex128), dirs, mic_grid, meas


class TestMeasuredEqualizer:
    def test_shape_and_dtype(self):
        H, dirs, _, _ = _make_ideal_rigid_array(
            max_array_order=6, n_mics=32, n_grid=128, n_fft=256, fs=48000,
            radius_m=0.042,
        )
        H_filt = measured_array_equalizer(
            H, dirs, max_order=3, n_fft=256,
        )
        assert H_filt.shape == (129, 16, 32)
        assert H_filt.dtype == np.complex128

    def test_fir_return_has_expected_length(self):
        H, dirs, _, _ = _make_ideal_rigid_array(
            max_array_order=6, n_mics=32, n_grid=128, n_fft=256, fs=48000,
            radius_m=0.042,
        )
        H_filt, h_filt = measured_array_equalizer(
            H, dirs, max_order=3, n_fft=256, return_fir=True,
        )
        assert H_filt.shape == (129, 16, 32)
        assert h_filt.shape == (256, 16, 32)
        assert np.isrealobj(h_filt)

    def test_regLS_vs_regLSHD_agree_on_clean_input(self):
        """Both methods should produce filters with ≈ the same SH
        reconstruction shape when fed a noise-free steering matrix."""
        H, dirs, _, meas = _make_ideal_rigid_array(
            max_array_order=6, n_mics=32, n_grid=128, n_fft=256, fs=48000,
            radius_m=0.042,
        )
        H_ls = measured_array_equalizer(
            H, dirs, max_order=3, n_fft=256, method="regLS",
        )
        H_hd = measured_array_equalizer(
            H, dirs, max_order=3, n_fft=256, method="regLSHD",
        )
        # Encode a plane wave at grid point g and compare SH pattern.
        g0 = 7
        Y_ref = np.asarray(
            sh_matrix(SHBasisSpec(max_order=3, basis="real"), meas)
        )  # (G, Q_r)
        expected = Y_ref[g0]
        k_mid = 32  # well-conditioned bin
        got_ls = (H_ls[k_mid] @ H[k_mid, :, g0]).real
        got_hd = (H_hd[k_mid] @ H[k_mid, :, g0]).real
        # Correlation of shapes.
        def shape_corr(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        # regLS inverts the mic matrix directly and reconstructs the
        # plane-wave SH within ~1%.  regLSHD regularises through the
        # array-SH basis first, so it trades a little shape fidelity
        # for WNG robustness — 0.97 is a realistic floor.
        assert shape_corr(got_ls, expected) > 0.99
        assert shape_corr(got_hd, expected) > 0.97

    def test_invalid_method(self):
        H, dirs, _, _ = _make_ideal_rigid_array(
            max_array_order=2, n_mics=16, n_grid=32, n_fft=128, fs=48000,
            radius_m=0.042,
        )
        with pytest.raises(ValueError, match="method"):
            measured_array_equalizer(
                H, dirs, max_order=1, n_fft=128, method="bogus",
            )


class TestApplyMeasuredEqualizer:
    def test_plane_wave_reconstruction(self):
        H, dirs, _, meas = _make_ideal_rigid_array(
            max_array_order=6, n_mics=32, n_grid=128, n_fft=256, fs=48000,
            radius_m=0.042,
        )
        H_filt = measured_array_equalizer(
            H, dirs, max_order=3, n_fft=256, amp_threshold_db=15.0,
        )
        # mic_stft at T=4 frames, each equal to the array response at
        # grid direction g0.
        g0 = 9
        mic_stft = np.broadcast_to(
            H[:, :, g0, None], H.shape[:2] + (4,)
        ).copy()
        sh_stft = apply_measured_equalizer(mic_stft, H_filt)
        assert sh_stft.shape == (129, 16, 4)
        # Reference real SH pattern at g0.
        Y_ref = np.asarray(
            sh_matrix(SHBasisSpec(max_order=3, basis="real"), meas)
        )
        expected = Y_ref[g0]
        got = sh_stft[32, :, 0].real
        corr = float(
            np.dot(got, expected)
            / (np.linalg.norm(got) * np.linalg.norm(expected))
        )
        assert corr > 0.99

    def test_shape_mismatch_raises(self):
        H, dirs, _, _ = _make_ideal_rigid_array(
            max_array_order=2, n_mics=16, n_grid=32, n_fft=128, fs=48000,
            radius_m=0.042,
        )
        H_filt = measured_array_equalizer(H, dirs, max_order=1, n_fft=128)
        # Wrong mic count in STFT.
        bad = np.zeros((H_filt.shape[0], 8, 3), dtype=np.complex128)
        with pytest.raises(ValueError, match="microphone-axis"):
            apply_measured_equalizer(bad, H_filt)
        # Wrong freq count.
        bad_f = np.zeros((10, 16, 3), dtype=np.complex128)
        with pytest.raises(ValueError, match="frequency-axis"):
            apply_measured_equalizer(bad_f, H_filt)

    def test_axis_override(self):
        """Apply filter when mic_stft is laid out as (T, F, M)."""
        H, dirs, _, _ = _make_ideal_rigid_array(
            max_array_order=3, n_mics=16, n_grid=64, n_fft=128, fs=48000,
            radius_m=0.042,
        )
        H_filt = measured_array_equalizer(H, dirs, max_order=2, n_fft=128)
        F = H_filt.shape[0]
        M = H_filt.shape[2]
        mic_stft = np.zeros((4, F, M), dtype=np.complex128)
        mic_stft[:, :, :] = H[None, :, :, 0]
        out = apply_measured_equalizer(
            mic_stft, H_filt, freq_axis=1, mic_axis=2,
        )
        assert out.shape == (4, F, H_filt.shape[1])

    def test_axis_override_with_extra_axes_and_manual_reference(self):
        rng = np.random.default_rng(0)
        F = 5
        Q = 4
        M = 3
        equalizer = rng.standard_normal((F, Q, M)) + 1j * rng.standard_normal((F, Q, M))
        mic_stft = (
            rng.standard_normal((2, M, 7, F, 6))
            + 1j * rng.standard_normal((2, M, 7, F, 6))
        )
        out = apply_measured_equalizer(
            mic_stft, equalizer, freq_axis=3, mic_axis=1,
        )
        manual = np.empty((2, Q, 7, F, 6), dtype=np.complex128)
        for a in range(2):
            for b in range(7):
                for f in range(F):
                    for c in range(6):
                        manual[a, :, b, f, c] = equalizer[f] @ mic_stft[a, :, b, f, c]
        assert out.shape == manual.shape
        assert_allclose(out, manual, atol=1e-12)

    def test_silent_input_stays_exactly_zero(self):
        H, dirs, _, _ = _make_ideal_rigid_array(
            max_array_order=3, n_mics=16, n_grid=64, n_fft=128, fs=48000,
            radius_m=0.042,
        )
        H_filt = measured_array_equalizer(H, dirs, max_order=2, n_fft=128)
        mic_stft = np.zeros((3, H_filt.shape[2], H_filt.shape[0], 4), dtype=np.complex128)
        out = apply_measured_equalizer(
            mic_stft, H_filt, freq_axis=2, mic_axis=1,
        )
        assert out.shape == (3, H_filt.shape[1], H_filt.shape[0], 4)
        assert np.all(out == 0.0)
