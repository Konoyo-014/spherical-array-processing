"""Tests for AmbisonicSpec and AmbisonicFrame containers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.ambi import (
    AmbisonicFrame,
    AmbisonicSignalReport,
    AmbisonicSpec,
    ambisonic_signal_report,
    channel_count,
    encode_plane_wave_frame,
    infer_order,
    order_channel_mask,
    order_channel_slices,
    per_order_energy,
)
from spherical_array_processing.types import SphericalGrid


def test_channel_count_and_order_inference():
    assert channel_count(0) == 1
    assert channel_count(3) == 16
    assert infer_order(16) == 3
    with pytest.raises(ValueError, match="channel"):
        infer_order(6)


def test_spec_validates_mixed_order_mask():
    mask = np.array([True, True, False, True])
    spec = AmbisonicSpec(max_order=1, mixed_order_mask=mask)
    assert spec.n_channels == 4
    assert spec.mixed_order_mask.tolist() == mask.tolist()
    with pytest.raises(ValueError, match="mixed_order_mask"):
        AmbisonicSpec(max_order=1, mixed_order_mask=np.ones(3, dtype=bool))


def test_order_channel_helpers_follow_acn_blocks():
    slices = order_channel_slices(3)
    assert slices == (slice(0, 1), slice(1, 4), slice(4, 9), slice(9, 16))
    mask = order_channel_mask(3, min_order=1, max_active_order=2)
    assert mask.shape == (16,)
    assert mask[:1].tolist() == [False]
    assert np.all(mask[1:9])
    assert not np.any(mask[9:])
    with pytest.raises(ValueError, match="order range"):
        order_channel_mask(2, min_order=2, max_active_order=1)


def test_frame_normalization_round_trip():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 256))
    frame = AmbisonicFrame(
        data,
        AmbisonicSpec(max_order=1, normalization="orthonormal"),
        channel_axis=0,
        sample_rate_hz=48000.0,
    )
    sn3d = frame.as_normalization("sn3d")
    back = sn3d.as_normalization("orthonormal")
    assert sn3d.spec.normalization == "sn3d"
    assert_allclose(back.data, data, atol=1e-12)


def test_frame_applies_mixed_order_mask_on_channel_axis():
    data = np.ones((10, 4))
    spec = AmbisonicSpec(
        max_order=1,
        mixed_order_mask=np.array([True, False, True, False]),
    )
    frame = AmbisonicFrame(data, spec, channel_axis=1)
    masked = frame.with_mixed_order_mask_applied()
    assert_allclose(masked.data[:, [1, 3]], 0.0, atol=0)
    assert_allclose(masked.data[:, [0, 2]], 1.0, atol=0)


def test_frequency_frame_validates_freq_axis_length():
    spec = AmbisonicSpec(max_order=1, domain="stft")
    AmbisonicFrame(np.zeros((5, 4, 3)), spec, channel_axis=1, freqs_hz=np.arange(5))
    frame = AmbisonicFrame(
        np.zeros((4, 7, 3)),
        spec,
        channel_axis=0,
        freqs_hz=np.arange(7),
        freq_axis=1,
    )
    assert frame.freq_axis == 1
    with pytest.raises(ValueError, match="freqs_hz"):
        AmbisonicFrame(np.zeros((5, 4, 3)), spec, channel_axis=1, freqs_hz=np.arange(4))
    with pytest.raises(ValueError, match="freq_axis"):
        AmbisonicFrame(np.zeros((5, 4, 3)), spec, channel_axis=1, freqs_hz=np.arange(4), freq_axis=1)


def test_encode_plane_wave_frame_uses_spec_and_mask():
    t = np.arange(64, dtype=float)
    direction = SphericalGrid(
        azimuth=np.array([0.0]),
        angle2=np.array([np.pi / 2]),
        convention="az_colat",
    )
    spec = AmbisonicSpec(
        max_order=1,
        normalization="sn3d",
        mixed_order_mask=np.array([True, False, True, True]),
    )
    frame = encode_plane_wave_frame(t, direction, spec=spec, sample_rate_hz=16000)
    assert isinstance(frame, AmbisonicFrame)
    assert frame.data.shape == (4, 64)
    assert frame.sample_rate_hz == 16000
    assert_allclose(frame.data[1], 0.0, atol=0)


def test_per_order_energy_groups_acn_channels():
    data = np.zeros((2, 9))
    data[:, 0] = 1.0
    data[:, 1:4] = 2.0
    data[:, 4:9] = 3.0
    energies = per_order_energy(data, axis=1)
    assert_allclose(energies, [2.0, 24.0, 90.0])


def test_ambisonic_signal_report_flags_health_and_orders():
    data = np.zeros((8, 4))
    data[:, 0] = 1.0
    data[:, 1] = 0.5
    report = ambisonic_signal_report(
        data,
        spec=AmbisonicSpec(max_order=1),
        axis=1,
        active_threshold_db=-40.0,
    )
    assert isinstance(report, AmbisonicSignalReport)
    assert report.max_order == 1
    assert report.n_channels == 4
    assert report.channel_axis == 1
    assert report.active_channels == 2
    assert report.has_nan is False
    assert report.has_inf is False
    assert report.peak_abs == 1.0
    assert_allclose(report.per_order_energy, [8.0, 2.0])
    assert_allclose(report.per_order_energy_fraction.sum(), 1.0)


def test_ambisonic_signal_report_ignores_nonfinite_in_numeric_summaries():
    data = np.zeros((4, 4))
    data[0, 0] = np.nan
    data[1, 1] = np.inf
    report = ambisonic_signal_report(data, axis=1)
    assert report.has_nan is True
    assert report.has_inf is True
    assert np.all(np.isfinite(report.channel_rms))
