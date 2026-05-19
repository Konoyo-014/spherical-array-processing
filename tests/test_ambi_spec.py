"""Tests for AmbisonicSpec and AmbisonicFrame containers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.ambi import (
    AmbisonicFrame,
    AmbisonicSpec,
    channel_count,
    encode_plane_wave_frame,
    infer_order,
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
    with pytest.raises(ValueError, match="freqs_hz"):
        AmbisonicFrame(np.zeros((5, 4, 3)), spec, channel_axis=1, freqs_hz=np.arange(4))


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
