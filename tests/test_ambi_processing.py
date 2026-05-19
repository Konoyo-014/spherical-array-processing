"""Tests for Ambisonic signal processing helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing import ambi


def test_order_truncate_pad_and_change():
    data = np.arange(18).reshape(2, 9)
    assert ambi.ensure_channel_axis(data, axis=1)[1] == 2
    trunc = ambi.truncate_order(data, 1, axis=1)
    assert trunc.shape == (2, 4)
    padded = ambi.pad_order(trunc, 2, axis=1)
    assert padded.shape == data.shape
    assert_allclose(padded[:, :4], trunc)
    assert_allclose(padded[:, 4:], 0.0)
    assert_allclose(ambi.change_order(data, 1, axis=1), trunc)
    assert_allclose(ambi.change_order(trunc, 2, axis=1), padded)


def test_level_normalization_and_w_channel():
    data = np.array([[2.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    assert ambi.ambi_peak(data) == 2.0
    assert_allclose(ambi.normalize_peak(data, target_peak=1.0).max(), 1.0)
    assert_allclose(ambi.ambi_rms(ambi.normalize_rms(data, target_rms=2.0)), 2.0)
    assert_allclose(ambi.w_channel(data, axis=1), [2.0, 1.0])
    assert_allclose(ambi.mono_from_w(data, axis=1, w_gain=0.5), [1.0, 0.5])


def test_channel_and_order_gains():
    data = np.ones((3, 4))
    out = ambi.apply_channel_gains(data, [1.0, 2.0, 3.0, 4.0], axis=1)
    assert_allclose(out[0], [1.0, 2.0, 3.0, 4.0])
    gains = ambi.per_order_gains(1, [10.0, 2.0])
    assert_allclose(gains, [10.0, 2.0, 2.0, 2.0])
    out_order = ambi.apply_per_order_gains(data, [10.0, 2.0], axis=1)
    assert_allclose(out_order[0], gains)
    assert_allclose(ambi.order_rms(out_order, axis=1), [10.0, 2.0])
    assert_allclose(ambi.order_balance_db(out_order, axis=1), [0.0, -13.9794000867])


def test_covariance_correlation_mix_and_fade():
    data = np.eye(4)
    cov = ambi.channel_covariance(data, axis=1)
    assert cov.shape == (4, 4)
    corr = ambi.channel_correlation(data, axis=1)
    assert_allclose(np.diag(corr), np.ones(4))
    frames = np.stack([np.zeros((2, 4)), np.ones((2, 4))])
    assert_allclose(ambi.mix_ambi_frames(frames), 0.5)
    assert_allclose(ambi.mix_ambi_frames(frames, weights=[0.25, 0.75]), 0.75)
    assert_allclose(ambi.fade_ambi_frames(np.zeros(4), np.ones(4), 0.25), 0.25)


def test_processing_validation():
    with pytest.raises(ValueError, match="target_order"):
        ambi.truncate_order(np.zeros(4), 2)
    with pytest.raises(ValueError, match="gains length"):
        ambi.apply_channel_gains(np.zeros(4), [1.0, 2.0])
    with pytest.raises(ValueError, match="alpha"):
        ambi.fade_ambi_frames(np.zeros(4), np.ones(4), 2.0)
