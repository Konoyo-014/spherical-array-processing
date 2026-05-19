"""Tests for measured-array transfer-function helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spherical_array_processing as sap


def _measurement():
    freqs = np.array([100.0, 200.0, 400.0])
    transfer = np.array(
        [
            [[1.0 + 0.0j, 0.5 + 0.0j], [0.8 + 0.1j, 0.3 + 0.2j]],
            [[1.0 + 0.0j, 0.4 + 0.1j], [0.7 + 0.2j, 0.2 + 0.3j]],
            [[1.0 + 0.0j, 0.3 + 0.2j], [0.6 + 0.3j, 0.1 + 0.4j]],
        ],
        dtype=np.complex128,
    )
    grid = sap.array.circular_array(2, radius_m=0.05).sensor_grid
    return sap.measurement.ArrayMeasurement(
        frequencies_hz=freqs,
        transfer=transfer,
        array=sap.array.circular_array(2, radius_m=0.05),
        source_grid=grid,
        sample_rate_hz=48000.0,
    )


def test_array_measurement_container_validates_shape():
    m = _measurement()
    assert m.n_frequencies == 3
    assert m.n_sensors == 2
    assert m.n_sources == 2
    assert m.source_slice(1).shape == (3, 2)
    copied = m.with_transfer(m.transfer)
    assert copied.transfer.shape == m.transfer.shape
    with pytest.raises(ValueError, match="source_index"):
        m.source_slice(9)


def test_transfer_magnitude_phase_and_group_delay():
    m = _measurement()
    mag = sap.measurement.transfer_magnitude_db(m.transfer)
    assert mag.shape == m.transfer.shape
    assert_allclose(mag[:, 0, 0], 0.0, atol=1e-12)
    phase = sap.measurement.transfer_phase_rad(m.transfer)
    assert phase.shape == m.transfer.shape
    pure_delay = np.exp(-1j * 2 * np.pi * m.frequencies_hz[:, None] * 0.002)
    gd = sap.measurement.transfer_group_delay_s(pure_delay, m.frequencies_hz)
    assert_allclose(gd[:, 0], 0.002, atol=1e-12)


def test_normalization_and_capsule_mismatch():
    m = _measurement()
    norm = sap.measurement.reference_channel_equalization(m.transfer)
    assert_allclose(norm[:, 0, :], np.ones((3, 2)), atol=1e-12)
    mean = sap.measurement.sensor_mean_magnitude(m.transfer)
    rms = sap.measurement.sensor_rms_magnitude(m.transfer)
    assert mean.shape == (2,)
    assert rms.shape == (2,)
    gain = sap.measurement.capsule_gain_mismatch_db(m.transfer)
    phase = sap.measurement.capsule_phase_mismatch_rad(m.transfer)
    assert gain.shape == (2,)
    assert phase.shape == (2,)


def test_condition_rank_and_inverse_bank():
    m = _measurement()
    cond = sap.measurement.steering_condition_numbers(m.transfer)
    ranks = sap.measurement.steering_ranks(m.transfer)
    assert cond.shape == (3,)
    assert ranks.tolist() == [2, 2, 2]
    inv = sap.measurement.tikhonov_inverse_bank(m.transfer, regularization=1e-8)
    assert inv.shape == (3, 2, 2)
    eye_like = np.einsum("fqm,fms->fqs", inv, m.transfer)
    assert_allclose(eye_like, np.broadcast_to(np.eye(2), (3, 2, 2)), atol=1e-5)


def test_regularization_sweep_and_diagnostics():
    m = _measurement()
    targets = m.transfer @ np.array([1.0, 0.0])
    errors = sap.measurement.regularization_sweep_error(
        m.transfer,
        targets,
        regularizations=[0.0, 1e-3],
    )
    assert errors.shape == (2,)
    assert errors[0] <= errors[1] + 1e-12
    best = sap.measurement.best_regularization(
        m.transfer,
        targets,
        regularizations=[0.0, 1e-3],
    )
    assert best == 0.0
    diag = sap.measurement.measurement_diagnostics(m)
    assert isinstance(diag, sap.measurement.MeasurementDiagnostics)
    assert diag.rank_min == 2
    assert diag.n_sources == 2


def test_interpolation_and_frequency_smoothing():
    m = _measurement()
    interp = sap.measurement.interpolate_transfer_linear(m, [150.0, 300.0])
    assert interp.transfer.shape == (2, 2, 2)
    assert interp.frequencies_hz.tolist() == [150.0, 300.0]
    smoothed = sap.measurement.frequency_smooth_magnitude_db(
        sap.measurement.transfer_magnitude_db(m.transfer),
        window_bins=3,
    )
    assert smoothed.shape == m.transfer.shape
    assert sap.measurement.source_frequency_slice(m, 0).shape == (2, 2)


def test_validation_rejects_bad_measurements():
    with pytest.raises(ValueError, match="frequency"):
        sap.measurement.ArrayMeasurement([1.0, 2.0], np.ones((3, 2)))
    with pytest.raises(ValueError, match="mode"):
        sap.measurement.normalize_transfer(np.ones((2, 2)), mode="bad")
    m = _measurement()
    with pytest.raises(ValueError, match="new frequencies"):
        sap.measurement.interpolate_transfer_linear(m, [50.0])
