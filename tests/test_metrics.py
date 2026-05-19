"""Tests for cross-module spatial-audio metrics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spherical_array_processing as sap


def test_angular_errors_and_reports():
    est = np.deg2rad([[0.0, 0.0], [90.0, 0.0]])
    ref = np.deg2rad([[0.0, 0.0], [80.0, 0.0]])
    err = sap.metrics.angular_error_deg(est, ref)
    assert_allclose(err, [0.0, 10.0], atol=1e-12)
    report = sap.metrics.direction_error_report(est, ref)
    assert isinstance(report, sap.metrics.DirectionErrorReport)
    assert_allclose(report.mean_deg, 5.0, atol=1e-12)
    assert_allclose(report.rmse_deg, np.sqrt(50.0), atol=1e-12)
    assert sap.metrics.wrap_azimuth_rad(3 * np.pi) == pytest.approx(-np.pi)


def test_direction_assignment_uses_minimum_cost():
    refs = np.deg2rad([[0.0, 0.0], [90.0, 0.0]])
    ests = refs[::-1]
    assignment = sap.metrics.assign_directions(ests, refs)
    assert isinstance(assignment, sap.metrics.DirectionAssignment)
    assert assignment.estimate_indices.tolist() == [0, 1]
    assert assignment.reference_indices.tolist() == [1, 0]
    assert_allclose(assignment.errors_rad, [0.0, 0.0], atol=1e-12)
    report = sap.metrics.matched_direction_error_report(ests, refs)
    assert_allclose(report.max_deg, 0.0, atol=1e-12)


def test_resultant_energy_and_velocity_vectors():
    dirs = np.deg2rad([[0.0, 0.0], [180.0, 0.0]])
    assert_allclose(sap.metrics.resultant_vector(dirs), [0.0, 0.0, 0.0], atol=1e-15)
    assert_allclose(sap.metrics.resultant_length(dirs), 0.0, atol=1e-15)
    evec = sap.metrics.energy_vector(dirs, [1.0, 0.0])
    assert_allclose(evec, [1.0, 0.0, 0.0], atol=1e-15)
    vvec = sap.metrics.velocity_vector(dirs, [1.0, 0.0])
    assert_allclose(vvec, [1.0, 0.0, 0.0], atol=1e-15)
    assert_allclose(sap.metrics.vector_direction_error_deg(evec, np.deg2rad([0.0, 0.0])), 0.0, atol=1e-12)


def test_signal_and_spectral_errors():
    ref = np.array([1.0, 2.0, 4.0])
    est = 2.0 * ref
    assert_allclose(sap.metrics.normalized_correlation(ref, est), 1.0)
    assert sap.metrics.scale_invariant_error_db(ref, est) < -300.0
    assert_allclose(sap.metrics.log_spectral_distance_db(ref, ref), 0.0)
    assert_allclose(
        sap.metrics.magnitude_response_error_db(ref, est),
        np.full(3, 6.020599913279624),
    )
    complex_ref = np.array([1.0 + 0.0j, 1.0 + 0.0j])
    complex_est = np.array([1.0j, -1.0j])
    assert_allclose(sap.metrics.phase_error_rad(complex_ref, complex_est), [np.pi / 2, -np.pi / 2])
    assert_allclose(sap.metrics.rms_error(ref, ref + 1.0), 1.0)
    assert_allclose(sap.metrics.relative_error(ref, ref), 0.0)


def test_metrics_validation():
    with pytest.raises(ValueError, match="directions"):
        sap.metrics.angular_error_rad([[0.0, 0.0, 0.0]], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="matching shape"):
        sap.metrics.normalized_correlation([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-zero"):
        sap.metrics.velocity_vector([[0.0, 0.0]], [0.0])
