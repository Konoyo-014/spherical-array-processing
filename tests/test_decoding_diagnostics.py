"""Tests for Ambisonic decoder diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spherical_array_processing as sap


def test_basic_decoder_matrix_diagnostics():
    decoder = np.eye(4)
    assert sap.decoding.infer_decoder_order(decoder) == 1
    assert sap.decoding.decoder_rank(decoder) == 4
    assert_allclose(sap.decoding.decoder_singular_values(decoder), np.ones(4))
    assert_allclose(sap.decoding.decoder_condition_number(decoder), 1.0)
    assert_allclose(sap.decoding.decoder_frobenius_norm(decoder), 2.0)
    assert_allclose(sap.decoding.decoder_column_norms(decoder), np.ones(4))
    assert_allclose(sap.decoding.decoder_row_norms(decoder), np.ones(4))
    assert_allclose(sap.decoding.decoder_column_power(decoder), np.ones(4))
    assert_allclose(sap.decoding.decoder_row_power(decoder), np.ones(4))
    assert_allclose(sap.decoding.decoder_diffuse_loudspeaker_levels_db(decoder), np.zeros(4))
    assert_allclose(sap.decoding.decoder_loudspeaker_gain_spread_db(decoder), 0.0)
    assert_allclose(sap.decoding.decoder_mode_gain_spread_db(decoder), 0.0)


def test_energy_leakage_and_correlation_metrics():
    decoder = np.eye(4)
    assert_allclose(sap.decoding.decoder_energy_matrix(decoder), np.eye(4))
    assert_allclose(sap.decoding.decoder_energy_preservation_error(decoder), 0.0)
    assert_allclose(sap.decoding.decoder_mode_leakage_ratio(decoder), 0.0)
    assert_allclose(sap.decoding.decoder_column_correlation(decoder), np.eye(4))
    assert_allclose(sap.decoding.decoder_row_correlation(decoder), np.eye(4))
    assert_allclose(sap.decoding.decoder_projection_matrix(decoder), np.eye(4))


def test_mode_matching_helpers_with_identity_basis():
    decoder = np.eye(4)
    basis = np.eye(4)
    response = sap.decoding.mode_matching_matrix(decoder, basis)
    assert_allclose(response, np.eye(4))
    assert_allclose(sap.decoding.mode_matching_error(decoder, basis), 0.0)
    assert_allclose(sap.decoding.mode_response_diagonal(decoder, basis), np.ones(4))
    assert_allclose(sap.decoding.mode_response_leakage_ratio(decoder, basis), 0.0)


def test_probe_vector_metrics_on_simple_three_speaker_layout():
    grid = sap.SphericalGrid(
        azimuth=np.array([0.0, np.pi / 2.0, 0.0]),
        angle2=np.array([0.0, 0.0, np.pi / 2.0]),
        convention="az_el",
    )
    decoder = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    probes = decoder.copy()
    response = sap.decoding.probe_response(decoder, probes)
    assert_allclose(response, np.eye(3))
    assert_allclose(sap.decoding.probe_response_energy(decoder, probes), np.ones(3))
    assert_allclose(sap.decoding.probe_response_peak_speaker(decoder, probes), [0, 1, 2])
    evec = sap.decoding.probe_energy_vector(decoder, probes, grid)
    vvec = sap.decoding.probe_velocity_vector(decoder, probes, grid)
    target = sap.decoding.speaker_directions_cartesian(grid)
    assert_allclose(evec, target, atol=1e-15)
    assert_allclose(vvec, target, atol=1e-15)
    assert_allclose(sap.decoding.vector_magnitudes(evec), np.ones(3))
    assert_allclose(sap.decoding.vector_angle_errors_deg(evec, target), np.zeros(3), atol=1e-10)
    assert_allclose(
        sap.decoding.probe_energy_vector_errors_deg(decoder, probes, grid, grid),
        np.zeros(3),
        atol=1e-10,
    )
    assert_allclose(
        sap.decoding.probe_velocity_vector_errors_deg(decoder, probes, grid, grid),
        np.zeros(3),
        atol=1e-10,
    )


def test_decoder_normalization_and_health_report():
    decoder = np.diag([1.0, 2.0, 4.0, 8.0])
    col = sap.decoding.normalize_decoder_column_norms(decoder)
    row = sap.decoding.normalize_decoder_row_norms(decoder)
    assert_allclose(sap.decoding.decoder_column_norms(col), np.ones(4))
    assert_allclose(sap.decoding.decoder_row_norms(row), np.ones(4))
    report = sap.decoding.decoder_health_report(np.eye(4), loudspeaker_basis=np.eye(4))
    assert report["max_order"] == 1
    assert report["rank"] == 4
    assert_allclose(report["condition_number"], 1.0)
    assert_allclose(report["mode_matching_error"], 0.0)


def test_decoder_diagnostics_validation():
    with pytest.raises(ValueError, match="2-D"):
        sap.decoding.validate_decoder_matrix(np.zeros(4))
    with pytest.raises(ValueError, match="column count"):
        sap.decoding.infer_decoder_order(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="max_order"):
        sap.decoding.validate_decoder_matrix(np.eye(4), max_order=2)
    with pytest.raises(ValueError, match="same shape"):
        sap.decoding.mode_matching_matrix(np.eye(4), np.zeros((5, 4)))
