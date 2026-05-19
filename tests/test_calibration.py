"""Tests for spherical-array geometry calibration helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spherical_array_processing as sap


def test_grid_cartesian_round_trip_and_normalization():
    grid = sap.array.fibonacci_grid(12)
    points = sap.calibration.grid_to_cartesian(grid, radius=0.2)
    assert points.shape == (12, 3)
    assert_allclose(np.linalg.norm(points, axis=1), 0.2, atol=1e-14)
    out_grid, radii = sap.calibration.cartesian_to_grid(points)
    assert out_grid.size == 12
    assert_allclose(radii, 0.2, atol=1e-14)
    assert_allclose(np.linalg.norm(sap.calibration.normalize_points(points), axis=1), 1.0)


def test_fit_sphere_and_radial_errors():
    grid = sap.array.fibonacci_grid(32)
    points = sap.calibration.grid_to_cartesian(grid, radius=0.5) + np.array([1.0, -2.0, 0.25])
    fit = sap.calibration.fit_sphere(points)
    assert isinstance(fit, sap.calibration.SphereFit)
    assert_allclose(fit.center, [1.0, -2.0, 0.25], atol=1e-12)
    assert_allclose(fit.radius, 0.5, atol=1e-12)
    assert fit.rms_error < 1e-12
    assert sap.calibration.radial_rms_error(points) < 1e-12


def test_kabsch_alignment_and_application():
    source = sap.calibration.grid_to_cartesian(sap.array.fibonacci_grid(20))
    rot = sap.calibration.rotation_matrix_from_vectors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    target = sap.calibration.rotate_points(source, rot) + np.array([0.1, -0.2, 0.3])
    alignment = sap.calibration.kabsch_alignment(source, target)
    assert isinstance(alignment, sap.calibration.RigidAlignment)
    aligned = sap.calibration.apply_rigid_alignment(source, alignment)
    assert_allclose(aligned, target, atol=1e-12)
    assert alignment.rms_error < 1e-12


def test_grid_alignment_and_error_reports():
    grid = sap.array.fibonacci_grid(10)
    rotated = sap.layouts.rotate_layout_z(sap.layouts.layout_from_grid(grid), np.deg2rad(20.0)).as_grid()
    alignment = sap.calibration.align_grid_to_grid(grid, rotated)
    recovered = sap.calibration.apply_alignment_to_grid(grid, alignment)
    assert sap.calibration.angular_rms_error(recovered, rotated) < 1e-12
    err = sap.calibration.angular_position_errors(recovered, rotated)
    assert err.shape == (10,)
    points = sap.calibration.grid_to_cartesian(grid)
    report = sap.calibration.sensor_position_error_report(points, points + 0.01)
    assert report["rms_m"] == pytest.approx(np.sqrt(3) * 0.01)


def test_array_geometry_from_points_records_fit_metadata():
    grid = sap.array.cubic_array(radius_m=0.1).sensor_grid
    points = sap.calibration.grid_to_cartesian(grid, radius=0.1) + np.array([0.5, 0.0, 0.0])
    geom = sap.calibration.array_geometry_from_points(points)
    assert geom.n_sensors == 8
    assert_allclose(geom.radius_m, 0.1, atol=1e-12)
    assert "fit_center_m" in geom.metadata


def test_calibration_validation():
    with pytest.raises(ValueError, match="N>=4"):
        sap.calibration.fit_sphere(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-zero"):
        sap.calibration.normalize_points([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        sap.calibration.kabsch_alignment(np.zeros((2, 3)), np.zeros((2, 2)))
