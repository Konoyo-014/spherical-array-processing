"""Tests for spherical sampling and array diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing import array


def test_grid_distance_and_weight_helpers():
    grid = array.fibonacci_grid(16)
    xyz = array.grid_cartesian(grid)
    assert xyz.shape == (16, 3)
    assert_allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-14)
    d = array.pairwise_angular_distance_matrix(grid)
    assert d.shape == (16, 16)
    assert_allclose(np.diag(d), 0.0, atol=1e-12)
    nn = array.nearest_neighbor_angles(grid)
    assert nn.shape == (16,)
    assert array.grid_min_angular_distance(grid) > 0.0
    assert array.grid_covering_radius(grid) >= array.grid_min_angular_distance(grid)
    assert_allclose(array.grid_weight_sum(grid), 4.0 * np.pi, atol=1e-12)
    assert_allclose(array.normalized_grid_weights(grid).sum(), 4.0 * np.pi, atol=1e-12)


def test_gauss_legendre_grid_is_exact_for_low_order_orthonormal_basis():
    grid = array.gauss_legendre_sampling(3)
    err = array.sh_gram_error(grid, 3)
    assert err.shape == (16, 16)
    assert np.max(np.abs(err)) < 1e-12
    assert array.exact_quadrature_order(grid, max_search_order=4, tolerance=1e-10) >= 3
    assert array.sh_sampling_rank(grid, 3) == 16
    assert array.sh_condition_number(grid, 3) < 10.0


def test_grid_diagnostics_bundle():
    grid = array.gauss_legendre_sampling(2)
    diag = array.grid_diagnostics(grid, 2)
    assert isinstance(diag, array.GridDiagnostics)
    assert diag.n_points == grid.size
    assert diag.gram_error_max < 1e-12
    assert diag.quadrature_condition > 0.0


def test_array_diagnostics_for_eigenmike():
    geom = array.em32_eigenmike()
    diag = array.array_diagnostics(geom, 3)
    assert isinstance(diag, array.ArrayDiagnostics)
    assert diag.n_sensors == 32
    assert diag.max_order == 3
    assert diag.recommended_order == 4
    assert not diag.underdetermined
    assert diag.aliasing_frequency_hz > 0.0
    assert_allclose(array.array_aperture_diameter(geom), 0.084)
    assert array.array_is_order_supported(geom, 4)
    assert not array.array_is_order_supported(geom, 5)


def test_sensor_distance_and_noise_gain_shapes():
    geom = array.cubic_array(radius_m=0.1)
    d_euclid = array.sensor_distance_matrix(geom)
    d_ang = array.sensor_angular_distance_matrix(geom)
    assert d_euclid.shape == (8, 8)
    assert d_ang.shape == (8, 8)
    assert_allclose(np.diag(d_euclid), 0.0, atol=1e-12)
    assert_allclose(np.diag(d_ang), 0.0, atol=1e-12)
    gain = array.modal_noise_gain(geom, 1)
    assert gain.shape == (4,)
    assert np.all(np.isfinite(gain))
    gain_db = array.modal_noise_gain_db(geom, 1)
    assert gain_db.shape == (4,)


def test_validation_rejects_too_small_grids():
    with pytest.raises(ValueError, match="at least two"):
        array.nearest_neighbor_angles(array.fibonacci_grid(1))
    with pytest.raises(ValueError, match="positive"):
        array.recommended_order_from_sensor_count(0)
