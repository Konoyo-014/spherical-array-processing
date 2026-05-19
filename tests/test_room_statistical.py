"""Tests for statistical room-acoustics prediction helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.room import (
    ShoeboxAcousticStats,
    air_absorption_attenuation_iso9613,
    air_absorption_coefficient_iso9613,
    arau_puchades_rt60,
    classify_room_modes,
    critical_distance,
    critical_distance_from_rt60,
    equivalent_absorption_area,
    eyring_rt60,
    mean_absorption,
    millington_sette_rt60,
    rectangular_room_modes,
    room_constant,
    sabine_rt60,
    schroeder_frequency,
    shoebox_acoustic_stats,
    shoebox_axis_surface_areas,
    shoebox_surface_areas,
    shoebox_volume,
)


def test_shoebox_geometry_helpers():
    dims = (5.0, 4.0, 3.0)
    assert shoebox_volume(dims) == 60.0
    assert_allclose(
        shoebox_surface_areas(dims),
        [12.0, 12.0, 15.0, 15.0, 20.0, 20.0],
    )
    assert_allclose(shoebox_axis_surface_areas(dims), [24.0, 30.0, 40.0])


def test_absorption_area_and_mean_accept_frequency_bands():
    surfaces = shoebox_surface_areas((5.0, 4.0, 3.0))
    absorption = np.column_stack(
        [
            np.full(6, 0.1),
            np.full(6, 0.2),
        ]
    )
    assert_allclose(equivalent_absorption_area(surfaces, absorption), [9.4, 18.8])
    assert_allclose(mean_absorption(surfaces, absorption), [0.1, 0.2])


def test_sabine_eyring_and_millington_uniform_absorption():
    dims = (5.0, 4.0, 3.0)
    surfaces = shoebox_surface_areas(dims)
    volume = shoebox_volume(dims)
    alpha = 0.2
    constant = 24.0 * np.log(10.0) / 343.0
    expected_sabine = constant * volume / (surfaces.sum() * alpha)
    expected_eyring = constant * volume / (-surfaces.sum() * np.log1p(-alpha))
    assert_allclose(sabine_rt60(volume, surfaces, alpha), expected_sabine)
    assert_allclose(eyring_rt60(volume, surfaces, alpha), expected_eyring)
    assert_allclose(millington_sette_rt60(volume, surfaces, alpha), expected_eyring)


def test_arau_puchades_reduces_to_eyring_for_uniform_absorption():
    dims = (5.0, 4.0, 3.0)
    surfaces = shoebox_surface_areas(dims)
    expected = eyring_rt60(shoebox_volume(dims), surfaces, 0.35)
    assert_allclose(arau_puchades_rt60(dims, 0.35), expected, rtol=1e-12)


def test_arau_puchades_accepts_per_wall_bands():
    dims = (5.0, 4.0, 3.0)
    absorption = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.5, 0.6],
        ]
    )
    rt = arau_puchades_rt60(dims, absorption)
    assert rt.shape == (2,)
    assert np.all(np.isfinite(rt))
    assert np.all(rt > 0.0)


def test_room_constant_and_critical_distance_formulas():
    surfaces = shoebox_surface_areas((5.0, 4.0, 3.0))
    alpha = 0.25
    total = surfaces.sum()
    expected_room_constant = total * alpha / (1.0 - alpha)
    assert_allclose(room_constant(surfaces, alpha), expected_room_constant)
    assert_allclose(
        critical_distance(surfaces, alpha, directivity_factor=2.0),
        np.sqrt(2.0 * expected_room_constant / (16.0 * np.pi)),
    )


def test_schroeder_frequency_and_critical_distance_from_rt60():
    volume = 120.0
    rt = 0.6
    assert_allclose(schroeder_frequency(rt, volume), 2000.0 * np.sqrt(rt / volume))
    constant = 24.0 * np.log(10.0) / 343.0
    expected_area = constant * volume / rt
    assert_allclose(
        critical_distance_from_rt60(volume, rt, directivity_factor=1.5),
        np.sqrt(1.5 * expected_area / (16.0 * np.pi)),
    )


def test_rectangular_room_modes_and_classification():
    modes = rectangular_room_modes((5.0, 4.0, 3.0), 100.0, c=340.0)
    assert modes.shape[1] == 4
    assert_allclose(modes[0], [1.0, 0.0, 0.0, 34.0])
    labels = classify_room_modes(modes[:, :3].astype(int))
    assert labels[0] == "axial"
    assert "tangential" in set(labels.tolist())
    assert np.all(np.diff(modes[:, 3]) >= 0.0)


def test_air_absorption_iso9613_has_expected_scale_and_monotonicity():
    freqs = np.array([1000.0, 4000.0, 8000.0])
    coeff = air_absorption_coefficient_iso9613(
        freqs,
        temperature_c=20.0,
        relative_humidity=0.5,
        pressure_kpa=101.325,
    )
    assert coeff.shape == freqs.shape
    assert np.all(coeff > 0.0)
    assert coeff[2] > coeff[1] > coeff[0]
    assert 0.003 < coeff[0] < 0.006
    assert_allclose(
        air_absorption_attenuation_iso9613(freqs, 10.0),
        10.0 * air_absorption_coefficient_iso9613(freqs),
    )


def test_shoebox_acoustic_stats_bundle():
    stats = shoebox_acoustic_stats(
        (5.0, 4.0, 3.0),
        np.full(6, 0.25),
        directivity_factor=1.2,
    )
    assert isinstance(stats, ShoeboxAcousticStats)
    assert stats.volume_m3 == 60.0
    assert stats.surface_area_m2 == 94.0
    assert stats.schroeder_frequency_hz > 0.0
    assert stats.critical_distance_m > 0.0


def test_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="dimensions"):
        shoebox_surface_areas((1.0, 2.0))
    with pytest.raises(ValueError, match="absorption"):
        sabine_rt60(10.0, [1.0, 2.0], [0.2, 1.2])
    with pytest.raises(ValueError, match="positive"):
        critical_distance_from_rt60(10.0, 0.0)
    with pytest.raises(ValueError, match="relative_humidity"):
        air_absorption_coefficient_iso9613([1000.0], relative_humidity=1.2)
