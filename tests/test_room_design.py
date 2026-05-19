"""Tests for inverse room-acoustics design helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing import room


def test_inverse_sabine_and_eyring_targets_round_trip():
    dims = (5.0, 4.0, 3.0)
    volume = room.shoebox_volume(dims)
    surface = room.shoebox_surface_areas(dims).sum()
    rt = 0.6
    alpha_s = room.target_mean_absorption_sabine(volume, surface, rt)
    alpha_e = room.target_mean_absorption_eyring(volume, surface, rt)
    assert 0.0 < alpha_s < 1.0
    assert 0.0 < alpha_e < 1.0
    assert_allclose(room.verify_target_rt60(dims, alpha_s, model="sabine"), rt)
    assert_allclose(room.verify_target_rt60(dims, alpha_e, model="eyring"), rt)
    assert_allclose(
        room.target_uniform_reflection_sabine(volume, surface, rt),
        room.mean_absorption_to_reflection(alpha_s),
    )


def test_room_design_target_bundle_and_absorption_budget():
    target = room.room_design_target((5.0, 4.0, 3.0), 0.5)
    assert isinstance(target, room.RoomDesignTarget)
    assert target.volume_m3 == 60.0
    assert target.surface_area_m2 == 94.0
    surfaces = room.shoebox_surface_areas((5.0, 4.0, 3.0))
    budget = room.absorption_budget_per_surface(
        surfaces,
        target.equivalent_absorption_area_m2,
    )
    assert budget.shape == (6,)
    assert_allclose(np.sum(surfaces * budget), target.equivalent_absorption_area_m2)


def test_room_ratio_bolt_area_and_modal_density():
    ratio = room.room_ratio((5.0, 4.0, 3.0))
    assert_allclose(ratio, [1.0, 4.0 / 3.0, 5.0 / 3.0])
    assert room.bolt_area_score((5.0, 4.0, 3.0)) >= 0.0
    cumulative = room.modal_density_weyl([100.0, 200.0], 60.0)
    density = room.modal_density_per_hz([100.0, 200.0], 60.0)
    assert cumulative[1] > cumulative[0]
    assert density[1] > density[0]


def test_modal_band_overlap_and_target_critical_distance():
    dims = (5.0, 4.0, 3.0)
    count = room.count_modes_in_band(dims, 20.0, 120.0)
    assert count > 0
    overlap = room.modal_overlap_factor([100.0, 200.0], room.shoebox_volume(dims), 0.5)
    assert overlap.shape == (2,)
    assert overlap[1] > overlap[0]
    assert room.room_constant_for_target(dims, 0.5) > 0.0
    assert room.critical_distance_for_target(dims, 0.5) > 0.0


def test_design_validation():
    with pytest.raises(ValueError, match="positive"):
        room.target_absorption_area_sabine(0.0, 1.0)
    with pytest.raises(ValueError, match="reflection"):
        room.reflection_to_mean_absorption(2.0)
    with pytest.raises(ValueError, match="band"):
        room.count_modes_in_band((5.0, 4.0, 3.0), 200.0, 100.0)
