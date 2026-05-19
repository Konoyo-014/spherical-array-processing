"""Tests for loudspeaker layout utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spherical_array_processing as sap


def test_layout_container_and_conversion():
    layout = sap.layouts.stereo_layout()
    assert layout.n_speakers == 2
    assert layout.labels == ("L", "R")
    assert layout.as_grid().size == 2
    assert_allclose(sap.layouts.layout_to_degrees(layout), [[30.0, 0.0], [-30.0, 0.0]])
    xyz = sap.layouts.layout_cartesian(layout)
    assert xyz.shape == (2, 3)
    assert_allclose(np.linalg.norm(xyz, axis=1), 1.0)


def test_common_layout_presets_have_expected_counts():
    assert sap.layouts.itu_5_1_layout().n_speakers == 5
    assert sap.layouts.itu_5_1_layout(include_lfe=True).n_speakers == 6
    assert sap.layouts.itu_7_1_layout().n_speakers == 7
    assert sap.layouts.itu_5_1_4_layout().n_speakers == 9
    assert sap.layouts.itu_7_1_4_layout().n_speakers == 11
    assert sap.layouts.cube_layout().n_speakers == 8
    assert sap.layouts.octahedral_layout().n_speakers == 6
    assert sap.layouts.tetrahedral_layout().n_speakers == 4
    assert "itu_5_1_4" in sap.layouts.layout_registry()
    assert sap.layouts.get_layout("itu-5.1.4").n_speakers == 9


def test_layout_geometry_diagnostics():
    layout = sap.layouts.horizontal_layout(4)
    pairwise = sap.layouts.layout_pairwise_angles(layout)
    assert pairwise.shape == (4, 4)
    assert_allclose(np.diag(pairwise), 0.0, atol=1e-12)
    assert_allclose(sap.layouts.layout_nearest_neighbor_angles(layout), np.full(4, np.pi / 2), atol=1e-12)
    assert_allclose(sap.layouts.layout_min_separation(layout), np.pi / 2, atol=1e-12)
    assert sap.layouts.layout_is_hemispherical(layout)
    assert not sap.layouts.layout_has_upper_hemisphere(layout)
    assert not sap.layouts.layout_has_lower_hemisphere(layout)
    centroid = sap.layouts.layout_centroid_vector(layout)
    assert_allclose(centroid, np.zeros(3), atol=1e-15)


def test_layout_transformations_and_subsets():
    layout = sap.layouts.itu_5_1_4_layout()
    mirrored = sap.layouts.mirror_layout_z(layout)
    assert_allclose(mirrored.as_grid().elevation, -layout.as_grid().elevation)
    rotated = sap.layouts.rotate_layout_z(sap.layouts.stereo_layout(), np.deg2rad(30.0))
    assert_allclose(np.rad2deg(rotated.azimuth), [60.0, 0.0], atol=1e-12)
    subset = sap.layouts.subset_layout(layout, [0, 1, 2])
    assert subset.n_speakers == 3
    assert subset.labels == ("L", "R", "C")


def test_layout_validation():
    with pytest.raises(ValueError, match="shape"):
        sap.layouts.layout_from_degrees([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="labels"):
        sap.layouts.layout_from_degrees([[0.0, 0.0]], labels=("a", "b"))
    with pytest.raises(ValueError, match="unknown"):
        sap.layouts.get_layout("not-a-layout")
    with pytest.raises(ValueError, match="at least two"):
        sap.layouts.layout_nearest_neighbor_angles(
            sap.layouts.layout_from_degrees([[0.0, 0.0]])
        )
