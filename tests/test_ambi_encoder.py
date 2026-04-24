"""Tests for the ambisonic plane-wave encoder."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.ambi import (
    convert_ambi_normalization,
    encode_plane_wave,
)
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


class TestEncodePlaneWave:
    def test_single_source_shape_and_semantics(self):
        rng = np.random.default_rng(0)
        T = 1024
        sig = rng.standard_normal(T)
        direction = SphericalGrid(
            azimuth=np.array([np.pi / 4]),
            angle2=np.array([np.pi / 3]),
            convention="az_colat",
        )
        out = encode_plane_wave(sig, direction, max_order=3, basis="real")
        assert out.shape == (16, T)
        y = np.asarray(
            sh_matrix(SHBasisSpec(max_order=3, basis="real"), direction)
        )  # (1, 16)
        expected = y.T * sig[None, :]  # (16, T)
        assert_allclose(out, expected, atol=1e-12)

    def test_multiple_sources_are_summed(self):
        rng = np.random.default_rng(1)
        T = 256
        K = 5
        sig = rng.standard_normal((K, T))
        grid = SphericalGrid(
            azimuth=rng.uniform(0.0, 2 * np.pi, K),
            angle2=rng.uniform(0.1, np.pi - 0.1, K),
            convention="az_colat",
        )
        out = encode_plane_wave(sig, grid, max_order=3, basis="real")
        assert out.shape == (16, T)
        y = np.asarray(
            sh_matrix(SHBasisSpec(max_order=3, basis="real"), grid)
        )  # (K, 16)
        expected = y.T @ sig  # (16, T)
        assert_allclose(out, expected, atol=1e-10)

    def test_complex_basis(self):
        rng = np.random.default_rng(2)
        sig = rng.standard_normal(128)
        direction = SphericalGrid(
            azimuth=np.array([1.0]),
            angle2=np.array([0.8]),
            convention="az_colat",
        )
        out = encode_plane_wave(
            sig, direction, max_order=2, basis="complex",
        )
        assert out.shape == (9, 128)
        assert np.iscomplexobj(out)

    def test_normalization_conversion(self):
        """Encoding in SN3D matches ortho encoding converted to SN3D."""
        rng = np.random.default_rng(3)
        sig = rng.standard_normal(512)
        direction = SphericalGrid(
            azimuth=np.array([np.pi / 5]),
            angle2=np.array([np.pi / 4]),
            convention="az_colat",
        )
        out_ortho = encode_plane_wave(
            sig, direction, max_order=3, normalization="orthonormal",
        )
        out_sn3d = encode_plane_wave(
            sig, direction, max_order=3, normalization="sn3d",
        )
        converted = convert_ambi_normalization(
            out_ortho, max_order=3,
            from_="orthonormal", to="sn3d", axis=0,
        )
        assert_allclose(out_sn3d, converted, atol=1e-12)

    def test_rejects_mismatched_direction_count(self):
        sig = np.zeros((3, 100))
        # Only 2 directions for 3 sources.
        grid = SphericalGrid(
            azimuth=np.zeros(2),
            angle2=np.zeros(2),
            convention="az_colat",
        )
        with pytest.raises(ValueError, match="direction"):
            encode_plane_wave(sig, grid, max_order=1)

    def test_rejects_3d_input(self):
        grid = SphericalGrid(
            azimuth=np.zeros(1),
            angle2=np.zeros(1),
            convention="az_colat",
        )
        with pytest.raises(ValueError, match="1-D or 2-D"):
            encode_plane_wave(np.zeros((2, 3, 4)), grid, max_order=1)
