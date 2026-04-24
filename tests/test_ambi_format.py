"""Tests for `spherical_array_processing.ambi.format`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.ambi import (
    acn_to_fuma,
    convert_ambi_normalization,
    fuma_to_acn,
)
from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec


class TestNormalizationConverter:
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        for fr in ("orthonormal", "n3d", "sn3d"):
            for to in ("orthonormal", "n3d", "sn3d"):
                coeffs = rng.standard_normal(16)
                out = convert_ambi_normalization(
                    coeffs, max_order=3, from_=fr, to=to,
                )
                back = convert_ambi_normalization(
                    out, max_order=3, from_=to, to=fr,
                )
                assert_allclose(back, coeffs, atol=1e-12)

    def test_identity(self):
        rng = np.random.default_rng(1)
        c = rng.standard_normal(16)
        out = convert_ambi_normalization(
            c, max_order=3, from_="sn3d", to="sn3d",
        )
        assert_allclose(out, c, atol=0)

    def test_n3d_to_sn3d_scaling_is_per_order(self):
        """c_SN3D = c_N3D · √(2n+1)."""
        c_n3d = np.ones(16)
        out = convert_ambi_normalization(
            c_n3d, max_order=3, from_="n3d", to="sn3d",
        )
        expected = np.concatenate([
            np.full(1, np.sqrt(1)),   # n=0
            np.full(3, np.sqrt(3)),   # n=1
            np.full(5, np.sqrt(5)),   # n=2
            np.full(7, np.sqrt(7)),   # n=3
        ])
        assert_allclose(out, expected, atol=1e-14)

    def test_axis_override(self):
        rng = np.random.default_rng(2)
        coeffs = rng.standard_normal((4, 16, 3))  # channel axis=1
        out = convert_ambi_normalization(
            coeffs, max_order=3, from_="orthonormal", to="sn3d", axis=1,
        )
        assert out.shape == (4, 16, 3)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match=r"\(max_order\+1\)"):
            convert_ambi_normalization(
                np.zeros(15), max_order=3, from_="sn3d", to="n3d",
            )

    def test_physical_field_preserved(self):
        """Reconstructing a plane wave from ortho coeffs + ortho basis
        must equal reconstructing the same field from SN3D coeffs +
        SN3D basis."""
        grid = fibonacci_grid(300)
        # Build ortho basis matrix.
        spec_ortho = SHBasisSpec(max_order=3, basis="real", normalization="orthonormal")
        Y_ortho = np.asarray(sh_matrix(spec_ortho, grid))  # (G, Q)
        # Field = a single SH channel (e.g. the n=2, m=0 component).
        q_target = 6  # ACN for n=2, m=0
        c_ortho = np.zeros(16)
        c_ortho[q_target] = 1.0
        f_from_ortho = Y_ortho @ c_ortho  # (G,)
        # Convert coeffs to SN3D and use an SN3D basis to rebuild.
        c_sn3d = convert_ambi_normalization(
            c_ortho, max_order=3, from_="orthonormal", to="sn3d",
        )
        spec_sn3d = SHBasisSpec(max_order=3, basis="real", normalization="sn3d")
        Y_sn3d = np.asarray(sh_matrix(spec_sn3d, grid))
        f_from_sn3d = Y_sn3d @ c_sn3d
        assert_allclose(f_from_sn3d, f_from_ortho, atol=1e-12)


class TestFumaConverter:
    def test_fuma_round_trip(self):
        rng = np.random.default_rng(3)
        for N in (0, 1, 2, 3):
            n_ch = (N + 1) ** 2
            c = rng.standard_normal(n_ch)
            fuma = acn_to_fuma(c, max_order=N, from_sn3d=True)
            back = fuma_to_acn(fuma, max_order=N, to_sn3d=True)
            assert_allclose(back, c, atol=1e-12)

    def test_fuma_w_is_half_sqrt2(self):
        c_sn3d = np.array([1.0, 0.0, 0.0, 0.0])  # W-only
        c_fuma = acn_to_fuma(c_sn3d, max_order=1, from_sn3d=True)
        assert_allclose(c_fuma[0], 1.0 / np.sqrt(2.0), atol=1e-14)
        # FuMa X/Y/Z (channels 1,2,3) should be zero.
        assert_allclose(c_fuma[1:], 0.0, atol=0)

    def test_fuma_x_maps_from_acn_3(self):
        """FuMa channel 1 (X) ↔ ACN channel 3 (n=1, m=+1)."""
        c_sn3d = np.zeros(4)
        c_sn3d[3] = 1.0  # ACN n=1 m=+1
        c_fuma = acn_to_fuma(c_sn3d, max_order=1, from_sn3d=True)
        assert_allclose(c_fuma[0], 0.0, atol=0)
        assert_allclose(c_fuma[1], 1.0, atol=0)  # X

    def test_higher_order_rejected(self):
        with pytest.raises(ValueError, match="FuMa"):
            acn_to_fuma(np.zeros(25), max_order=4)
        with pytest.raises(ValueError, match="FuMa"):
            fuma_to_acn(np.zeros(25), max_order=4)

    def test_from_orthonormal_path(self):
        """acn_to_fuma(from_sn3d=False) must equal
        convert_ambi_normalization(→sn3d) then acn_to_fuma(from_sn3d=True)."""
        rng = np.random.default_rng(4)
        c_ortho = rng.standard_normal(16)
        direct = acn_to_fuma(c_ortho, max_order=3, from_sn3d=False)
        via_sn3d = acn_to_fuma(
            convert_ambi_normalization(
                c_ortho, max_order=3, from_="orthonormal", to="sn3d",
            ),
            max_order=3, from_sn3d=True,
        )
        assert_allclose(direct, via_sn3d, atol=1e-14)

    def test_axis_override(self):
        rng = np.random.default_rng(5)
        # (batch, Q=4, time)
        c = rng.standard_normal((2, 4, 8))
        fuma = acn_to_fuma(c, max_order=1, axis=1)
        assert fuma.shape == (2, 4, 8)
        back = fuma_to_acn(fuma, max_order=1, axis=1)
        assert_allclose(back, c, atol=1e-12)

    def test_order_2_and_3_round_trip_with_all_channels_active(self):
        grid = fibonacci_grid(300)
        for N in (2, 3):
            n_ch = (N + 1) ** 2
            coeffs = np.linspace(-0.75, 0.75, n_ch)
            coeffs += 0.1 * np.cos(np.arange(n_ch))
            fuma = acn_to_fuma(coeffs, max_order=N, from_sn3d=True)
            back = fuma_to_acn(fuma, max_order=N, to_sn3d=True)
            assert_allclose(back, coeffs, atol=1e-12)

            y_sn3d = np.asarray(
                sh_matrix(
                    SHBasisSpec(max_order=N, basis="real", normalization="sn3d"),
                    grid,
                )
            )
            field_direct = y_sn3d @ coeffs
            field_round_trip = y_sn3d @ back
            assert_allclose(field_round_trip, field_direct, atol=1e-12)

    def test_requested_fuma_spot_checks(self):
        cases = [
            (2, 5, 6, 2.0 / np.sqrt(3.0)),
            (2, 4, 8, 2.0 / np.sqrt(3.0)),
            (3, 11, 11, np.sqrt(45.0 / 32.0)),
            (3, 9, 15, np.sqrt(8.0 / 5.0)),
        ]
        for max_order, acn_idx, fuma_idx, expected_weight in cases:
            coeffs = np.zeros((max_order + 1) ** 2)
            coeffs[acn_idx] = 1.0
            fuma = acn_to_fuma(coeffs, max_order=max_order, from_sn3d=True)
            expected = np.zeros_like(fuma)
            expected[fuma_idx] = expected_weight
            assert_allclose(fuma, expected, atol=1e-14)
