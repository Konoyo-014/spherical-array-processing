"""Tests for the head-tracking SH rotation helper."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.sh import (
    rotate_ambi_over_time,
    sh_rotation_matrix,
)
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


class TestStaticRotation:
    def test_static_vector_matches_direct_matmul(self):
        rng = np.random.default_rng(0)
        N = 3
        q = (N + 1) ** 2
        sig = rng.standard_normal((q, 1024))
        ang = np.array([0.3, 0.5, -0.2])
        expected = sh_rotation_matrix(N, *ang, basis="real") @ sig
        out = rotate_ambi_over_time(sig, ang, max_order=N, basis="real")
        assert_allclose(out, expected, atol=1e-12)

    def test_identity_keyframes_preserve_signal(self):
        rng = np.random.default_rng(1)
        N = 2
        q = (N + 1) ** 2
        sig = rng.standard_normal((q, 2048))
        angles = np.zeros((8, 3))
        out = rotate_ambi_over_time(
            sig, angles, max_order=N, basis="real",
            block_samples=256, crossfade_samples=64,
        )
        assert_allclose(out, sig, atol=1e-12)

    def test_time_major_auto_detected(self):
        rng = np.random.default_rng(2)
        N = 3
        q = (N + 1) ** 2
        sig = rng.standard_normal((q, 512))
        ang = np.array([0.1, 0.2, 0.3])
        out_chan = rotate_ambi_over_time(sig, ang, max_order=N, basis="real")
        out_time = rotate_ambi_over_time(sig.T, ang, max_order=N, basis="real")
        assert out_time.shape == sig.T.shape
        assert_allclose(out_time, out_chan.T, atol=1e-12)


class TestBlockRotation:
    def test_block_wise_rotation_equals_static_per_block(self):
        """With crossfade=0, each block should be exactly R_k · X_k."""
        rng = np.random.default_rng(3)
        N = 2
        q = (N + 1) ** 2
        block = 128
        K = 5
        sig = rng.standard_normal((q, K * block))
        angles = rng.standard_normal((K, 3)) * 0.5
        out = rotate_ambi_over_time(
            sig, angles, max_order=N, basis="real",
            block_samples=block, crossfade_samples=0,
        )
        for k in range(K):
            r = sh_rotation_matrix(N, *angles[k], basis="real")
            expected = r @ sig[:, k * block : (k + 1) * block]
            assert_allclose(
                out[:, k * block : (k + 1) * block], expected, atol=1e-12,
            )

    def test_crossfade_is_linear_and_continuous(self):
        """Verify the (1-α, α) linear ramp at the block boundary."""
        N = 1
        q = (N + 1) ** 2
        block = 64
        crossfade = 16
        # Deterministic signal: all-ones so both rotations scatter the
        # same energy pattern; crossfade output becomes exactly the
        # convex combination.
        sig = np.ones((q, 2 * block), dtype=float)
        angles = np.array([[0.0, 0.0, 0.0], [np.pi / 3, 0.0, 0.0]])
        out = rotate_ambi_over_time(
            sig, angles, max_order=N, basis="real",
            block_samples=block, crossfade_samples=crossfade,
        )
        r0 = sh_rotation_matrix(N, 0.0, 0.0, 0.0, basis="real")
        r1 = sh_rotation_matrix(N, np.pi / 3, 0.0, 0.0, basis="real")
        # At the start of block 1, alpha ramps 1/C, 2/C, ..., 1.
        alpha = np.linspace(1.0 / crossfade, 1.0, crossfade)
        x_head = sig[:, block : block + crossfade]
        expected = (1.0 - alpha) * (r0 @ x_head) + alpha * (r1 @ x_head)
        assert_allclose(
            out[:, block : block + crossfade], expected, atol=1e-12,
        )
        # Post-crossfade samples use R_1 cleanly.
        assert_allclose(
            out[:, block + crossfade : 2 * block],
            r1 @ sig[:, block + crossfade : 2 * block],
            atol=1e-12,
        )

    def test_trailing_tail_uses_last_rotation(self):
        rng = np.random.default_rng(4)
        N = 2
        q = (N + 1) ** 2
        block = 64
        K = 3
        sig = rng.standard_normal((q, K * block + 50))
        angles = rng.standard_normal((K, 3)) * 0.2
        out = rotate_ambi_over_time(
            sig, angles, max_order=N, basis="real",
            block_samples=block, crossfade_samples=0,
        )
        r_last = sh_rotation_matrix(N, *angles[-1], basis="real")
        assert_allclose(
            out[:, K * block :], r_last @ sig[:, K * block :], atol=1e-12,
        )


class TestPlaneWaveSemantics:
    def test_plane_wave_direction_rotates_with_keyframe(self):
        """A time-constant plane wave from +x, rotated by +π/2 about
        z in the second half, should decode to peak at +y in the second
        half and +x in the first half."""
        N = 3
        q = (N + 1) ** 2
        # Encode a constant-amplitude plane wave from azimuth=0, colat=π/2.
        src = SphericalGrid(
            azimuth=np.array([0.0]),
            angle2=np.array([np.pi / 2]),
            convention="az_colat",
        )
        y = np.asarray(
            sh_matrix(SHBasisSpec(max_order=N, basis="real"), src)
        )  # (1, Q)
        block = 128
        n_blocks = 8
        # All blocks carry the same plane wave.
        sig = np.tile(y.T, (1, n_blocks * block))  # (Q, T)
        # First four blocks identity; last four rotate by +π/2 about z.
        angles = np.zeros((n_blocks, 3))
        angles[4:, 0] = np.pi / 2
        out = rotate_ambi_over_time(
            sig, angles, max_order=N, basis="real",
            block_samples=block, crossfade_samples=0,
        )
        # Decode onto 2 probe directions: +x and +y.
        probe = SphericalGrid(
            azimuth=np.array([0.0, np.pi / 2]),
            angle2=np.array([np.pi / 2, np.pi / 2]),
            convention="az_colat",
        )
        y_probe = np.asarray(
            sh_matrix(SHBasisSpec(max_order=N, basis="real"), probe)
        )  # (2, Q)
        # Sample the middle of block 1 and block 5.
        idx_before = block + block // 2
        idx_after = 5 * block + block // 2
        amp_before = y_probe @ out[:, idx_before]
        amp_after = y_probe @ out[:, idx_after]
        # Before: +x > +y.  After: +y > +x.
        assert amp_before[0] > amp_before[1] + 1.0
        assert amp_after[1] > amp_after[0] + 1.0


class TestErrors:
    def test_bad_shape(self):
        with pytest.raises(ValueError, match=r"max_order\+1"):
            rotate_ambi_over_time(
                np.zeros((7, 100)), np.zeros(3), max_order=2, basis="real",
            )

    def test_too_short_signal(self):
        with pytest.raises(ValueError, match="shorter than"):
            rotate_ambi_over_time(
                np.zeros((4, 100)), np.zeros((4, 3)),
                max_order=1, basis="real", block_samples=50,
            )

    def test_bad_euler_shape(self):
        with pytest.raises(ValueError, match="euler"):
            rotate_ambi_over_time(
                np.zeros((4, 100)), np.zeros((5,)),
                max_order=1, basis="real",
            )
