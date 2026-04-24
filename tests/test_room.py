"""Tests for `spherical_array_processing.room` — shoebox image-source RIR."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.room import ShoeboxRoom, shoebox_rir, shoebox_sh_rir


class TestMonauralShoeboxRIR:
    def test_scalar_reflection_broadcast_matches_uniform_tuple(self):
        room_scalar = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=0.8)
        room_tuple = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=(0.8,) * 6)
        src = np.array([1.5, 2.0, 1.5])
        lis = np.array([4.0, 3.0, 1.5])
        rir_scalar, dirs_scalar, delays_scalar = shoebox_rir(
            room_scalar, src, lis, fs=16000.0, ir_length=2048, max_reflection_order=6,
        )
        rir_tuple, dirs_tuple, delays_tuple = shoebox_rir(
            room_tuple, src, lis, fs=16000.0, ir_length=2048, max_reflection_order=6,
        )
        assert_allclose(rir_scalar, rir_tuple, atol=1e-12)
        assert_allclose(dirs_scalar, dirs_tuple, atol=1e-12)
        assert_allclose(delays_scalar, delays_tuple, atol=1e-12)

    def test_direct_path_delay_and_direction(self):
        room = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=(0.0,) * 6)
        src = np.array([1.5, 2.0, 1.5])
        lis = np.array([4.0, 3.0, 1.5])
        rir, dirs, delays = shoebox_rir(
            room, src, lis, fs=16000.0, ir_length=512, max_reflection_order=0,
        )
        # With zero reflection only the direct path exists.
        assert np.count_nonzero(rir) == 1
        expected_delay = np.linalg.norm(src - lis) / 343.0
        assert_allclose(delays[0], expected_delay, atol=1e-9)
        expected_dir = (src - lis) / np.linalg.norm(src - lis)
        assert_allclose(dirs[0], expected_dir, atol=1e-9)

    def test_reflection_coeff_scales_image_strength(self):
        room_strong = ShoeboxRoom(
            dimensions_m=(4.0, 4.0, 3.0), reflection=(0.9,) * 6
        )
        room_weak = ShoeboxRoom(
            dimensions_m=(4.0, 4.0, 3.0), reflection=(0.1,) * 6
        )
        rir_strong, _, _ = shoebox_rir(
            room_strong, (1.0, 1.0, 1.0), (3.0, 3.0, 1.5),
            fs=16000.0, ir_length=8192, max_reflection_order=12,
        )
        rir_weak, _, _ = shoebox_rir(
            room_weak, (1.0, 1.0, 1.0), (3.0, 3.0, 1.5),
            fs=16000.0, ir_length=8192, max_reflection_order=12,
        )
        # Late-tail energy should be orders of magnitude larger for the
        # strong-reflection room.
        tail_start = int(0.1 * 16000)
        energy_strong = float(np.sum(rir_strong[tail_start:] ** 2))
        energy_weak = float(np.sum(rir_weak[tail_start:] ** 2))
        assert energy_strong > energy_weak * 100

    def test_rejects_out_of_room_source(self):
        room = ShoeboxRoom(dimensions_m=(4.0, 4.0, 3.0))
        with pytest.raises(ValueError, match="source"):
            shoebox_rir(
                room, (5.0, 2.0, 1.5), (1.0, 1.0, 1.5),
                fs=16000.0, ir_length=512,
            )

    def test_rejects_negative_max_reflection_order(self):
        room = ShoeboxRoom(dimensions_m=(4.0, 4.0, 3.0))
        with pytest.raises(ValueError, match="max_reflection_order"):
            shoebox_rir(
                room, (1.0, 2.0, 1.5), (3.0, 1.0, 1.5),
                fs=16000.0, ir_length=512, max_reflection_order=-1,
            )

    def test_sinc_interpolation_sharper_timing_than_nearest(self):
        """Sinc interpolation should place the direct path much closer
        to its fractional sample location than nearest-sample does."""
        room = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=(0.0,) * 6)
        fs = 16000.0
        src = (1.5, 2.0, 1.5)
        lis = (4.0, 3.0, 1.5)
        # Expected fractional delay:
        # distance ≈ sqrt(6.25 + 1 + 0) = 2.6926 m, at c=343 → 7.85 ms → 125.6 samples.
        rir_n, _, delays = shoebox_rir(
            room, src, lis, fs=fs, ir_length=512,
            max_reflection_order=0, interpolation="nearest",
        )
        rir_s, _, _ = shoebox_rir(
            room, src, lis, fs=fs, ir_length=512,
            max_reflection_order=0, interpolation="sinc", fir_taps=21,
        )
        expected = float(delays[0] * fs)
        n = np.arange(rir_n.size)
        err_nearest = abs(
            float(np.sum(n * rir_n ** 2) / np.sum(rir_n ** 2)) - expected
        )
        err_sinc = abs(
            float(np.sum(n * rir_s ** 2) / np.sum(rir_s ** 2)) - expected
        )
        assert err_sinc < err_nearest
        assert err_sinc < 0.3  # sub-sample accuracy

    def test_sinc_kernel_length_effect(self):
        """Larger fir_taps should further tighten the kernel main lobe
        (energy more concentrated around the image-source arrival)."""
        room = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=(0.0,) * 6)
        rir_short, _, _ = shoebox_rir(
            room, (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            fs=16000.0, ir_length=512,
            max_reflection_order=0, interpolation="sinc", fir_taps=9,
        )
        rir_long, _, _ = shoebox_rir(
            room, (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            fs=16000.0, ir_length=512,
            max_reflection_order=0, interpolation="sinc", fir_taps=33,
        )
        assert int(np.count_nonzero(rir_short)) == 9
        assert int(np.count_nonzero(rir_long)) == 33

    def test_sinc_rejects_tiny_fir_taps(self):
        room = ShoeboxRoom(dimensions_m=(4.0, 4.0, 3.0), reflection=(0.5,) * 6)
        with pytest.raises(ValueError, match="fir_taps"):
            shoebox_rir(
                room, (1.0, 2.0, 1.5), (3.0, 1.0, 1.5),
                fs=16000.0, ir_length=256,
                interpolation="sinc", fir_taps=1,
            )

    def test_sinc_keeps_kernel_that_overlaps_buffer_end(self):
        """Regression: a fractional-delay kernel centred slightly beyond
        the last sample must still contribute when its clipped support
        overlaps the buffer."""
        fs = 16000.0
        ir_length = 10
        delay_samples = ir_length - 0.2
        distance_m = delay_samples * 343.0 / fs
        room = ShoeboxRoom(dimensions_m=(2.0, 2.0, 2.0), reflection=(0.0,) * 6)
        src = (0.5, 1.0, 1.0)
        lis = (0.5 + distance_m, 1.0, 1.0)
        rir_nearest, _, _ = shoebox_rir(
            room, src, lis, fs=fs, ir_length=ir_length,
            max_reflection_order=0, interpolation="nearest",
        )
        rir_sinc, dirs, delays = shoebox_rir(
            room, src, lis, fs=fs, ir_length=ir_length,
            max_reflection_order=0, interpolation="sinc", fir_taps=5,
        )
        assert not np.any(rir_nearest)
        assert np.any(rir_sinc)
        assert rir_sinc[-1] != 0.0
        assert delays.shape == (1,)
        assert_allclose(delays[0] * fs, delay_samples, atol=1e-12)
        assert_allclose(dirs[0], np.array([-1.0, 0.0, 0.0]), atol=1e-12)

    def test_invalid_interpolation_mode(self):
        room = ShoeboxRoom(dimensions_m=(4.0, 4.0, 3.0), reflection=(0.5,) * 6)
        with pytest.raises(ValueError, match="interpolation"):
            shoebox_rir(
                room, (1.0, 2.0, 1.5), (3.0, 1.0, 1.5),
                fs=16000.0, ir_length=256, interpolation="cubic",
            )


class TestShoeboxSHRIR:
    def test_w_channel_matches_monaural(self):
        """The W (omni) channel of the SH-RIR equals the monaural RIR
        divided by ``sqrt(4π)`` because each image contributes
        amplitude ``A`` to ``rir`` and ``A · Y_0^0 = A / sqrt(4π)`` to
        the W channel."""
        room = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=(0.8,) * 6)
        src = (1.5, 2.0, 1.5)
        lis = (4.0, 3.0, 1.5)
        rir, _, _ = shoebox_rir(
            room, src, lis, fs=16000.0, ir_length=4096, max_reflection_order=10,
        )
        sh_rir = shoebox_sh_rir(
            room, src, lis, fs=16000.0, ir_length=4096, max_order=3,
            max_reflection_order=10,
        )
        assert sh_rir.shape == (16, 4096)
        assert_allclose(sh_rir[0], rir / np.sqrt(4 * np.pi), atol=1e-12)

    def test_w_channel_matches_monaural_with_sinc_boundary_clipping(self):
        room = ShoeboxRoom(dimensions_m=(2.0, 2.0, 2.0), reflection=(0.0,) * 6)
        fs = 16000.0
        ir_length = 10
        delay_samples = ir_length - 0.2
        distance_m = delay_samples * 343.0 / fs
        src = (0.5, 1.0, 1.0)
        lis = (0.5 + distance_m, 1.0, 1.0)
        rir, _, _ = shoebox_rir(
            room, src, lis, fs=fs, ir_length=ir_length,
            max_reflection_order=0, interpolation="sinc", fir_taps=5,
        )
        sh_rir = shoebox_sh_rir(
            room, src, lis, fs=fs, ir_length=ir_length, max_order=1,
            max_reflection_order=0, interpolation="sinc", fir_taps=5,
        )
        assert_allclose(sh_rir[0], rir / np.sqrt(4 * np.pi), atol=1e-12)

    def test_complex_basis_returns_complex_sh_rir(self):
        room = ShoeboxRoom(dimensions_m=(5.0, 5.0, 3.0), reflection=(0.0,) * 6)
        src = np.array([1.0, 1.0, 1.5])
        lis = np.array([4.0, 4.0, 1.5])
        sh_rir = shoebox_sh_rir(
            room, src, lis, fs=16000.0, ir_length=512, max_order=1,
            basis="complex", max_reflection_order=0,
        )
        assert sh_rir.shape == (4, 512)
        assert np.iscomplexobj(sh_rir)
        assert np.any(np.abs(sh_rir[1:]) > 0.0)

    def test_direct_path_points_at_source(self):
        """The first non-zero sample of the SH-RIR should, after
        inverse-SHT onto a dense scan grid, peak at the direction from
        the listener to the source."""
        from spherical_array_processing.array import fibonacci_grid
        from spherical_array_processing.sh import matrix as sh_matrix
        from spherical_array_processing.types import SHBasisSpec

        room = ShoeboxRoom(
            dimensions_m=(5.0, 5.0, 3.0), reflection=(0.0,) * 6
        )
        src = np.array([1.0, 1.0, 1.5])
        lis = np.array([4.0, 4.0, 1.5])
        sh_rir = shoebox_sh_rir(
            room, src, lis, fs=16000.0, ir_length=512, max_order=3,
            max_reflection_order=0,
        )
        nonzero = np.argwhere(np.any(sh_rir != 0, axis=0))
        assert nonzero.size == 1
        sh_sample = sh_rir[:, int(nonzero[0, 0])]
        scan = fibonacci_grid(4000)
        y_scan = np.asarray(
            sh_matrix(SHBasisSpec(max_order=3, basis="real"), scan)
        )
        spatial = y_scan @ sh_sample
        peak = int(np.argmax(spatial))
        # fibonacci_grid returns az_colat (colatitude measured from +z),
        # so Cartesian is (sin·cos, sin·sin, cos).
        u_peak = np.array(
            [
                np.sin(scan.angle2[peak]) * np.cos(scan.azimuth[peak]),
                np.sin(scan.angle2[peak]) * np.sin(scan.azimuth[peak]),
                np.cos(scan.angle2[peak]),
            ]
        )
        expected = (src - lis) / np.linalg.norm(src - lis)
        assert u_peak @ expected > 0.95  # within ~18° of the ideal direction
