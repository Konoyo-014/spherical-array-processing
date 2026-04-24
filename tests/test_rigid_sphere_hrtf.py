"""Tests for `spherical_array_processing.hrtf.rigid_sphere_hrtf`."""

from __future__ import annotations

import numpy as np
import pytest

from spherical_array_processing.array.simulation import simulate_sh_array_response
from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.coords import cart_to_sph
from spherical_array_processing.hrtf import HRTFDataset, rigid_sphere_hrtf
from spherical_array_processing.types import ArrayGeometry, SphericalGrid


def _ears_on_sphere(head_r: float) -> np.ndarray:
    """Standard ±y ear positions on a sphere of the given radius."""
    return np.array([[0.0, +head_r, 0.0], [0.0, -head_r, 0.0]])


class TestRigidSphereHrtf:
    def test_returns_valid_dataset(self):
        ds = rigid_sphere_hrtf(
            head_radius_m=0.085,
            ear_positions_m=_ears_on_sphere(0.085),
            source_grid=fibonacci_grid(50),
            fs=16000.0,
            n_taps=256,
            max_order=20,
        )
        assert isinstance(ds, HRTFDataset)
        assert ds.hrirs.shape == (50, 2, 256)
        assert ds.fs == 16000.0
        assert ds.ear_positions_m is not None
        assert "head_radius_m" in ds.metadata

    def test_ipsilateral_ear_is_louder(self):
        head_r = 0.085
        ears = _ears_on_sphere(head_r)
        # Source at +y (same side as the left ear).
        src = SphericalGrid(
            azimuth=np.array([np.pi / 2]),
            angle2=np.array([np.pi / 2]),
            convention="az_colat",
        )
        ds = rigid_sphere_hrtf(head_r, ears, src, 48000.0, 512, max_order=30)
        left_e = float(np.sum(ds.hrirs[0, 0] ** 2))
        right_e = float(np.sum(ds.hrirs[0, 1] ** 2))
        assert left_e > 2.0 * right_e  # head shadow factor ~3.4 in practice

    def test_itd_matches_woodworth(self):
        """Peak-difference ITD must match the Woodworth
        ``(r/c)(θ + sin θ)`` estimate within 20% for θ=90°."""
        head_r = 0.085
        ears = _ears_on_sphere(head_r)
        src = SphericalGrid(
            azimuth=np.array([np.pi / 2]),
            angle2=np.array([np.pi / 2]),
            convention="az_colat",
        )
        fs = 48000.0
        ds = rigid_sphere_hrtf(head_r, ears, src, fs, 512, max_order=30)
        peak_l = int(np.argmax(np.abs(ds.hrirs[0, 0])))
        peak_r = int(np.argmax(np.abs(ds.hrirs[0, 1])))
        itd_s = (peak_r - peak_l) / fs
        woodworth = (head_r / 343.0) * (np.pi / 2 + 1)
        assert abs(itd_s - woodworth) < 0.2 * woodworth

    def test_mirror_symmetry(self):
        """Swapping source from +y to -y must swap left/right ears."""
        head_r = 0.085
        ears = _ears_on_sphere(head_r)
        src_ly = SphericalGrid(
            azimuth=np.array([np.pi / 2]),
            angle2=np.array([np.pi / 2]),
            convention="az_colat",
        )
        src_ry = SphericalGrid(
            azimuth=np.array([3 * np.pi / 2]),
            angle2=np.array([np.pi / 2]),
            convention="az_colat",
        )
        ds_l = rigid_sphere_hrtf(head_r, ears, src_ly, 48000.0, 256, max_order=25)
        ds_r = rigid_sphere_hrtf(head_r, ears, src_ry, 48000.0, 256, max_order=25)
        np.testing.assert_allclose(ds_l.hrirs[0, 0], ds_r.hrirs[0, 1], atol=1e-12)
        np.testing.assert_allclose(ds_l.hrirs[0, 1], ds_r.hrirs[0, 0], atol=1e-12)

    def test_frequency_response_matches_modal_simulator(self):
        head_r = 0.085
        ears = _ears_on_sphere(head_r)
        grid = fibonacci_grid(40)
        fft_len = 256
        fs = 16000.0
        max_order = 20
        ds = rigid_sphere_hrtf(
            head_radius_m=head_r,
            ear_positions_m=ears,
            source_grid=grid,
            fs=fs,
            n_taps=fft_len,
            max_order=max_order,
        )

        ear_az, ear_colat, _ = cart_to_sph(
            ears[:, 0], ears[:, 1], ears[:, 2], convention="az_colat"
        )
        ear_grid = SphericalGrid(
            azimuth=ear_az,
            angle2=ear_colat,
            convention="az_colat",
        )
        geom = ArrayGeometry(radius_m=head_r, sensor_grid=ear_grid)
        _, h_modal = simulate_sh_array_response(
            fft_len,
            fs,
            geom,
            grid,
            max_order=max_order,
            array_type="rigid",
        )
        h_modal = np.transpose(h_modal, (0, 2, 1))
        h_modal[0] = np.real(h_modal[0])
        if fft_len % 2 == 0:
            h_modal[-1] = np.real(h_modal[-1])

        h_unshifted = np.fft.rfft(np.fft.ifftshift(ds.hrirs, axes=-1), n=fft_len, axis=-1)
        h_unshifted = h_unshifted.transpose(2, 0, 1)
        np.testing.assert_allclose(h_unshifted, h_modal, atol=1e-10, rtol=1e-10)

    def test_rejects_ears_off_sphere(self):
        with pytest.raises(ValueError, match="sphere surface"):
            rigid_sphere_hrtf(
                head_radius_m=0.085,
                ear_positions_m=np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
                source_grid=fibonacci_grid(10),
                fs=16000.0,
                n_taps=128,
            )

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            rigid_sphere_hrtf(
                head_radius_m=0.085,
                ear_positions_m=np.zeros((3, 3)),
                source_grid=fibonacci_grid(10),
                fs=16000.0,
                n_taps=128,
            )

    def test_feeds_binaural_pipeline(self):
        """Full integration: analytic HRTF → ambi_to_binaural pipeline."""
        from spherical_array_processing.binaural import ambi_to_binaural_time_domain
        from spherical_array_processing.sh import matrix as sh_matrix
        from spherical_array_processing.types import SHBasisSpec

        head_r = 0.085
        ears = _ears_on_sphere(head_r)
        grid = fibonacci_grid(200)
        ds = rigid_sphere_hrtf(head_r, ears, grid, 16000.0, 256, max_order=25)
        # Encode a plane wave at +y (left side) as order-3 ACN real SH.
        src = SphericalGrid(
            azimuth=np.array([np.pi / 2]),
            angle2=np.array([np.pi / 2]),
            convention="az_colat",
        )
        y = np.asarray(
            sh_matrix(SHBasisSpec(max_order=3, basis="real"), src)
        )[0]
        rng = np.random.default_rng(0)
        t_env = rng.standard_normal(4000) * np.hanning(4000) * 0.2
        sh_signal = y[:, None] * t_env[None, :]
        out = ambi_to_binaural_time_domain(
            sh_signal, ds, max_order=3, n_iterations=5,
        )
        left_e = float(np.sum(out[0] ** 2))
        right_e = float(np.sum(out[1] ** 2))
        # Analytic rigid-sphere gives a gentler shadow than the synthetic
        # "clip(u_x, 0, 1)" toy model — 1.3× is still clearly audible.
        assert left_e > 1.3 * right_e
