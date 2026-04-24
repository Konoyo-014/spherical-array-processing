"""Tests for the DirAC analysis / synthesis module (``sap.dirac``)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.coords import unit_sph_to_cart
from spherical_array_processing.dirac import DirACParameters, dirac_analysis, dirac_synthesize
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


def _plane_wave_stft(az_rad, col_rad, n_bins=64, n_frames=32, seed=0):
    spec = SHBasisSpec(max_order=1, basis="real")
    src = SphericalGrid(azimuth=[az_rad], angle2=[col_rad], convention="az_colat")
    y = np.asarray(sh_matrix(spec, src))[0]  # (4,)
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n_bins, n_frames)) + 1j * rng.normal(
        size=(n_bins, n_frames)
    )
    stft = np.einsum("ft,q->fqt", signal, y)
    return stft, y


class TestDirACAnalysis:
    def test_plane_wave_direction_recovered_exactly(self):
        az, col = np.radians(50.0), np.radians(70.0)
        stft, _ = _plane_wave_stft(az, col)
        freqs = np.arange(stft.shape[0]) * 16000.0 / stft.shape[0]
        params = dirac_analysis(stft, freqs, smoothing_alpha=0.3, coeff_axis=1)
        u_true = unit_sph_to_cart(
            np.array([az]), np.array([col]), convention="az_colat"
        )[0]
        # Skip a few IIR-transient frames at the start, average over F
        # and T for a robust single-source estimate.
        dir_mean = params.direction_xyz[:, 5:, :].mean(axis=(0, 1))
        dir_mean /= np.linalg.norm(dir_mean)
        cos_sep = np.clip(dir_mean @ u_true, -1.0, 1.0)
        assert np.degrees(np.arccos(cos_sep)) < 2.0

    def test_plane_wave_diffuseness_is_low(self):
        az, col = np.radians(30.0), np.radians(45.0)
        stft, _ = _plane_wave_stft(az, col)
        freqs = np.arange(stft.shape[0]) * 16000.0 / stft.shape[0]
        params = dirac_analysis(stft, freqs, smoothing_alpha=0.3, coeff_axis=1)
        # A coherent plane wave should collapse to ψ = 0 up to floating
        # point noise, not merely "low" diffuseness.
        assert float(params.diffuseness[:, 5:].mean()) < 1e-12

    def test_diffuse_field_diffuseness_is_high(self):
        rng = np.random.default_rng(42)
        stft = rng.normal(size=(64, 4, 64)) + 1j * rng.normal(size=(64, 4, 64))
        freqs = np.arange(64) * 16000.0 / 64
        params = dirac_analysis(stft, freqs, smoothing_alpha=0.1, coeff_axis=1)
        assert float(params.diffuseness[:, 5:].mean()) > 0.6

    def test_rejects_non_foa_input(self):
        with pytest.raises(ValueError, match="at least 4"):
            # Shape (F=16, Q=3, T=4) — only 3 SH channels on the
            # default coeff_axis=-2, which is less than the required 4.
            dirac_analysis(np.zeros((16, 3, 4), dtype=complex), np.arange(16))

    def test_rejects_freq_axis_as_coeff(self):
        with pytest.raises(ValueError, match="frequency axis"):
            dirac_analysis(
                np.zeros((16, 4, 4), dtype=complex), np.arange(16), coeff_axis=0
            )

    def test_normalization_invariance_psi_and_doa(self):
        """Feeding the same physical field in orthonormal, N3D or
        SN3D (with `normalization` declared) must give **identical**
        DOA and diffuseness — not just "close"."""
        from spherical_array_processing.ambi import (
            convert_ambi_normalization, encode_plane_wave,
        )
        from spherical_array_processing.stft import stft
        from spherical_array_processing.types import SphericalGrid
        T = 4096
        fs = 16000.0
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(T) * 0.3
        grid = SphericalGrid(
            azimuth=np.array([np.pi / 4]),
            angle2=np.array([np.pi / 2]),
            convention="az_colat",
        )
        foa_ortho = encode_plane_wave(
            sig, grid, max_order=1, normalization="orthonormal",
        )
        foa_sn3d = convert_ambi_normalization(
            foa_ortho, max_order=1,
            from_="orthonormal", to="sn3d", axis=0,
        )
        foa_n3d = convert_ambi_normalization(
            foa_ortho, max_order=1,
            from_="orthonormal", to="n3d", axis=0,
        )
        freqs, _, z_ortho = stft(foa_ortho, fs, nperseg=512)
        _, _, z_sn3d = stft(foa_sn3d, fs, nperseg=512)
        _, _, z_n3d = stft(foa_n3d, fs, nperseg=512)
        p_ortho = dirac_analysis(z_ortho, freqs, normalization="orthonormal")
        p_sn3d = dirac_analysis(z_sn3d, freqs, normalization="sn3d")
        p_n3d = dirac_analysis(z_n3d, freqs, normalization="n3d")
        assert float(p_ortho.diffuseness[:, 5:].mean()) < 1e-12
        # Diffuseness identical across declared normalisations.
        np.testing.assert_allclose(
            p_sn3d.diffuseness, p_ortho.diffuseness, atol=1e-12,
        )
        np.testing.assert_allclose(
            p_n3d.diffuseness, p_ortho.diffuseness, atol=1e-12,
        )
        # DOA identical too.
        np.testing.assert_allclose(
            p_sn3d.direction_xyz, p_ortho.direction_xyz, atol=1e-12,
        )


class TestDirACSynthesis:
    def test_output_shape_and_peaks_at_source_speaker(self):
        az, col = np.radians(60.0), np.radians(80.0)
        stft, _ = _plane_wave_stft(az, col, n_bins=32, n_frames=16)
        freqs = np.arange(stft.shape[0]) * 16000.0 / stft.shape[0]
        params = dirac_analysis(stft, freqs, smoothing_alpha=0.4, coeff_axis=1)
        spk = fibonacci_grid(20)
        out = dirac_synthesize(
            params, spk, decorrelate_diffuse=False, rng=np.random.default_rng(0)
        )
        assert out.shape == (stft.shape[0], spk.size, stft.shape[2])
        # The speaker nearest the source direction should carry the
        # largest average energy over time.
        energies = np.mean(np.abs(out) ** 2, axis=(0, 2))  # (L,)
        u_src = unit_sph_to_cart(
            np.array([az]), np.array([col]), convention="az_colat"
        )[0]
        spk_xyz = unit_sph_to_cart(
            spk.azimuth, spk.angle2, convention=spk.convention
        )
        nearest = int(np.argmax(spk_xyz @ u_src))
        assert int(np.argmax(energies)) == nearest

    def test_dirac_render_time_domain_peaks_near_source(self):
        """Time-domain DirAC rendering of a mono plane-wave input
        should concentrate energy on the speaker closest to the
        source direction."""
        from spherical_array_processing.dirac import dirac_render_time_domain

        fs = 16000.0
        duration_s = 0.4
        t = np.arange(int(fs * duration_s), dtype=float) / fs
        rng = np.random.default_rng(0)
        signal = rng.normal(size=t.size) * np.hanning(t.size) * 0.3

        spec = SHBasisSpec(max_order=1, basis="real")
        az_src, col_src = np.radians(70.0), np.radians(60.0)
        src = SphericalGrid(
            azimuth=[az_src], angle2=[col_src], convention="az_colat"
        )
        y = np.asarray(sh_matrix(spec, src))[0]  # (4,)
        ambi = np.outer(y, signal)  # (4, T)

        spk = fibonacci_grid(16)
        out = dirac_render_time_domain(
            ambi, fs, spk, nperseg=512, decorrelate_diffuse=False,
            rng=np.random.default_rng(0),
        )
        assert out.shape[0] == spk.size
        energies = np.sqrt(np.mean(out ** 2, axis=1))

        src_u = unit_sph_to_cart(
            np.array([az_src]), np.array([col_src]), convention="az_colat"
        )[0]
        spk_u = unit_sph_to_cart(
            spk.azimuth, spk.angle2, convention=spk.convention
        )
        nearest = int(np.argmax(spk_u @ src_u))
        assert int(np.argmax(energies)) == nearest

    def test_dirac_render_time_domain_accepts_short_channels_first_input(self):
        """Regression: short signals with shape ``(Q, T)`` and ``Q > T``
        must not be mistaken for ``(T, Q)`` input."""
        from spherical_array_processing.dirac import dirac_render_time_domain

        fs = 16000.0
        spec = SHBasisSpec(max_order=1, basis="real")
        src = SphericalGrid(
            azimuth=[np.radians(30.0)],
            angle2=[np.radians(70.0)],
            convention="az_colat",
        )
        y = np.asarray(sh_matrix(spec, src))[0]
        ambi = np.outer(y, np.array([1.0, 0.5], dtype=float))  # (Q=4, T=2)

        spk = fibonacci_grid(8)
        out = dirac_render_time_domain(
            ambi,
            fs,
            spk,
            nperseg=2,
            decorrelate_diffuse=False,
            rng=np.random.default_rng(0),
        )

        assert out.shape[0] == spk.size
        assert out.shape[1] > 0

    def test_diffuse_synthesis_spreads_energy(self):
        """For a fully isotropic input the synthesised loudspeaker
        energy should be roughly uniform across speakers (up to the
        decorrelation phase sampling)."""
        rng = np.random.default_rng(1)
        stft = rng.normal(size=(32, 4, 64)) + 1j * rng.normal(size=(32, 4, 64))
        freqs = np.arange(32) * 16000.0 / 32
        params = dirac_analysis(stft, freqs, smoothing_alpha=0.1, coeff_axis=1)
        spk = fibonacci_grid(16)
        out = dirac_synthesize(
            params, spk, decorrelate_diffuse=True, rng=np.random.default_rng(2)
        )
        energies = np.mean(np.abs(out) ** 2, axis=(0, 2))
        assert energies.std() / energies.mean() < 0.5
