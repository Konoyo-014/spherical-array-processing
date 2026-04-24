"""Tests for active / reactive ambisonic intensity decomposition."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.ambi import (
    convert_ambi_normalization,
    doa_from_intensity,
    encode_plane_wave,
    intensity_vector,
)
from spherical_array_processing.types import SphericalGrid


def _plane_wave_foa_stft(az_deg: float, T: int = 512) -> np.ndarray:
    az = np.radians(az_deg)
    grid = SphericalGrid(
        azimuth=np.array([az]),
        angle2=np.array([np.pi / 2.0]),
        convention="az_colat",
    )
    sig = np.sin(2 * np.pi * 100 * np.arange(T) / 16000.0)
    foa = encode_plane_wave(sig, grid, max_order=1, basis="real")
    return foa[None, :, :]  # (1, 4, T)


class TestIntensityVector:
    def test_shape_matches_coeff_axis_replacement(self):
        foa_stft = _plane_wave_foa_stft(30.0)
        I = intensity_vector(foa_stft, coeff_axis=1)
        assert I.shape == (1, 3, foa_stft.shape[-1])

    def test_return_reactive_tuple(self):
        foa_stft = _plane_wave_foa_stft(30.0)
        I_act, I_re = intensity_vector(
            foa_stft, coeff_axis=1, return_reactive=True,
        )
        assert I_act.shape == I_re.shape

    def test_real_plane_wave_has_zero_reactive(self):
        """A single real-valued propagating plane wave should have
        purely-active intensity (reactive part is identically zero)."""
        foa_stft = _plane_wave_foa_stft(45.0)
        I_act, I_re = intensity_vector(
            foa_stft, coeff_axis=1, return_reactive=True,
        )
        assert_allclose(I_re, 0.0, atol=1e-12)
        # Active part should be non-zero.
        assert float(np.sum(I_act ** 2)) > 0.0

    def test_standing_wave_has_large_reactive(self):
        """Two counter-propagating plane waves from ±x form a standing
        wave, whose active intensity cancels but whose reactive part
        (with complex spectral phasing) does not."""
        T = 512
        freq_hz = 3 * 16000.0 / T  # Bin-centered to avoid FFT leakage.
        sig_a = np.cos(2 * np.pi * freq_hz * np.arange(T) / 16000.0)
        sig_b = np.cos(2 * np.pi * freq_hz * np.arange(T) / 16000.0 + np.pi / 2)
        grid = SphericalGrid(
            azimuth=np.array([0.0, np.pi]),
            angle2=np.array([np.pi / 2.0, np.pi / 2.0]),
            convention="az_colat",
        )
        foa = encode_plane_wave(
            np.stack([sig_a, sig_b], axis=0), grid, max_order=1,
        )
        # FFT to get complex spectrum (simple 1-frame STFT).
        foa_f = np.fft.rfft(foa, axis=-1)[..., None]  # (4, F, 1)
        foa_stft = np.moveaxis(foa_f, 0, 1)           # (F, 4, 1)
        I_act, I_re = intensity_vector(
            foa_stft, coeff_axis=1, return_reactive=True,
        )
        reactive_energy = float(np.sum(I_re ** 2))
        active_energy = float(np.sum(I_act ** 2))
        assert reactive_energy > active_energy

    def test_normalization_invariance(self):
        foa_stft = _plane_wave_foa_stft(30.0)
        I_ortho = intensity_vector(
            foa_stft, coeff_axis=1, normalization="orthonormal",
        )
        foa_sn3d = convert_ambi_normalization(
            foa_stft, max_order=1,
            from_="orthonormal", to="sn3d", axis=1,
        )
        I_sn3d = intensity_vector(
            foa_sn3d, coeff_axis=1, normalization="sn3d",
        )
        assert_allclose(I_sn3d, I_ortho, atol=1e-12)

    def test_rejects_less_than_4_channels(self):
        with pytest.raises(ValueError, match="at least 4"):
            intensity_vector(np.zeros((3, 100)), coeff_axis=0)

    def test_default_backward_compat_is_coefficient_space_intensity(self):
        """Default ``physical_units=False`` must keep the historical
        coefficient-space formula ``Re{W^* · (X, Y, Z)}`` exactly —
        no ``1/√3`` velocity scaling.  The b14 consolidation
        intentionally left this path untouched; this test locks the
        contract so any future refactor can't silently change it."""
        foa_stft = _plane_wave_foa_stft(30.0)
        got = intensity_vector(
            foa_stft, coeff_axis=1, normalization="orthonormal",
        )
        # Historical formula: direct ACN → Cartesian, no velocity rescale.
        w = foa_stft[:, 0, :]
        y = foa_stft[:, 1, :]
        z = foa_stft[:, 2, :]
        x = foa_stft[:, 3, :]
        w_conj = np.conj(w)
        expected = np.stack(
            [np.real(w_conj * x), np.real(w_conj * y), np.real(w_conj * z)],
            axis=0,
        )
        # got has shape (F, 3, T); expected has shape (3, F, T).
        np.testing.assert_allclose(
            got, np.moveaxis(expected, 0, 1), atol=0.0,
        )

    def test_physical_units_matches_dirac_internal_intensity(self):
        """``intensity_vector(physical_units=True)`` must produce the
        same intensity vectors that ``dirac_analysis`` builds internally
        — both pipelines share the ``_canonical_foa_pv`` helper, so
        they cannot drift apart."""
        from spherical_array_processing.ambi.intensity import _canonical_foa_pv
        foa_stft = _plane_wave_foa_stft(37.0)
        # Shared canonicalisation result.
        w, vx, vy, vz = _canonical_foa_pv(
            foa_stft, normalization="orthonormal", coeff_axis=1,
        )
        w_conj = np.conj(w)
        expected_i = np.stack(
            [np.real(w_conj * vx),
             np.real(w_conj * vy),
             np.real(w_conj * vz)],
            axis=0,
        )
        got = intensity_vector(
            foa_stft, coeff_axis=1, physical_units=True,
        )
        # got has shape (1, 3, T); expected_i has shape (3, 1, T).
        np.testing.assert_allclose(
            got, np.moveaxis(expected_i, 0, 1), atol=1e-14,
        )

    def test_physical_units_plane_wave_intensity_equals_energy(self):
        """For a coherent plane wave in physical units, ``||I|| = E``,
        so DirAC ψ = 0.  This is the textbook DirAC limit."""
        foa_stft = _plane_wave_foa_stft(45.0)
        I_phys = intensity_vector(
            foa_stft, coeff_axis=1, physical_units=True,
        )
        # Energy in physical units.
        from spherical_array_processing.ambi.intensity import _canonical_foa_pv
        w, vx, vy, vz = _canonical_foa_pv(
            foa_stft, normalization="orthonormal", coeff_axis=1,
        )
        energy = 0.5 * (
            np.abs(w) ** 2 + np.abs(vx) ** 2
            + np.abs(vy) ** 2 + np.abs(vz) ** 2
        )
        # Move Cartesian axis to last for norm.
        I_mag = np.linalg.norm(
            np.moveaxis(I_phys, 1, -1), axis=-1,
        )
        # Interior (non-transient) samples: ||I|| should equal E.
        np.testing.assert_allclose(
            I_mag[:, 20:], energy[:, 20:], atol=1e-14,
        )


class TestDoaFromIntensity:
    @pytest.mark.parametrize("az_deg", [0.0, 45.0, 90.0, 180.0, -60.0])
    def test_doa_matches_source_direction(self, az_deg):
        foa_stft = _plane_wave_foa_stft(az_deg)
        doa = doa_from_intensity(foa_stft, coeff_axis=1)
        mean_doa = doa.mean(axis=-1)[0]
        az = np.radians(az_deg)
        expected = np.array([np.cos(az), np.sin(az), 0.0])
        cos_sim = float(
            mean_doa @ expected
            / max(np.linalg.norm(mean_doa), 1e-12)
            / max(np.linalg.norm(expected), 1e-12)
        )
        assert cos_sim > 0.99

    def test_zero_energy_bin_returns_zero(self):
        foa_stft = np.zeros((1, 4, 10))
        doa = doa_from_intensity(foa_stft, coeff_axis=1)
        assert_allclose(doa, 0.0, atol=0)

    def test_doa_is_unit_norm(self):
        foa_stft = _plane_wave_foa_stft(20.0)
        doa = doa_from_intensity(foa_stft, coeff_axis=1)
        # Move Cartesian axis to -1 for norm.
        doa_last = np.moveaxis(doa, 1, -1)
        norms = np.linalg.norm(doa_last, axis=-1)
        # Bins with positive intensity should have unit norm.
        mask = norms > 1e-6
        assert_allclose(norms[mask], 1.0, atol=1e-12)
