"""Tests for the `spherical_array_processing.binaural` module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.binaural import magls_binaural_filters
from spherical_array_processing.coords import unit_sph_to_cart
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec, SphericalGrid


def _synthetic_hrtfs(
    n_order: int, fs: float, fft_len: int, grid: SphericalGrid
) -> tuple[np.ndarray, np.ndarray]:
    """Build a toy two-ear HRTF dataset: pure delay + soft head shadow."""
    freqs = np.arange(fft_len // 2 + 1, dtype=float) * fs / fft_len
    u = unit_sph_to_cart(grid.azimuth, grid.angle2, convention=grid.convention)
    c = 343.0
    ear_offset = 0.09
    tau_L = u @ np.array([-ear_offset, 0.0, 0.0]) / c
    tau_R = u @ np.array([ear_offset, 0.0, 0.0]) / c
    shadow_L = 1.0 - 0.5 * np.clip(u[:, 0], 0.0, 1.0)
    shadow_R = 1.0 - 0.5 * np.clip(-u[:, 0], 0.0, 1.0)
    hrtfs = np.zeros((freqs.size, grid.size, 2), dtype=np.complex128)
    for fi, f in enumerate(freqs):
        hrtfs[fi, :, 0] = shadow_L * np.exp(-2j * np.pi * f * tau_L)
        hrtfs[fi, :, 1] = shadow_R * np.exp(-2j * np.pi * f * tau_R)
    return freqs, hrtfs


class TestMagls:
    @pytest.fixture
    def setup(self):
        fs = 48000.0
        fft_len = 512
        n_order = 4
        grid = fibonacci_grid(180)
        freqs, hrtfs = _synthetic_hrtfs(n_order, fs, fft_len, grid)
        return grid, n_order, freqs, hrtfs

    def test_shape(self, setup):
        grid, n_order, freqs, hrtfs = setup
        filters = magls_binaural_filters(hrtfs, freqs, grid, n_order)
        q = (n_order + 1) ** 2
        assert filters.shape == (freqs.size, q, 2)
        assert np.issubdtype(filters.dtype, np.complexfloating)

    def test_complex_ls_regime_matches_pinv(self, setup):
        """Below ``f_cut`` MagLS coincides with ``pinv(Y) @ H`` per bin."""
        grid, n_order, freqs, hrtfs = setup
        filters = magls_binaural_filters(
            hrtfs, freqs, grid, n_order, f_cut_hz=1500.0
        )
        spec = SHBasisSpec(
            max_order=n_order, basis="real", angle_convention=grid.convention
        )
        y = np.asarray(sh_matrix(spec, grid))
        pinv_y = np.linalg.pinv(y)
        low_bins = np.where(freqs < 1500.0)[0]
        for fi in low_bins:
            for ear in (0, 1):
                expected = pinv_y @ hrtfs[fi, :, ear]
                assert_allclose(filters[fi, :, ear], expected, atol=1e-10)

    def test_magls_improves_magnitude_match_above_cutoff(self, setup):
        """For high frequencies, MagLS should give a smaller magnitude
        error than the complex LS fit at the same bin."""
        grid, n_order, freqs, hrtfs = setup
        spec = SHBasisSpec(
            max_order=n_order, basis="real", angle_convention=grid.convention
        )
        y = np.asarray(sh_matrix(spec, grid))
        pinv_y = np.linalg.pinv(y)
        filters = magls_binaural_filters(
            hrtfs, freqs, grid, n_order, f_cut_hz=1500.0
        )
        high_bins = np.where(freqs >= 1500.0)[0]
        improvements = 0
        trials = 0
        for fi in high_bins[::5]:  # every fifth bin to keep the test quick
            for ear in (0, 1):
                complex_ls = pinv_y @ hrtfs[fi, :, ear]
                mag_err_cls = np.mean(
                    np.abs(np.abs(y @ complex_ls) - np.abs(hrtfs[fi, :, ear]))
                )
                mag_err_magls = np.mean(
                    np.abs(np.abs(y @ filters[fi, :, ear]) - np.abs(hrtfs[fi, :, ear]))
                )
                if mag_err_magls <= mag_err_cls + 1e-12:
                    improvements += 1
                trials += 1
        # MagLS is an alternating-projection magnitude-minimiser: it can
        # (rarely) tie with complex LS but should not lose on average.
        assert improvements >= 0.8 * trials, (
            f"MagLS improved magnitude on {improvements}/{trials} bins; expected ≥ 80 %"
        )

    def test_input_shape_validation(self, setup):
        grid, n_order, freqs, hrtfs = setup
        with pytest.raises(ValueError, match="hrtfs must have shape"):
            magls_binaural_filters(hrtfs[..., 0], freqs, grid, n_order)
        with pytest.raises(ValueError, match="freqs_hz length"):
            magls_binaural_filters(hrtfs, freqs[:-10], grid, n_order)
        with pytest.raises(ValueError, match="spatial axis"):
            bad_grid = fibonacci_grid(grid.size + 5)
            magls_binaural_filters(hrtfs, freqs, bad_grid, n_order)

    def test_bimagls_shape_and_delay_shapes(self, setup):
        from spherical_array_processing.binaural import bimagls_binaural_filters

        grid, n_order, freqs, hrtfs = setup
        ear_positions = np.array([[-0.09, 0.0, 0.0], [0.09, 0.0, 0.0]])
        filters, delay_sh = bimagls_binaural_filters(
            hrtfs, freqs, grid, n_order, ear_positions_m=ear_positions
        )
        q = (n_order + 1) ** 2
        assert filters.shape == (freqs.size, q, 2)
        assert delay_sh.shape == (q, 2)
        # Left / right delays must have opposite dominant sign in the
        # first-order dipole channels because ear positions are mirror
        # images along the x-axis.
        assert np.sign(delay_sh[:, 0]).sum() != np.sign(delay_sh[:, 1]).sum()

    def test_bimagls_lf_plus_delay_recovers_hrtf(self, setup):
        """Below the cutoff frequency, BiMagLS is complex-LS fit on
        delay-aligned HRTFs.  Reattaching the delay must reconstruct
        the original HRTF to within the SH truncation error of the
        fit.
        """
        from spherical_array_processing.binaural import bimagls_binaural_filters

        grid, n_order, freqs, hrtfs = setup
        ear_positions = np.array([[-0.09, 0.0, 0.0], [0.09, 0.0, 0.0]])
        filters, delay_sh = bimagls_binaural_filters(
            hrtfs, freqs, grid, n_order, ear_positions_m=ear_positions,
            f_cut_hz=2000.0,  # keep the LF regime broad for this test
        )
        spec = SHBasisSpec(
            max_order=n_order, basis="real", angle_convention=grid.convention
        )
        y = np.asarray(sh_matrix(spec, grid))
        delays_grid = y @ delay_sh  # (G, 2)
        # Reattach delay and project back to the grid.
        lf_bins = np.where(freqs < 2000.0)[0]
        recon_mag_err = []
        for ear in (0, 1):
            recon_aligned = np.einsum(
                "fq,gq->fg", filters[:, :, ear], y
            )
            phase = np.exp(
                -1j * 2.0 * np.pi * freqs[:, None] * delays_grid[None, :, ear]
            )
            recon = recon_aligned * phase
            err = np.max(
                np.abs(np.abs(recon[lf_bins]) - np.abs(hrtfs[lf_bins, :, ear]))
            )
            recon_mag_err.append(err)
        # The LS fit won't be bit-exact at N=4 for this HRTF grid, but
        # the magnitude error should be moderate.
        assert max(recon_mag_err) < 0.2, (
            f"BiMagLS LF reconstruction magnitude error {recon_mag_err} "
            "larger than expected"
        )

    def test_bimagls_rejects_bad_ear_positions_shape(self, setup):
        from spherical_array_processing.binaural import bimagls_binaural_filters

        grid, n_order, freqs, hrtfs = setup
        with pytest.raises(ValueError, match="ear_positions_m"):
            bimagls_binaural_filters(
                hrtfs, freqs, grid, n_order,
                ear_positions_m=np.array([0.09, 0.0, 0.0]),  # 1-D, bad
            )

    def test_bimagls_complex_basis_preserves_complex_delay_coeffs(self):
        from spherical_array_processing.binaural import bimagls_binaural_filters

        fs = 48000.0
        fft_len = 128
        n_order = 1
        grid = fibonacci_grid(180)
        freqs = np.arange(fft_len // 2 + 1, dtype=float) * fs / fft_len
        directions = unit_sph_to_cart(
            grid.azimuth, grid.angle2, convention=grid.convention
        )
        ear_positions = np.array([[-0.09, 0.05, 0.02], [0.09, -0.05, -0.02]])
        delays = directions @ ear_positions.T / 343.0
        hrtfs = np.exp(-2j * np.pi * freqs[:, None, None] * delays[None, :, :])

        _, delay_sh = bimagls_binaural_filters(
            hrtfs, freqs, grid, n_order, ear_positions_m=ear_positions, basis="complex"
        )

        spec = SHBasisSpec(
            max_order=n_order, basis="complex", angle_convention=grid.convention
        )
        y = np.asarray(sh_matrix(spec, grid))
        rebuilt = y @ delay_sh
        assert np.iscomplexobj(delay_sh)
        assert_allclose(rebuilt, delays, atol=1e-10)

    def test_phase_continuation_actually_changes_hf_result(self, setup):
        """Regression test: codex b2 review found the
        ``phase_continuation`` option was silently a no-op because the
        seeded phase was overwritten inside the alternating loop before
        the first LS solve.  This test asserts the two branches now
        diverge above the cutoff.
        """
        grid, n_order, freqs, hrtfs = setup
        f_with = magls_binaural_filters(
            hrtfs, freqs, grid, n_order, f_cut_hz=1500.0, phase_continuation=True
        )
        f_without = magls_binaural_filters(
            hrtfs, freqs, grid, n_order, f_cut_hz=1500.0, phase_continuation=False
        )
        lf = freqs < 1500.0
        hf = freqs >= 1500.0
        # Below cutoff the two solutions are identical by construction.
        assert_allclose(f_with[lf], f_without[lf], atol=1e-12)
        # Above cutoff the phase-continuation path must produce a
        # nontrivially different filter.
        assert np.max(np.abs(f_with[hf] - f_without[hf])) > 1e-3
