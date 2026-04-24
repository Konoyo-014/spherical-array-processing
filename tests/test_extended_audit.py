"""Extended audit tests — covers experimental modules, repro layers,
simulation, plotting, additional stress tests, and API consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose


# ---------------------------------------------------------------------------
# 1.  Top-level convenience: sap.sh, sap.array, etc. accessible
# ---------------------------------------------------------------------------


class TestSubmoduleAccess:
    def test_sap_sh_accessible(self):
        import spherical_array_processing as sap
        Y = sap.sh.real_matrix(
            sap.SHBasisSpec(max_order=1),
            sap.array.fibonacci_grid(10),
        )
        assert Y.shape == (10, 4)

    def test_sap_coords_accessible(self):
        import spherical_array_processing as sap
        x, y, z = sap.coords.sph_to_cart(0.0, 0.0)
        assert_allclose(x, 1.0, atol=1e-14)

    def test_sap_beamforming_accessible(self):
        import spherical_array_processing as sap
        w = sap.beamforming.beam_weights_hypercardioid(2)
        assert w.shape == (3,)

    def test_gauss_legendre_alias(self):
        import spherical_array_processing as sap
        g1 = sap.array.equiangle_sampling(2)
        g2 = sap.array.gauss_legendre_sampling(2)
        assert_allclose(g1.azimuth, g2.azimuth)
        assert_allclose(g1.weights, g2.weights)


# ---------------------------------------------------------------------------
# 2.  Simulation module
# ---------------------------------------------------------------------------
from spherical_array_processing.array.simulation import simulate_plane_wave_array_response
from spherical_array_processing.types import ArrayGeometry, SphericalGrid


class TestSimulation:
    def _make_geometry(self, n_mics: int = 8) -> ArrayGeometry:
        az = np.linspace(0, 2 * np.pi, n_mics, endpoint=False)
        el = np.zeros(n_mics)
        grid = SphericalGrid(azimuth=az, angle2=el, convention="az_el")
        return ArrayGeometry(radius_m=0.042, sensor_grid=grid)

    def test_shape(self):
        geo = self._make_geometry(8)
        src = SphericalGrid(azimuth=[0.0, 1.0], angle2=[0.0, 0.5], convention="az_el")
        freqs, H = simulate_plane_wave_array_response(1024, 16000, geo, src)
        n_bins = 1024 // 2 + 1
        assert freqs.shape == (n_bins,)
        assert H.shape == (n_bins, 8, 2)

    def test_dc_bin_is_one(self):
        """At f=0 (k=0), all phase factors are 1."""
        geo = self._make_geometry(4)
        src = SphericalGrid(azimuth=[0.5], angle2=[0.3], convention="az_el")
        freqs, H = simulate_plane_wave_array_response(256, 8000, geo, src)
        assert_allclose(H[0, :, 0], 1.0, atol=1e-14)

    def test_unit_magnitude_free_field(self):
        """Free-field plane wave has unit magnitude at all frequencies."""
        geo = self._make_geometry(4)
        src = SphericalGrid(azimuth=[0.0], angle2=[0.0], convention="az_el")
        _, H = simulate_plane_wave_array_response(512, 16000, geo, src)
        mags = np.abs(H[:, :, 0])
        assert_allclose(mags, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3.  Plotting module (non-visual smoke tests)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestPlotting:
    def test_plot_mic_array_runs(self):
        from spherical_array_processing.plotting import plot_mic_array
        dirs = np.array([[0, 0], [90, 0], [180, 0], [270, 0]], dtype=float)
        ax = plot_mic_array(dirs, radius_m=0.05)
        assert ax is not None
        plt.close("all")

    def test_plot_mic_array_validation(self):
        from spherical_array_processing.plotting import plot_mic_array
        with pytest.raises(ValueError, match="\\[M,2\\]"):
            plot_mic_array(np.zeros(5), radius_m=0.05)

    def test_directional_map_runs(self):
        from spherical_array_processing.plotting import plot_directional_map_from_grid
        vals = np.random.default_rng(0).normal(size=37 * 19)
        ax = plot_directional_map_from_grid(vals, azi_res_deg=10, polar_res_deg=10)
        assert ax is not None
        plt.close("all")

    def test_figure_repro_context(self):
        from spherical_array_processing.plotting import figure_repro_context
        old_dpi = matplotlib.rcParams["figure.dpi"]
        with figure_repro_context():
            # Inside context, DPI should have been changed
            pass
        # After context, should be restored
        assert matplotlib.rcParams["figure.dpi"] == old_dpi

    def test_apply_matlab_like_style(self):
        from spherical_array_processing.plotting import apply_matlab_like_style
        apply_matlab_like_style()
        assert matplotlib.rcParams["axes.grid"] is True


# ---------------------------------------------------------------------------
# 4.  Experimental module — stereo → FOA
# ---------------------------------------------------------------------------
from spherical_array_processing.experimental.foa_from_stereo import (
    estimate_incomplete_foa_from_stereo,
    StereoFOAConfig,
    FOAEstimate,
)


class TestExperimentalFOA:
    def _make_stereo(self, n_samples: int = 4096, fs: float = 16000.0):
        rng = np.random.default_rng(42)
        return rng.normal(size=(n_samples, 2)), fs

    def test_basic_output_shape(self):
        stereo, fs = self._make_stereo()
        result = estimate_incomplete_foa_from_stereo(stereo, fs)
        assert isinstance(result, FOAEstimate)
        assert result.foa_stft.shape[0] == 4  # W, X, Y, Z
        assert result.observability_mask.shape == (4,)
        # W and X observable, Y and Z not
        assert result.observability_mask[0] is True or result.observability_mask[0] == True
        assert result.observability_mask[2] is False or result.observability_mask[2] == False

    def test_confidence_range(self):
        stereo, fs = self._make_stereo()
        result = estimate_incomplete_foa_from_stereo(stereo, fs)
        assert np.all(result.confidence >= 0.0)
        assert np.all(result.confidence <= 1.0)

    def test_uncertainty_range(self):
        stereo, fs = self._make_stereo()
        result = estimate_incomplete_foa_from_stereo(stereo, fs)
        assert np.all(result.uncertainty >= 0.0)
        assert np.all(result.uncertainty <= 1.0)

    def test_y_z_channels_zero(self):
        """Y and Z channels should be zero (unobservable)."""
        stereo, fs = self._make_stereo()
        result = estimate_incomplete_foa_from_stereo(stereo, fs)
        assert_allclose(result.foa_stft[2], 0.0)
        assert_allclose(result.foa_stft[3], 0.0)

    def test_custom_config(self):
        stereo, fs = self._make_stereo()
        cfg = StereoFOAConfig(nperseg=512, noverlap=256, ipd_clip_mode="hard")
        result = estimate_incomplete_foa_from_stereo(stereo, fs, config=cfg)
        assert result.metadata["ipd_clip_mode"] == "hard"

    def test_input_validation(self):
        with pytest.raises(ValueError, match="shape"):
            estimate_incomplete_foa_from_stereo(np.zeros(100), 16000)
        with pytest.raises(ValueError, match="shape"):
            estimate_incomplete_foa_from_stereo(np.zeros((100, 3)), 16000)

    def test_silent_input(self):
        """All-zero stereo should not crash."""
        stereo = np.zeros((2048, 2))
        result = estimate_incomplete_foa_from_stereo(stereo, 16000)
        assert np.all(np.isfinite(result.foa_stft))
        assert np.all(np.isfinite(result.confidence))


# ---------------------------------------------------------------------------
# 5.  Regression tooling
# ---------------------------------------------------------------------------
from spherical_array_processing.regression.matlab import (
    detect_matlab,
    detect_octave,
    matlab_available,
)


class TestRegressionTooling:
    def test_detect_functions_do_not_crash(self):
        """These should return None or a runtime object, never crash."""
        m = detect_matlab()
        assert m is None or hasattr(m, "executable")
        o = detect_octave()
        assert o is None or hasattr(o, "executable")

    def test_matlab_available_returns_bool(self):
        assert isinstance(matlab_available(), bool)


# ---------------------------------------------------------------------------
# 6.  Stress tests — large inputs, high orders
# ---------------------------------------------------------------------------
from spherical_array_processing.sh.basis import complex_matrix, real_matrix, acn_index
from spherical_array_processing.array.sampling import fibonacci_grid, gauss_legendre_sampling
from spherical_array_processing.types import SHBasisSpec


class TestStress:
    def test_high_order_sh_basis(self):
        """SH basis at order 10 should produce correct shapes."""
        N = 10
        n_coeffs = (N + 1) ** 2  # 121
        grid = fibonacci_grid(500)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)
        assert Y.shape == (500, n_coeffs)
        assert np.all(np.isfinite(Y))

    def test_gl_high_order_orthogonality(self):
        """GL quadrature should be exact even at order 8."""
        N = 8
        grid = gauss_legendre_sampling(N)
        spec = SHBasisSpec(max_order=N, basis="real", angle_convention="az_colat")
        Y = real_matrix(spec, grid)
        gram = Y.T @ np.diag(grid.weights) @ Y
        assert_allclose(gram, np.eye(spec.n_coeffs), atol=1e-12)

    def test_large_fibonacci_grid(self):
        g = fibonacci_grid(10000)
        assert g.size == 10000
        assert_allclose(g.weights.sum(), 4 * np.pi, atol=1e-12)

    def test_bn_matrix_many_frequencies(self):
        from spherical_array_processing.acoustics.radial import bn_matrix
        kr = np.linspace(0.01, 20.0, 500)
        B = bn_matrix(5, kr, sphere="rigid")
        assert B.shape == (500, 36)
        assert np.all(np.isfinite(B))

    def test_beamforming_high_order(self):
        from spherical_array_processing.beamforming.fixed import (
            beam_weights_hypercardioid,
            beam_weights_maxev,
            axisymmetric_pattern,
        )
        for N in [5, 8, 12]:
            w = beam_weights_hypercardioid(N)
            assert w.shape == (N + 1,)
            w2 = beam_weights_maxev(N)
            front = axisymmetric_pattern(np.array([0.0]), w2)[0]
            assert_allclose(front, 1.0, atol=1e-12)
            theta = np.linspace(0, np.pi, 361)
            p = axisymmetric_pattern(theta, w)
            assert p.shape == (361,)
            # Front should be dominant
            assert np.argmax(p) < 5


# ---------------------------------------------------------------------------
# 7.  Repro subpackage — smoke tests
# ---------------------------------------------------------------------------


class TestRepro:
    def test_rafaely_repro_importable(self):
        from spherical_array_processing.repro import rafaely
        assert hasattr(rafaely, "__name__")

    def test_politis_repro_importable(self):
        from spherical_array_processing.repro import politis
        assert hasattr(politis, "__name__")

    def test_sht_repro_importable(self):
        from spherical_array_processing.repro import sht
        assert hasattr(sht, "__name__")

    def test_array_response_simulator_importable(self):
        from spherical_array_processing.repro import array_response_simulator
        assert hasattr(array_response_simulator, "__name__")


# ---------------------------------------------------------------------------
# 8.  Types — deeper validation
# ---------------------------------------------------------------------------


class TestTypesDeep:
    def test_sh_basis_spec_defaults(self):
        spec = SHBasisSpec(max_order=3)
        assert spec.basis == "complex"
        assert spec.normalization == "orthonormal"
        assert spec.angle_convention == "az_colat"

    def test_spatial_spectrum_result(self):
        from spherical_array_processing.types import SpatialSpectrumResult
        grid = SphericalGrid(azimuth=[0.0, 1.0, 2.0], angle2=[0.0, 0.5, 1.0])
        r = SpatialSpectrumResult(
            spectrum=np.array([1.0, 2.0, 3.0]),
            grid=grid,
            peak_indices=np.array([2]),
            peak_dirs_rad=np.array([[2.0, 1.0]]),
        )
        assert r.spectrum.shape == (3,)
        assert r.peak_indices[0] == 2

    def test_sh_signal_frame(self):
        from spherical_array_processing.types import SHSignalFrame
        f = SHSignalFrame(
            data=np.zeros((4, 9), dtype=complex),
            freqs_hz=np.array([100, 200, 300, 400], dtype=float),
            basis=SHBasisSpec(max_order=2),
        )
        assert f.data.shape == (4, 9)

    def test_sh_covariance(self):
        from spherical_array_processing.types import SHCovariance
        c = SHCovariance(
            data=np.eye(9, dtype=complex),
            freqs_hz=None,
            basis=SHBasisSpec(max_order=2),
        )
        assert c.data.shape == (9, 9)


# ---------------------------------------------------------------------------
# 9.  API consistency checks
# ---------------------------------------------------------------------------


class TestAPIConsistency:
    def test_all_public_functions_have_docstrings(self):
        """Every function in __all__ should have a docstring."""
        import spherical_array_processing.sh as sh_mod
        import spherical_array_processing.acoustics as ac_mod
        import spherical_array_processing.array as arr_mod
        import spherical_array_processing.beamforming as bf_mod
        import spherical_array_processing.doa as doa_mod
        import spherical_array_processing.diffuseness as diff_mod
        import spherical_array_processing.coherence as coh_mod
        import spherical_array_processing.coords as co_mod

        for mod in [sh_mod, ac_mod, arr_mod, bf_mod, doa_mod, diff_mod, coh_mod, co_mod]:
            for name in getattr(mod, "__all__", []):
                obj = getattr(mod, name)
                if callable(obj):
                    assert obj.__doc__ is not None, f"{mod.__name__}.{name} has no docstring"

    def test_all_submodules_have_all(self):
        """Every public submodule should define __all__."""
        import spherical_array_processing.sh as sh_mod
        import spherical_array_processing.acoustics as ac_mod
        import spherical_array_processing.array as arr_mod
        import spherical_array_processing.beamforming as bf_mod
        import spherical_array_processing.doa as doa_mod
        import spherical_array_processing.diffuseness as diff_mod
        import spherical_array_processing.coherence as coh_mod
        import spherical_array_processing.plotting as plt_mod
        import spherical_array_processing.coords as coords_mod

        for mod in [sh_mod, ac_mod, arr_mod, bf_mod, doa_mod, diff_mod, coh_mod, plt_mod, coords_mod]:
            assert hasattr(mod, "__all__"), f"{mod.__name__} missing __all__"

    def test_version_consistency(self):
        import spherical_array_processing as sap
        # Read pyproject.toml version
        from pathlib import Path
        import re
        toml = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        m = re.search(r'version\s*=\s*"([^"]+)"', toml)
        assert m is not None
        assert sap.__version__ == m.group(1)


# ---------------------------------------------------------------------------
# 10.  Radial function edge cases (ka=0 no warning)
# ---------------------------------------------------------------------------
import warnings


class TestRadialEdgeCases:
    def test_bn_rigid_ka_zero_no_warning(self):
        """bn for rigid sphere at ka=0 should not produce RuntimeWarning."""
        from spherical_array_processing.acoustics.radial import plane_wave_radial_bn
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for n in range(5):
                result = plane_wave_radial_bn(n, np.array([0.0]), sphere="rigid")
                assert np.all(np.isfinite(result))

    def test_bn_matrix_ka_zero_no_warning(self):
        from spherical_array_processing.acoustics.radial import bn_matrix
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            kr = np.array([0.0, 0.5, 1.0])
            B = bn_matrix(3, kr, sphere="rigid")
            assert np.all(np.isfinite(B))
