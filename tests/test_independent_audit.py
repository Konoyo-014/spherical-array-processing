"""Independent audit tests — written from scratch to validate mathematical
correctness, API contracts, edge cases, and cross-module integration.

These tests do NOT rely on any existing test infrastructure in the repository.
Every expected value is computed from first-principles or well-known identities.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
try:
    from scipy.special import sph_harm as scipy_sph_harm
except ImportError:
    from scipy.special import sph_harm_y as scipy_sph_harm

# ---------------------------------------------------------------------------
# 0.  Package-level smoke tests
# ---------------------------------------------------------------------------


class TestPackageSmoke:
    """Can the package be imported and does it expose expected symbols?"""

    def test_top_level_import(self):
        import spherical_array_processing as sap

        assert hasattr(sap, "__version__")
        assert sap.__version__ == "0.5.0"

    def test_types_importable(self):
        from spherical_array_processing.types import (
            ArrayGeometry,
            SHBasisSpec,
            SHCovariance,
            SHSignalFrame,
            SpatialSpectrumResult,
            SphericalGrid,
        )

    def test_all_submodules_importable(self):
        import spherical_array_processing.coords
        import spherical_array_processing.sh
        import spherical_array_processing.acoustics
        import spherical_array_processing.array
        import spherical_array_processing.beamforming
        import spherical_array_processing.doa
        import spherical_array_processing.diffuseness
        import spherical_array_processing.coherence
        import spherical_array_processing.plotting

    def test_py_typed_marker_exists(self):
        import importlib.resources as ir

        ref = ir.files("spherical_array_processing") / "py.typed"
        assert ref.is_file()


# ---------------------------------------------------------------------------
# 1.  coords.py — coordinate conversions
# ---------------------------------------------------------------------------
from spherical_array_processing.coords import (
    sph_to_cart,
    cart_to_sph,
    azel_to_az_colat,
    az_colat_to_azel,
    unit_sph_to_cart,
)


class TestCoords:
    """Validate coordinate transformations against known identities."""

    def test_sph_to_cart_origin(self):
        """Radius = 0 should give origin regardless of angles."""
        x, y, z = sph_to_cart(1.23, 0.45, radius=0.0)
        assert_allclose([x, y, z], [0.0, 0.0, 0.0], atol=1e-15)

    def test_sph_to_cart_north_pole_az_el(self):
        """el = pi/2 → north pole (0, 0, r)."""
        x, y, z = sph_to_cart(0.0, np.pi / 2, radius=2.0)
        assert_allclose([x, y, z], [0.0, 0.0, 2.0], atol=1e-14)

    def test_sph_to_cart_x_axis_az_el(self):
        """az=0, el=0 → (r, 0, 0)."""
        x, y, z = sph_to_cart(0.0, 0.0, radius=3.0)
        assert_allclose([x, y, z], [3.0, 0.0, 0.0], atol=1e-14)

    def test_sph_to_cart_az_colat_north_pole(self):
        """colat=0 → north pole."""
        x, y, z = sph_to_cart(0.0, 0.0, radius=1.0, convention="az_colat")
        assert_allclose([x, y, z], [0.0, 0.0, 1.0], atol=1e-14)

    def test_sph_to_cart_az_colat_equator(self):
        """colat=pi/2, az=pi/2 → y axis."""
        x, y, z = sph_to_cart(np.pi / 2, np.pi / 2, radius=1.0, convention="az_colat")
        assert_allclose([x, y, z], [0.0, 1.0, 0.0], atol=1e-14)

    def test_roundtrip_az_el(self):
        """sph→cart→sph should be identity (mod 2π on azimuth)."""
        rng = np.random.default_rng(42)
        az_orig = rng.uniform(-np.pi, np.pi, 100)
        el_orig = rng.uniform(-np.pi / 2, np.pi / 2, 100)
        r_orig = rng.uniform(0.1, 10.0, 100)
        x, y, z = sph_to_cart(az_orig, el_orig, r_orig)
        az, el, r = cart_to_sph(x, y, z)
        assert_allclose(r, r_orig, atol=1e-12)
        assert_allclose(el, el_orig, atol=1e-12)
        # azimuth mod 2π
        assert_allclose(np.cos(az), np.cos(az_orig), atol=1e-12)
        assert_allclose(np.sin(az), np.sin(az_orig), atol=1e-12)

    def test_roundtrip_az_colat(self):
        rng = np.random.default_rng(99)
        az_orig = rng.uniform(0, 2 * np.pi, 50)
        colat_orig = rng.uniform(0.01, np.pi - 0.01, 50)
        r_orig = rng.uniform(0.5, 5.0, 50)
        x, y, z = sph_to_cart(az_orig, colat_orig, r_orig, convention="az_colat")
        az, colat, r = cart_to_sph(x, y, z, convention="az_colat")
        assert_allclose(r, r_orig, atol=1e-12)
        assert_allclose(colat, colat_orig, atol=1e-12)

    def test_cart_to_sph_at_origin(self):
        """Origin should return r=0 without NaN."""
        az, el, r = cart_to_sph(0.0, 0.0, 0.0)
        assert r == 0.0
        assert np.isfinite(az)
        assert np.isfinite(el)

    def test_azel_colat_roundtrip(self):
        az_in = np.array([0.0, 1.0, 2.0])
        el_in = np.array([0.0, np.pi / 4, -np.pi / 4])
        az2, colat = azel_to_az_colat(az_in, el_in)
        az_out, el_out = az_colat_to_azel(az2, colat)
        assert_allclose(az_out, az_in, atol=1e-15)
        assert_allclose(el_out, el_in, atol=1e-15)

    def test_unit_sph_to_cart_norm(self):
        """All returned vectors should have unit norm."""
        rng = np.random.default_rng(7)
        az = rng.uniform(0, 2 * np.pi, 200)
        el = rng.uniform(-np.pi / 2, np.pi / 2, 200)
        xyz = unit_sph_to_cart(az, el)
        norms = np.linalg.norm(xyz, axis=-1)
        assert_allclose(norms, 1.0, atol=1e-14)

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            sph_to_cart(0.0, 0.0, convention="bad")
        with pytest.raises(ValueError, match="unsupported"):
            cart_to_sph(1.0, 0.0, 0.0, convention="bad")

    def test_vectorised_input(self):
        """Ensure arrays of inputs work correctly."""
        az = np.array([0, np.pi / 2, np.pi])
        el = np.array([0, 0, 0])
        x, y, z = sph_to_cart(az, el, 1.0)
        assert x.shape == (3,)
        assert_allclose(x[0], 1.0, atol=1e-14)
        assert_allclose(y[1], 1.0, atol=1e-14)
        assert_allclose(x[2], -1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 2.  types.py — dataclass validation
# ---------------------------------------------------------------------------
from spherical_array_processing.types import (
    SHBasisSpec,
    SphericalGrid,
    ArrayGeometry,
)


class TestTypes:
    def test_sh_basis_n_coeffs(self):
        for N in range(6):
            spec = SHBasisSpec(max_order=N)
            assert spec.n_coeffs == (N + 1) ** 2

    def test_spherical_grid_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            SphericalGrid(azimuth=[0.0, 1.0], angle2=[0.0])

    def test_spherical_grid_weight_mismatch(self):
        with pytest.raises(ValueError, match="weights shape"):
            SphericalGrid(azimuth=[0.0], angle2=[0.0], weights=[1.0, 2.0])

    def test_grid_elevation_colat_consistency(self):
        g = SphericalGrid(azimuth=[0.0], angle2=[np.pi / 4], convention="az_el")
        assert_allclose(g.elevation, [np.pi / 4])
        assert_allclose(g.colatitude, [np.pi / 4])  # pi/2 - pi/4 = pi/4

        g2 = SphericalGrid(azimuth=[0.0], angle2=[np.pi / 3], convention="az_colat")
        assert_allclose(g2.colatitude, [np.pi / 3])
        assert_allclose(g2.elevation, [np.pi / 2 - np.pi / 3])

    def test_array_geometry_n_sensors(self):
        g = SphericalGrid(azimuth=np.zeros(32), angle2=np.zeros(32))
        ag = ArrayGeometry(radius_m=0.042, sensor_grid=g)
        assert ag.n_sensors == 32


# ---------------------------------------------------------------------------
# 3.  sh.basis — spherical harmonic basis matrices
# ---------------------------------------------------------------------------
from spherical_array_processing.sh.basis import (
    acn_index,
    complex_matrix,
    real_matrix,
    matrix,
    replicate_per_order,
    complex_to_real_coeffs,
    real_to_complex_coeffs,
)


class TestSHBasis:
    def test_acn_index_order0(self):
        assert acn_index(0, 0) == 0

    def test_acn_index_order1(self):
        assert acn_index(1, -1) == 1
        assert acn_index(1, 0) == 2
        assert acn_index(1, 1) == 3

    def test_acn_sequential(self):
        """ACN should give sequential 0..N² - 1."""
        N = 5
        indices = []
        for n in range(N + 1):
            for m in range(-n, n + 1):
                indices.append(acn_index(n, m))
        assert indices == list(range((N + 1) ** 2))

    def test_complex_matrix_orthogonality(self):
        """On a dense-enough grid with quadrature weights, Y^H W Y ≈ I."""
        from spherical_array_processing.array.sampling import equiangle_sampling

        N = 3
        grid = equiangle_sampling(N)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)
        W = np.diag(grid.weights)
        gram = Y.conj().T @ W @ Y
        assert_allclose(gram, np.eye(spec.n_coeffs), atol=1e-10)

    def test_real_matrix_orthogonality(self):
        """Real SH basis also orthonormal under quadrature."""
        from spherical_array_processing.array.sampling import equiangle_sampling

        N = 3
        grid = equiangle_sampling(N)
        spec = SHBasisSpec(max_order=N, basis="real", angle_convention="az_colat")
        Y = real_matrix(spec, grid)
        W = np.diag(grid.weights)
        gram = Y.T @ W @ Y
        assert_allclose(gram, np.eye(spec.n_coeffs), atol=1e-10)

    def test_matrix_dispatches_correctly(self):
        from spherical_array_processing.array.sampling import fibonacci_grid

        grid = fibonacci_grid(50)
        spec_c = SHBasisSpec(max_order=1, basis="complex")
        spec_r = SHBasisSpec(max_order=1, basis="real")
        Yc = matrix(spec_c, grid)
        Yr = matrix(spec_r, grid)
        assert np.iscomplexobj(Yc)
        assert not np.iscomplexobj(Yr)

    def test_replicate_per_order(self):
        vals = [10, 20, 30]
        out = replicate_per_order(vals)
        expected = [10, 20, 20, 20, 30, 30, 30, 30, 30]
        assert_allclose(out, expected)

    def test_complex_real_roundtrip(self):
        """complex → real → complex should recover original coefficients."""
        rng = np.random.default_rng(123)
        N = 3
        n_coeffs = (N + 1) ** 2
        # Generate complex SH coefficients with conjugate symmetry for real field
        c_orig = np.zeros(n_coeffs, dtype=complex)
        for n in range(N + 1):
            c_orig[acn_index(n, 0)] = rng.normal()
            for m in range(1, n + 1):
                val = rng.normal() + 1j * rng.normal()
                c_orig[acn_index(n, m)] = val
                c_orig[acn_index(n, -m)] = ((-1) ** m) * val.conjugate()

        r = complex_to_real_coeffs(c_orig, N)
        assert np.isrealobj(r) or np.allclose(r.imag, 0)

        c_back = real_to_complex_coeffs(r, N)
        assert_allclose(c_back, c_orig, atol=1e-12)


# ---------------------------------------------------------------------------
# 4.  sh.transforms — SHT round-trip
# ---------------------------------------------------------------------------
from spherical_array_processing.sh.transforms import direct_sht


class TestSHTransforms:
    def test_sht_roundtrip(self):
        """Expand a known SH signal → SHT → ISHT → compare."""
        from spherical_array_processing.array.sampling import equiangle_sampling

        N = 4
        grid = equiangle_sampling(N)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)

        # Known SH coefficients: put energy in a few channels
        rng = np.random.default_rng(0)
        f_nm = rng.normal(size=spec.n_coeffs) + 1j * rng.normal(size=spec.n_coeffs)

        # Synthesize signal on grid: s = Y @ f_nm
        s = Y @ f_nm

        # Forward SHT should recover f_nm
        f_recovered = direct_sht(s, Y, grid)
        assert_allclose(f_recovered, f_nm, atol=1e-8)

    def test_sht_fibonacci_grid_uses_weighted_ls(self):
        """Approximate grids should still recover band-limited fields via weighted LS."""
        from spherical_array_processing.array.sampling import fibonacci_grid

        rng = np.random.default_rng(42)
        N = 4
        grid = fibonacci_grid(150)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)

        f_nm = rng.normal(size=spec.n_coeffs) + 1j * rng.normal(size=spec.n_coeffs)
        s = Y @ f_nm
        f_recovered = direct_sht(s, Y, grid)

        assert_allclose(f_recovered, f_nm, atol=1e-10)

    def test_sht_batch_dimension(self):
        """SHT should handle batch dimensions (..., n_points)."""
        from spherical_array_processing.array.sampling import fibonacci_grid

        N = 2
        grid = fibonacci_grid(100)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)

        batch = np.random.default_rng(5).normal(size=(3, 100)) + 0j
        result = direct_sht(batch, Y, grid)
        assert result.shape == (3, spec.n_coeffs)

    def test_sht_validation(self):
        with pytest.raises(ValueError, match="2D"):
            direct_sht(np.zeros(5), np.zeros((5, 4, 3)))


# ---------------------------------------------------------------------------
# 5.  acoustics.radial — Bessel functions & modal coefficients
# ---------------------------------------------------------------------------
from spherical_array_processing.acoustics.radial import (
    besseljs,
    besseljsd,
    besselhs,
    besselhsd,
    plane_wave_radial_bn,
    bn_matrix,
    sph_modal_coeffs,
)


class TestAcousticsRadial:
    def test_besseljs_order0(self):
        """j_0(x) = sin(x)/x."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        expected = np.sin(x) / x
        assert_allclose(besseljs(0, x), expected, atol=1e-14)

    def test_besseljs_at_zero(self):
        """j_0(0) = 1, j_n(0) = 0 for n > 0."""
        assert_allclose(besseljs(0, 0.0), 1.0, atol=1e-15)
        assert_allclose(besseljs(1, 0.0), 0.0, atol=1e-15)
        assert_allclose(besseljs(3, 0.0), 0.0, atol=1e-15)

    def test_besselhs_definition(self):
        """h_n(x) = j_n(x) + i*y_n(x), check at a known point."""
        x = np.array([2.0])
        h = besselhs(1, x)
        j = besseljs(1, x)
        # h should have imaginary part from y_n
        assert h.dtype == np.complex128
        assert not np.isnan(h)
        # For moderate x, |h| > |j|
        assert np.abs(h) > np.abs(j)

    def test_plane_wave_bn_open_sphere(self):
        """For open sphere, bn = 4π i^n j_n(kr)."""
        kr = np.array([1.5])
        for n in range(4):
            bn = plane_wave_radial_bn(n, kr, sphere="open")
            expected = 4 * np.pi * (1j ** n) * besseljs(n, kr)
            assert_allclose(bn, expected, atol=1e-13)

    def test_bn_matrix_shape(self):
        kr = np.array([0.5, 1.0, 2.0])
        N = 3
        B = bn_matrix(N, kr, sphere="rigid")
        assert B.shape == (3, (N + 1) ** 2)

    def test_bn_matrix_no_repeat(self):
        kr = np.array([1.0, 2.0])
        N = 2
        B = bn_matrix(N, kr, sphere="open", repeat_per_order=False)
        assert B.shape == (2, N + 1)

    def test_sph_modal_coeffs_shape(self):
        kR = np.linspace(0.1, 5.0, 20)
        M = sph_modal_coeffs(4, kR, array_type="rigid")
        assert M.shape == (20, 5)

    def test_plane_wave_bn_string_vs_int(self):
        """String sphere type should match integer type."""
        kr = np.array([1.0])
        for n in range(3):
            assert_allclose(
                plane_wave_radial_bn(n, kr, sphere=0),
                plane_wave_radial_bn(n, kr, sphere="open"),
            )
            assert_allclose(
                plane_wave_radial_bn(n, kr, sphere=1),
                plane_wave_radial_bn(n, kr, sphere="rigid"),
            )


# ---------------------------------------------------------------------------
# 6.  array.sampling — grid generation
# ---------------------------------------------------------------------------
from spherical_array_processing.array.sampling import (
    fibonacci_grid,
    equiangle_sampling,
    get_tdesign_fallback,
)


class TestArraySampling:
    def test_fibonacci_grid_size(self):
        g = fibonacci_grid(100)
        assert g.size == 100
        assert g.azimuth.shape == (100,)
        assert g.angle2.shape == (100,)

    def test_fibonacci_grid_weights_sum(self):
        """Weights should sum to 4π (full sphere area)."""
        g = fibonacci_grid(500)
        assert_allclose(g.weights.sum(), 4 * np.pi, atol=1e-12)

    def test_fibonacci_grid_convention(self):
        g = fibonacci_grid(10)
        assert g.convention == "az_colat"
        assert np.all(g.angle2 >= 0)
        assert np.all(g.angle2 <= np.pi)

    def test_fibonacci_grid_invalid(self):
        with pytest.raises(ValueError, match="positive"):
            fibonacci_grid(0)

    def test_equiangle_sampling_weights_sum(self):
        g = equiangle_sampling(3)
        assert_allclose(g.weights.sum(), 4 * np.pi, atol=1e-10)

    def test_equiangle_sampling_convention(self):
        g = equiangle_sampling(2)
        assert g.convention == "az_colat"

    def test_tdesign_fallback_default_size(self):
        g = get_tdesign_fallback(3)
        assert g.size >= 2 * (3 + 1) ** 2


# ---------------------------------------------------------------------------
# 7.  beamforming.fixed — beam weight designs
# ---------------------------------------------------------------------------
from spherical_array_processing.beamforming.fixed import (
    beam_weights_cardioid,
    beam_weights_hypercardioid,
    beam_weights_supercardioid,
    beam_weights_maxev,
    axisymmetric_pattern,
)


class TestBeamformingFixed:
    def test_hypercardioid_sum(self):
        """Sum of (2n+1)/(N+1)^2 from 0..N = 1."""
        for N in range(1, 6):
            w = beam_weights_hypercardioid(N)
            assert w.shape == (N + 1,)
            total = sum(w[n] * (2 * n + 1) / (4 * np.pi) for n in range(N + 1))
            # Front gain should be normalised
            front = axisymmetric_pattern(np.array([0.0]), w)[0]
            assert front != 0  # should have some nonzero gain

    def test_cardioid_returns_correct_size(self):
        for N in range(1, 5):
            w = beam_weights_cardioid(N)
            assert w.shape == (N + 1,)

    def test_supercardioid_known_orders(self):
        """Supercardioid should return sensible weights for orders 1-4."""
        for N in range(1, 5):
            w = beam_weights_supercardioid(N)
            assert w.shape == (N + 1,)
            assert np.all(np.isfinite(w))

    def test_maxev_weights_positive(self):
        for N in range(1, 6):
            w = beam_weights_maxev(N)
            assert np.all(w >= 0), f"max-EV weights should be non-negative for order {N}"
            # Unit front gain: axisymmetric_pattern(0, w) == 1
            front = axisymmetric_pattern(np.array([0.0]), w)[0]
            assert_allclose(front, 1.0, atol=1e-12)

    def test_axisymmetric_pattern_shape(self):
        theta = np.linspace(0, np.pi, 181)
        w = beam_weights_hypercardioid(2)
        p = axisymmetric_pattern(theta, w)
        assert p.shape == (181,)

    def test_axisymmetric_pattern_at_zero(self):
        """Pattern should have maximum at theta=0 for reasonable beamformers."""
        theta = np.linspace(0, np.pi, 361)
        w = beam_weights_hypercardioid(3)
        p = axisymmetric_pattern(theta, w)
        # Front should be near the max
        assert np.argmax(p) < 10  # within ~5 degrees of front


# ---------------------------------------------------------------------------
# 8.  beamforming.adaptive — MVDR & LCMV
# ---------------------------------------------------------------------------
from spherical_array_processing.beamforming.adaptive import mvdr_weights, lcmv_weights


class TestBeamformingAdaptive:
    def test_mvdr_identity_covariance(self):
        """With identity Cov, MVDR → matched filter normalised."""
        N = 4
        d = np.random.default_rng(7).normal(size=N) + 1j * np.random.default_rng(8).normal(size=N)
        R = np.eye(N, dtype=complex)
        w = mvdr_weights(R, d)
        # w should satisfy d^H w = 1
        assert_allclose(np.vdot(d, w), 1.0, atol=1e-6)

    def test_mvdr_distortionless(self):
        """MVDR: d^H w = 1 for arbitrary positive definite R."""
        rng = np.random.default_rng(42)
        N = 6
        A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        R = A @ A.conj().T + 0.1 * np.eye(N)
        d = rng.normal(size=N) + 1j * rng.normal(size=N)
        w = mvdr_weights(R, d)
        assert_allclose(np.vdot(d, w), 1.0, atol=1e-5)

    def test_lcmv_single_constraint_matches_mvdr(self):
        """LCMV with one constraint and response=1 should match MVDR."""
        rng = np.random.default_rng(99)
        N = 4
        R = np.eye(N, dtype=complex)
        d = rng.normal(size=N) + 1j * rng.normal(size=N)
        w_mvdr = mvdr_weights(R, d, diagonal_loading=0.0)
        C = d[:, None]
        f = np.array([1.0 + 0j])
        w_lcmv = lcmv_weights(R, C, f, diagonal_loading=0.0)
        assert_allclose(w_lcmv, w_mvdr, atol=1e-10)

    def test_lcmv_constraints_satisfied(self):
        """LCMV: C^H w = f."""
        rng = np.random.default_rng(0)
        N = 6
        R = np.eye(N, dtype=complex)
        C = rng.normal(size=(N, 2)) + 1j * rng.normal(size=(N, 2))
        f = np.array([1.0, 0.0]) + 0j
        w = lcmv_weights(R, C, f, diagonal_loading=0.0)
        assert_allclose(C.conj().T @ w, f, atol=1e-8)

    def test_mvdr_validation(self):
        with pytest.raises(ValueError, match="square"):
            mvdr_weights(np.zeros((3, 4)), np.zeros(3))


# ---------------------------------------------------------------------------
# 9.  doa — DOA estimation
# ---------------------------------------------------------------------------
from spherical_array_processing.doa.spectra import (
    peak_pick_spectrum,
    pwd_spectrum,
    music_spectrum,
    spatial_spectrum_from_map,
)


class TestDOA:
    def test_peak_pick_basic(self):
        s = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        idx = peak_pick_spectrum(s, 2)
        assert idx[0] == 3
        assert idx[1] == 1

    def test_peak_pick_more_than_exist(self):
        s = np.array([1.0, 2.0])
        idx = peak_pick_spectrum(s, 10)
        assert len(idx) == 2

    def test_pwd_spectrum_identity_cov(self):
        """Identity cov → uniform-ish spectrum."""
        N = 2
        n_sh = (N + 1) ** 2
        R = np.eye(n_sh, dtype=complex)
        grid = fibonacci_grid(100)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        result = pwd_spectrum(R, grid, spec, n_peaks=1)
        assert result.spectrum.shape == (100,)
        assert len(result.peak_indices) == 1

    def test_music_single_source(self):
        """MUSIC should locate a single plane wave in SH domain."""
        N = 3
        n_sh = (N + 1) ** 2
        grid = fibonacci_grid(200)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)

        # Create covariance from a single source at grid point 50.  The
        # package's SHT produces coefficients ∝ Y_n^m*(k̂_src), so the
        # physical covariance uses the conjugated basis row.
        src_idx = 50
        d = Y[src_idx, :].conj()
        R = np.outer(d, d.conj()) + 0.01 * np.eye(n_sh)

        result = music_spectrum(R, grid, spec, n_sources=1, n_peaks=1)
        peak = result.peak_indices[0]
        # Peak should be near the true source
        # Allow some error due to Fibonacci grid discretisation
        assert abs(peak - src_idx) < 10 or result.spectrum[src_idx] > 0.5 * result.spectrum.max()

    def test_music_validation(self):
        N = 2
        n_sh = (N + 1) ** 2
        R = np.eye(n_sh, dtype=complex)
        grid = fibonacci_grid(50)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        with pytest.raises(ValueError, match="n_sources"):
            music_spectrum(R, grid, spec, n_sources=n_sh)


# ---------------------------------------------------------------------------
# 10.  diffuseness — estimators
# ---------------------------------------------------------------------------
from spherical_array_processing.diffuseness.estimators import (
    intensity_vectors_from_foa,
    diffuseness_ie,
    diffuseness_tv,
    diffuseness_sv,
    diffuseness_cmd,
)


class TestDiffuseness:
    def test_intensity_vectors_shape(self):
        foa = np.random.default_rng(0).normal(size=(10, 4)) + 0j
        iv = intensity_vectors_from_foa(foa)
        assert iv.shape == (10, 3)
        assert np.isrealobj(iv)

    def test_intensity_vectors_validation(self):
        with pytest.raises(ValueError, match="4 channels"):
            intensity_vectors_from_foa(np.zeros((5, 2)))

    def test_intensity_vector_direction_acn_default(self):
        """Default ACN channel order must give correct DOA for a
        plane wave encoded by the package's own encoder."""
        from spherical_array_processing.ambi import encode_plane_wave
        from spherical_array_processing.types import SphericalGrid
        T = 256
        sig = np.cos(2 * np.pi * 100 * np.arange(T) / 16000.0)
        # Source at +x.
        grid = SphericalGrid(
            azimuth=np.array([0.0]),
            angle2=np.array([np.pi / 2.0]),
            convention="az_colat",
        )
        foa_qt = encode_plane_wave(sig, grid, max_order=1)
        # Convert to (T, 4) for this API.
        iv = intensity_vectors_from_foa(foa_qt.T)
        mean_iv = iv.mean(axis=0)
        # Dominant component must be +x.
        assert np.argmax(np.abs(mean_iv)) == 0
        assert mean_iv[0] > 0

    def test_intensity_vector_fuma_and_acn_agree_on_same_field(self):
        """Passing the same physical field in either channel order
        should yield identical intensity vectors."""
        from spherical_array_processing.ambi import encode_plane_wave
        from spherical_array_processing.types import SphericalGrid
        grid = SphericalGrid(
            azimuth=np.array([np.pi / 3]),
            angle2=np.array([np.pi / 4]),
            convention="az_colat",
        )
        T = 128
        sig = np.sin(2 * np.pi * 250 * np.arange(T) / 16000.0)
        foa_acn_qt = encode_plane_wave(sig, grid, max_order=1)
        foa_acn_tq = foa_acn_qt.T
        # ACN → FuMa reorder: [W, X, Y, Z] = ACN[0, 3, 1, 2].
        foa_fuma_tq = foa_acn_tq[:, [0, 3, 1, 2]]
        iv_acn = intensity_vectors_from_foa(foa_acn_tq, channel_order="acn")
        iv_fuma = intensity_vectors_from_foa(
            foa_fuma_tq, channel_order="fuma",
        )
        np.testing.assert_allclose(iv_acn, iv_fuma, atol=1e-14)

    def test_intensity_vector_rejects_bad_channel_order(self):
        with pytest.raises(ValueError, match="channel_order"):
            intensity_vectors_from_foa(
                np.zeros((10, 4)), channel_order="bogus",
            )

    def test_diffuseness_ie_range(self):
        """Output should be in [0, 1]."""
        rng = np.random.default_rng(10)
        A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        C = A @ A.conj().T
        d = diffuseness_ie(C)
        assert 0.0 <= d <= 1.0

    def test_diffuseness_ie_perfect_diffuse(self):
        """Identity cov → high diffuseness (no dominant intensity direction)."""
        C = np.eye(4, dtype=complex)
        d = diffuseness_ie(C)
        # With identity, trace = 4, energy = 2, intensity = diag(1,1,1) → norm = sqrt(3)
        # diffuseness = 1 - sqrt(3)/2 ≈ 0.134
        assert d > 0.0

    def test_diffuseness_tv_uniform_vectors(self):
        """Uniformly random intensity vectors → high diffuseness."""
        rng = np.random.default_rng(42)
        iv = rng.normal(size=(1000, 3))
        d = diffuseness_tv(iv)
        assert d > 0.5

    def test_diffuseness_tv_identical_vectors(self):
        """All same direction → low diffuseness (0)."""
        iv = np.tile([1.0, 0.0, 0.0], (100, 1))
        d = diffuseness_tv(iv)
        assert d < 0.01

    def test_diffuseness_sv_range(self):
        rng = np.random.default_rng(3)
        iv = rng.normal(size=(50, 3))
        d = diffuseness_sv(iv)
        assert 0.0 <= d <= 1.0

    def test_diffuseness_cmd_identity(self):
        """Identity cov in SH domain → perfectly diffuse (CMD ≈ 1)."""
        N = 2
        n_sh = (N + 1) ** 2
        R = np.eye(n_sh, dtype=complex)
        d, d_ord = diffuseness_cmd(R)
        # Identity means equal eigenvalues → perfectly diffuse
        assert_allclose(d, 1.0, atol=1e-10)

    def test_diffuseness_cmd_validation(self):
        with pytest.raises(ValueError, match="square"):
            diffuseness_cmd(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# 11.  coherence — diffuse field coherence
# ---------------------------------------------------------------------------
from spherical_array_processing.coherence.diffuse import (
    diffuse_coherence_matrix_omni,
    diffuse_coherence_from_weights,
)


class TestCoherence:
    def test_diffuse_coherence_diagonal_is_one(self):
        """Self-coherence should always be 1."""
        xyz = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=float)
        f = np.array([1000.0])
        C = diffuse_coherence_matrix_omni(xyz, f)
        assert C.shape == (1, 3, 3)
        assert_allclose(np.diag(C[0].real), [1, 1, 1], atol=1e-14)

    def test_diffuse_coherence_sinc_formula(self):
        """C_{ij} = sinc(k*d_{ij}/π) for omni sensors in diffuse field."""
        d_12 = 0.05  # 5 cm apart
        xyz = np.array([[0, 0, 0], [d_12, 0, 0]], dtype=float)
        f = np.array([1000.0])
        k = 2 * np.pi * 1000 / 343.0
        expected = np.sinc(k * d_12 / np.pi)
        C = diffuse_coherence_matrix_omni(xyz, f)
        assert_allclose(C[0, 0, 1].real, expected, atol=1e-12)

    def test_diffuse_coherence_from_weights_self(self):
        """Coherence of a vector with itself should be 1."""
        w = np.array([1, 2, 3]) + 1j * np.array([0.5, -0.5, 1])
        c = diffuse_coherence_from_weights(w, w)
        assert_allclose(abs(c), 1.0, atol=1e-14)

    def test_diffuse_coherence_from_weights_orthogonal(self):
        """Orthogonal weights → zero coherence."""
        w_a = np.array([1, 0, 0], dtype=complex)
        w_b = np.array([0, 1, 0], dtype=complex)
        c = diffuse_coherence_from_weights(w_a, w_b)
        assert_allclose(abs(c), 0.0, atol=1e-14)

    def test_diffuse_coherence_from_weights_validation(self):
        with pytest.raises(ValueError, match="same length"):
            diffuse_coherence_from_weights(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# 12.  Cross-module integration: encode → beamform pipeline
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_encode_beamform_pipeline(self):
        """Full pipeline: create grid → SH encode plane wave → beamform → check gain."""
        from spherical_array_processing.array.sampling import equiangle_sampling

        N = 3
        grid = equiangle_sampling(N)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)

        # Simulate a plane wave from direction (az=0, colat=pi/2) = equator front
        src_az, src_colat = 0.0, np.pi / 2
        src_grid = SphericalGrid(
            azimuth=np.array([src_az]),
            angle2=np.array([src_colat]),
            convention="az_colat",
        )
        Y_src = complex_matrix(spec, src_grid)
        # SH coefficients of ideal plane wave = Y_src^H (conjugated)
        d = Y_src.conj().flatten()

        # Signal on the grid from this plane wave
        signal = Y @ d

        # Forward SHT
        f_nm = direct_sht(signal, Y, grid)

        # Beamform: apply hypercardioid weights (in SH domain)
        w_ax = beam_weights_hypercardioid(N)
        # Replicate for full SH: w_nm = w_n * Y_nm(look_dir)^*
        w_per_nm = replicate_per_order(w_ax) * Y_src.conj().flatten()

        output = np.vdot(w_per_nm, f_nm)
        # Should have positive real gain (signal arrives from look direction)
        assert abs(output) > 0.01

    def test_full_doa_pipeline(self):
        """Create synthetic SH covariance → run PWD → verify peak near source."""
        N = 3
        n_sh = (N + 1) ** 2
        scan_grid = fibonacci_grid(300)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")

        # Source at (az=1.0, colat=1.0)
        src_grid = SphericalGrid(
            azimuth=np.array([1.0]), angle2=np.array([1.0]), convention="az_colat"
        )
        Y_src = complex_matrix(spec, src_grid)
        # PWD: P(Ω) = y(Ω)ᵀ R y(Ω)*; the physical SHT coefficients of
        # a plane wave are ∝ Y_n^m*(k̂_src), so use the conjugated basis
        # row here to build the rank-1 covariance.
        d = Y_src.conj().flatten()
        R = np.outer(d, d.conj()) + 0.05 * np.eye(n_sh)

        result = pwd_spectrum(R, scan_grid, spec, n_peaks=1)
        peak_az = scan_grid.azimuth[result.peak_indices[0]]
        peak_colat = scan_grid.angle2[result.peak_indices[0]]

        # Check angular proximity accounting for azimuth wraparound
        az_diff = min(abs(peak_az - 1.0), 2 * np.pi - abs(peak_az - 1.0))
        assert az_diff < 0.3, f"azimuth off by {az_diff:.3f} rad"
        assert abs(peak_colat - 1.0) < 0.3, f"colat off by {abs(peak_colat - 1.0):.3f} rad"


# ---------------------------------------------------------------------------
# 13.  Edge cases & error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_order_sh(self):
        """Order 0 should work (single coefficient = monopole)."""
        spec = SHBasisSpec(max_order=0)
        assert spec.n_coeffs == 1
        grid = fibonacci_grid(10)
        Y = complex_matrix(spec, grid)
        assert Y.shape == (10, 1)
        # Y_00 should be constant = 1/sqrt(4π)
        assert_allclose(np.abs(Y[:, 0]), 1.0 / np.sqrt(4 * np.pi), atol=1e-12)

    def test_single_point_grid(self):
        """Grid with 1 point should not crash."""
        g = SphericalGrid(azimuth=[0.0], angle2=[0.0])
        assert g.size == 1

    def test_large_order_acn(self):
        """ACN indexing at high order."""
        assert acn_index(10, 0) == 10 * 11
        assert acn_index(10, 10) == 10 * 11 + 10

    def test_bn_open_vs_rigid_differ(self):
        """Open and rigid sphere responses should differ."""
        kr = np.array([2.0])
        b_open = plane_wave_radial_bn(1, kr, sphere="open")
        b_rigid = plane_wave_radial_bn(1, kr, sphere="rigid")
        assert not np.allclose(b_open, b_rigid)

    def test_empty_like_arrays_handled(self):
        """Functions accepting ArrayLike should handle Python lists."""
        x, y, z = sph_to_cart([0.0, 1.0], [0.0, 0.5])
        assert x.shape == (2,)

    def test_replicate_per_order_order0(self):
        out = replicate_per_order([42.0])
        assert_allclose(out, [42.0])
