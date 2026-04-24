import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre

from spherical_array_processing.array.sampling import get_tdesign_fallback
from spherical_array_processing.beamforming import (
    axisymmetric_pattern,
    beam_weights_butterworth,
    beam_weights_dolph_chebyshev,
    beam_weights_supercardioid,
    mvdr_weights,
    normalize_axisymmetric_weights,
)
from spherical_array_processing.beamforming.fixed import beam_weights_cardioid
from spherical_array_processing.doa import music_spectrum, pwd_spectrum
from spherical_array_processing.doa.spectra import peak_pick_spectrum, spatial_spectrum_from_map
from spherical_array_processing.sh.transforms import direct_sht
from spherical_array_processing.types import SHBasisSpec
from spherical_array_processing.types import SphericalGrid


def test_mvdr_unit_response():
    r = np.eye(4, dtype=complex)
    d = np.array([1, 0, 0, 0], dtype=complex)
    w = mvdr_weights(r, d)
    assert np.allclose(np.vdot(w, d), 1.0)


def test_pwd_and_music_return_peaks():
    basis = SHBasisSpec(max_order=1, basis="real")
    grid = get_tdesign_fallback(order=2, n_points=50)
    r = np.eye(basis.n_coeffs, dtype=complex)
    pwd = pwd_spectrum(r, grid, basis, n_peaks=3)
    mus = music_spectrum(r + 0.01 * np.eye(basis.n_coeffs), grid, basis, n_sources=1)
    assert pwd.peak_indices.size == 3
    assert mus.peak_indices.size == 1


def test_supercardioid_high_order_extension_has_unit_front_gain_and_small_rear_response():
    weights = beam_weights_supercardioid(6)
    response = axisymmetric_pattern(np.array([0.0, np.pi]), weights)
    assert np.all(np.isfinite(weights))
    assert np.isclose(response[0], 1.0, atol=1e-12)
    assert abs(response[1]) < 1e-3


def test_supercardioid_low_orders_match_front_back_ratio_reference():
    expected = {
        1: np.array([4.59961088, 2.65558658]),
        2: np.array([2.96774688, 1.96298005, 0.74193672]),
        3: np.array([2.22163052, 1.60881033, 0.79591595, 0.21981848]),
        4: np.array([1.77715903, 1.36560919, 0.78339197, 0.31003222, 0.06724429]),
    }
    for order, reference in expected.items():
        weights = beam_weights_supercardioid(order)
        assert np.allclose(weights, reference, rtol=2e-8, atol=2e-8)


def test_axisymmetric_weight_normalization_has_unit_front_gain():
    weights = normalize_axisymmetric_weights(np.ones(4))
    assert np.isclose(axisymmetric_pattern(0.0, weights), 1.0, atol=1e-14)


def test_butterworth_modal_weights_match_spaudiopy_reference():
    weights = beam_weights_butterworth(order=3, filter_order=5, cutoff_order=3)
    reference = np.array([0.90360532, 0.90359767, 0.89587082, 0.63894545])
    assert np.allclose(weights, reference, rtol=2e-8, atol=2e-8)
    assert np.isclose(axisymmetric_pattern(0.0, weights), 1.0, atol=1e-14)


def test_dolph_chebyshev_weights_match_spharpy_reference():
    weights = beam_weights_dolph_chebyshev(order=3, design_parameter=10.0, design_criterion="sidelobe")
    reference = np.array([1.27194752, 0.93029824, 0.87658385, 0.58865845])
    assert np.allclose(weights, reference, rtol=2e-8, atol=2e-8)
    assert np.isclose(axisymmetric_pattern(0.0, weights), 1.0, atol=1e-14)


def test_cardioid_high_order_matches_independent_legendre_projection():
    order = 24
    weights = beam_weights_cardioid(order)
    expected = []
    for n in range(order + 1):
        integrand = lambda x: ((1.0 + x) / 2.0) ** order * eval_legendre(n, x)
        integral, _ = quad(integrand, -1.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=400)
        expected.append(2.0 * np.pi * integral)
    expected = np.asarray(expected)
    expected /= np.sum(expected * (2 * np.arange(order + 1) + 1) / (4.0 * np.pi))
    assert np.allclose(weights, expected, rtol=1e-11, atol=1e-12)


def test_direct_sht_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        direct_sht([1.0, 2.0], np.eye(2), weights=[1.0, -1.0])


def test_peak_pick_spectrum_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        peak_pick_spectrum([], 1)


def test_spatial_spectrum_from_map_requires_grid_length_match():
    grid = SphericalGrid(azimuth=[0.0, 1.0], angle2=[0.0, 0.0], convention="az_el")
    with pytest.raises(ValueError, match="grid size"):
        spatial_spectrum_from_map([1.0, 2.0, 3.0], grid, n_peaks=1)
