import numpy as np
from scipy.special import spherical_jn, spherical_yn

from spherical_array_processing.acoustics import (
    besselhs2,
    besselhs2d,
    besselys,
    besselysd,
    bn_matrix,
    kr,
    plane_wave_radial_bn,
    wavenumber,
)


def test_bn_shapes():
    kr = np.linspace(0.1, 3.0, 10)
    b0 = plane_wave_radial_bn(0, kr, sphere="open")
    assert b0.shape == (10,)
    bm = bn_matrix(3, kr, sphere="rigid", repeat_per_order=False)
    assert bm.shape == (10, 4)
    br = bn_matrix(3, kr, sphere="rigid", repeat_per_order=True)
    assert br.shape == (10, 16)


def test_second_kind_radial_wrappers_match_scipy():
    x = np.linspace(0.2, 5.0, 11)
    for n in range(5):
        assert np.allclose(besselys(n, x), spherical_yn(n, x))
        assert np.allclose(besselysd(n, x), spherical_yn(n, x, derivative=True))
        assert np.allclose(besselhs2(n, x), spherical_jn(n, x) - 1j * spherical_yn(n, x))
        assert np.allclose(
            besselhs2d(n, x),
            spherical_jn(n, x, derivative=True) - 1j * spherical_yn(n, x, derivative=True),
        )


def test_wavenumber_and_kr_helpers():
    freqs = np.array([0.0, 343.0])
    assert np.allclose(wavenumber(freqs, c=343.0), [0.0, 2.0 * np.pi])
    assert np.allclose(kr(freqs, radius_m=0.5, c=343.0), [0.0, np.pi])
