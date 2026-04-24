import numpy as np

from spherical_array_processing.coords import cart_to_sph, sph_to_cart


def test_sph_cart_roundtrip_az_el():
    az = np.array([0.1, -1.2, 2.3])
    el = np.array([0.2, -0.4, 0.7])
    r = np.array([1.0, 2.0, 0.8])
    x, y, z = sph_to_cart(az, el, r, convention="az_el")
    az2, el2, r2 = cart_to_sph(x, y, z, convention="az_el")
    assert np.allclose(r2, r)
    assert np.allclose(np.unwrap(az2), np.unwrap(az))
    assert np.allclose(el2, el)

