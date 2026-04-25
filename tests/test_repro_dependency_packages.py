from pathlib import Path

import numpy as np
import pytest

from spherical_array_processing.repro import array_response_simulator as ars
from spherical_array_processing.repro import sht
from spherical_array_processing.repro.sht.functions import _FLIEGE_MAT


ROOT = Path(__file__).resolve().parents[1]


def _matlab_function_names(package_dir: str) -> set[str]:
    pkg_root = ROOT / "src" / package_dir
    names = {p.stem for p in pkg_root.rglob("*.m")}
    return {n for n in names if not n.startswith("TEST_")}


def test_sht_matlab_surface_is_exposed():
    matlab_names = _matlab_function_names("Spherical-Harmonic-Transform")
    exported = set(sht.__all__)
    missing = matlab_names - exported
    assert not missing


def test_array_response_simulator_matlab_surface_is_exposed():
    matlab_names = _matlab_function_names("Array-Response-Simulator")
    exported = set(ars.__all__)
    missing = matlab_names - exported
    assert not missing


def test_sht_smoke_roundtrip_and_geometry_helpers():
    dirs = sht.grid2dirs(90, 90, POLAR_OR_ELEV=1, ZEROED_OR_CENTERED=1)
    f = np.linspace(0.0, 1.0, dirs.shape[0])
    fn, _ = sht.directSHT(1, f, dirs, "real")
    recon = sht.inverseSHT(fn, dirs, "real")
    grid = sht.Fdirs2grid(f, 90, 90)

    _, t_dirs = sht.getTdesign(4)
    w = sht.getVoronoiWeights(t_dirs)

    assert fn.shape == (4, 1)
    assert recon.shape == (dirs.shape[0], 1)
    assert grid.shape == (3, 4)
    assert np.isclose(np.sum(w), 4 * np.pi, atol=5e-2)

    if not _FLIEGE_MAT.exists():
        pytest.skip(
            "Fliege/Maier MATLAB nodes file is part of the out-of-tree "
            "MATLAB reference material (see ``src/Spherical-Harmonic-Transform/``) "
            "and is intentionally not shipped in the wheel/sdist; skip the "
            "Fliege-specific assertions when running outside the developer "
            "checkout."
        )
    vecs, dirs_f, weights = sht.getFliegeNodes(2)
    assert vecs.shape[1] == 3
    assert dirs_f.shape[1] == 2
    assert weights.ndim == 1


def test_array_response_simulator_smoke():
    x = np.array([0.0, 0.3, 1.2])
    j0 = ars.sph_besselj(0, x)
    coeffs = ars.sphModalCoeffs(3, np.linspace(0.1, 2.0, 8), "open")

    mic_dirs = np.array([[0.0, 0.0], [np.pi / 2, 0.0]], dtype=float)
    src_dirs = np.array([[0.0, 0.0]], dtype=float)
    h_t, h_f = ars.simulateSphArray(32, mic_dirs, src_dirs, "open", 0.042, 2, 16000)

    u_doa = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    r_mic = np.array([[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float)
    irs, tfs = ars.getArrayResponse(u_doa, r_mic, None, None, 32, fs=16000)

    assert np.isclose(j0[0], 1.0)
    assert coeffs.shape == (8, 4)
    assert h_t.shape == (32, 2, 1)
    assert h_f.shape == (17, 2, 1)
    assert irs.shape == (32, 2, 2)
    assert tfs.shape == (17, 2, 2)
