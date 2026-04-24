import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_rafaely_ch1_spherical_grid import main as run_ch1_grid
from scripts.run_rafaely_ch1_xyz_coordinates import main as run_ch1_xyz
from scripts.run_rafaely_ch1_pn import compute_pn_data, main as run_ch1_pn
from scripts.run_rafaely_ch1_pnm import compute_pnm_data, main as run_ch1_pnm
from scripts.run_rafaely_ch1_sh_balloon import compute_sh_balloon_grid, main as run_ch1_sh_balloon
from scripts.run_rafaely_ch1_example_function import compute_example_function_coeffs, main as run_ch1_example
from scripts.run_rafaely_ch1_rotation import compute_rotation_data, main as run_ch1_rotation
from scripts.run_rafaely_ch1_sinc_and_cap import compute_sinc_and_cap_data, main as run_ch1_sinc_cap
from scripts.run_rafaely_ch1_truncated_cap import compute_truncated_cap_data, main as run_ch1_trunc_cap
from scripts.run_rafaely_ch6_chebyshev_polynomial import compute_chebyshev_demo, main as run_ch6_cheb
from scripts.run_rafaely_ch6_dolph import compute_dolph_demo, main as run_ch6_dolph
from scripts.run_rafaely_ch6_hypercardioid_beampattern import compute_hypercardioid_curves, main as run_ch6_hyper
from scripts.run_rafaely_ch6_supercardioid_beampatterns import compute_supercardioid_patterns, main as run_ch6_super
from scripts.run_rafaely_ch6_wng_and_di_example import compute_wng_di_example, main as run_ch6_wng_di
from scripts.run_rafaely_ch6_wng_open_and_rigid import compute_wng_open_and_rigid, main as run_ch6_wng_or
from scripts.run_rafaely_ch6_mixed_objectives_designs import compute_mixed_objectives_designs, main as run_ch6_mixed
from scripts.run_rafaely_ch6_multiple_objective_beampatterns import (
    compute_multiple_objective_beampatterns,
    main as run_ch6_multi,
)


def test_run_rafaely_ch1_spherical_grid_script():
    fig, ax = run_ch1_grid(show=False)
    assert fig is not None
    assert ax.name == "3d"
    plt.close(fig)


def test_run_rafaely_ch1_xyz_coordinates_script():
    figs_axes = run_ch1_xyz(show=False)
    assert len(figs_axes) == 4
    for fig, ax in figs_axes:
        assert fig is not None
        assert ax is not None
        plt.close(fig)


def test_run_rafaely_ch1_pn_script():
    x, polys = compute_pn_data()
    assert x.shape == (256,)
    assert set(polys.keys()) == {0, 1, 2, 3, 4}
    _, _, fig = run_ch1_pn(show=False)
    plt.close(fig)


def test_run_rafaely_ch1_pnm_script():
    x, pnm = compute_pnm_data()
    assert x.shape == (256,)
    assert (4, 4) in pnm
    _, _, fig = run_ch1_pnm(show=False)
    plt.close(fig)


def test_run_rafaely_ch1_sh_balloon_script_small():
    grid = compute_sh_balloon_grid(max_order=2, resolution=8)
    assert grid["Y"].shape[0] == 9
    _, figs = run_ch1_sh_balloon(show=False, max_order=1, resolution=8, views=("angle",))
    assert len(figs) == 1
    for fig in figs:
        plt.close(fig)


def test_run_rafaely_ch1_example_function_script():
    fnm = compute_example_function_coeffs()
    assert fnm.shape == (9,)
    _, figs = run_ch1_example(show=False)
    assert len(figs) == 3
    for fig in figs:
        plt.close(fig)


def test_run_rafaely_ch1_rotation_script():
    d = compute_rotation_data(order=2, alpha_deg=30.0)
    assert d["fnm"].shape == (9,)
    assert d["fnm1"].shape == (9,)
    _, figs = run_ch1_rotation(show=False)
    assert len(figs) == 1
    for fig in figs:
        plt.close(fig)


def test_run_rafaely_ch1_sinc_and_cap_script():
    d = compute_sinc_and_cap_data()
    assert d["S1"].shape[0] == 2
    assert d["S2"].shape[0] == 2
    _, figs = run_ch1_sinc_cap(show=False)
    assert len(figs) == 3
    for fig in figs:
        plt.close(fig)


def test_run_rafaely_ch1_truncated_cap_script_small():
    d = compute_truncated_cap_data(max_order=12, alpha_deg=30.0, add_dc=10.0)
    assert d["fnm"].shape == (13**2,)
    _, figs = run_ch1_trunc_cap(show=False, max_order=12, orders=(2, 4, 8, 12), resolution=12)
    assert len(figs) == 1
    for fig in figs:
        plt.close(fig)


def test_run_rafaely_ch6_chebyshev_polynomial_script():
    data = compute_chebyshev_demo()
    assert np.isclose(data["x0"], np.cos(np.pi / 16) / np.cos(np.pi / 8))
    assert data["R"] > 1.0
    _, figs = run_ch6_cheb(show=False)
    assert len(figs) == 2
    for fig in figs:
        plt.close(fig)


def test_run_rafaely_ch6_dolph_script():
    data = compute_dolph_demo()
    assert "plot1" in data and "plot2" in data
    _, figs = run_ch6_dolph(show=False)
    assert len(figs) == 2
    for fig in figs:
        plt.close(fig)


def test_run_rafaely_ch6_hypercardioid_script():
    theta, curves = compute_hypercardioid_curves()
    assert theta.shape == (512,)
    assert set(curves.keys()) == {1, 2, 3, 4, 5}
    _, _, fig = run_ch6_hyper(show=False)
    plt.close(fig)


def test_run_rafaely_ch6_supercardioid_script():
    theta, y, f_db = compute_supercardioid_patterns()
    assert theta.shape == (512,)
    assert set(y.keys()) == {0, 1, 2, 3, 4}
    assert set(f_db.keys()) == {0, 1, 2, 3, 4}
    _, _, _, fig = run_ch6_super(show=False)
    plt.close(fig)


def test_run_rafaely_ch6_wng_open_and_rigid_script():
    data = compute_wng_open_and_rigid(order=3, n_points=128)
    assert data["kr"].shape == (128,)
    assert data["WNG_open"].shape == (128,)
    assert data["WNG_rigid"].shape == (128,)
    assert np.all(np.isfinite(data["WNG_open"]))
    assert np.all(np.isfinite(data["WNG_rigid"]))
    _, fig = run_ch6_wng_or(show=False, order=3)
    plt.close(fig)


def test_run_rafaely_ch6_wng_di_example_script():
    data = compute_wng_di_example(order=3, n_points=128)
    assert data["kr"].shape == (128,)
    assert np.all(np.isfinite(data["DI_maxDI"]))
    _, figs = run_ch6_wng_di(show=False, order=3)
    assert len(figs) == 2
    for fig in figs:
        plt.close(fig)


def test_run_rafaely_ch6_mixed_objectives_designs_script():
    rows = compute_mixed_objectives_designs()
    assert len(rows) == 8
    assert all("DI" in r and "WNG" in r for r in rows)
    rows2 = run_ch6_mixed(print_table=False)
    assert len(rows2) == 8


def test_run_rafaely_ch6_multiple_objective_beampatterns_script_small():
    data = compute_multiple_objective_beampatterns(
        order=3,
        kr=2.0,
        sphere=1,
        wng_min_db=5.0,
        sidelobe_db=-20.0,
        n_sl_angles=15,
        n_plot_angles=181,
    )
    assert data["dn1"].shape == (4,)
    assert data["dn2"].shape == (4,)
    assert np.isfinite(data["DI1"]) and np.isfinite(data["WNG1"])
    _, figs = run_ch6_multi(
        show=False,
        order=3,
        kr=2.0,
        sphere=1,
        wng_min_db=5.0,
        sidelobe_db=-20.0,
        n_sl_angles=15,
        n_plot_angles=181,
    )
    assert len(figs) == 2
    for fig in figs:
        plt.close(fig)
