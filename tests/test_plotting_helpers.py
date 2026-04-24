import numpy as np
import pytest

# Plotting is an optional extra.  Skip these tests cleanly when
# ``matplotlib`` is absent (e.g. in the minimal ``.[dev]``
# release-check environment) rather than failing collection.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — imported after importorskip

from spherical_array_processing.plotting import plot_directional_map_from_grid, plot_mic_array


def test_plot_mic_array_runs():
    dirs = np.array([[0, 0], [90, 0], [0, 45], [180, -45]], dtype=float)
    ax = plot_mic_array(dirs, 0.042)
    assert ax.name == "3d"
    plt.close(ax.figure)


def test_plot_directional_map_from_grid_runs():
    azi_res = 30
    pol_res = 30
    n_azi = int(round(360 / azi_res)) + 1
    n_pol = int(round(180 / pol_res)) + 1
    vals = np.linspace(0, 1, n_azi * n_pol)
    ax = plot_directional_map_from_grid(vals, azi_res, pol_res)
    assert ax is not None
    plt.close(ax.figure)
