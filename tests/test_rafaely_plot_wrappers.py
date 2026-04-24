import numpy as np
import pytest

# Plotting is an optional extra.  Skip these tests cleanly when
# ``matplotlib`` is absent (e.g. in the minimal ``.[dev]``
# release-check environment) rather than failing collection.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — imported after importorskip

from spherical_array_processing.repro.rafaely import (
    plot_aliasing,
    plot_balloon,
    plot_contour,
    plot_sampling,
    plot_sphere,
)


def _rand_fnm(order: int) -> np.ndarray:
    n = (order + 1) ** 2
    rng = np.random.default_rng(order)
    return rng.normal(size=n) + 1j * rng.normal(size=n)


def test_rafaely_plot_wrappers_run():
    ax1 = plot_balloon(_rand_fnm(2))
    ax2 = plot_sphere(_rand_fnm(2))
    ax3 = plot_contour(_rand_fnm(2))
    ax4a, ax4b = plot_sampling(np.array([0.2, 0.4]), np.array([0.1, 2.1]))
    ax5 = plot_aliasing(np.eye(9))
    for ax in (ax1, ax2, ax3, ax4a, ax4b, ax5):
        assert ax is not None
        plt.close(ax.figure)

