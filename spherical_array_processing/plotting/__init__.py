"""Plotting helpers — matplotlib-based visualisations.

``matplotlib`` is an **optional** dependency (install as the
``[plotting]`` extra: ``pip install 'spherical-array-processing[plotting]'``).
All entry points import matplotlib lazily, so just ``import
spherical_array_processing`` does not pull plotting into headless
environments.  Attempting to call any plotting function without
matplotlib installed raises a clean ``ImportError`` with the install
instructions.
"""

from __future__ import annotations


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:  # pragma: no cover — guarded by extras
        raise ImportError(
            "matplotlib is required for the plotting submodule; "
            "install with `pip install "
            "'spherical-array-processing[plotting]'` or "
            "`pip install matplotlib`."
        ) from exc


def apply_matlab_like_style(*args, **kwargs):
    _require_matplotlib()
    from .style import apply_matlab_like_style as _impl
    return _impl(*args, **kwargs)


def figure_repro_context(*args, **kwargs):
    _require_matplotlib()
    from .style import figure_repro_context as _impl
    return _impl(*args, **kwargs)


def figure_style_context(*args, **kwargs):
    _require_matplotlib()
    from .style import figure_style_context as _impl
    return _impl(*args, **kwargs)


def plot_directional_map_from_grid(*args, **kwargs):
    _require_matplotlib()
    from .politis_helpers import (
        plot_directional_map_from_grid as _impl,
    )
    return _impl(*args, **kwargs)


def plot_mic_array(*args, **kwargs):
    _require_matplotlib()
    from .politis_helpers import plot_mic_array as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "apply_matlab_like_style",
    "figure_repro_context",
    "figure_style_context",
    "plot_directional_map_from_grid",
    "plot_mic_array",
]
