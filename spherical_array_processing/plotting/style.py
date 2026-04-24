from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl

from ..types import FigureReproConfig


def apply_matlab_like_style(config: FigureReproConfig | None = None) -> None:
    cfg = config or FigureReproConfig()
    mpl.rcParams.update(
        {
            "figure.dpi": cfg.dpi,
            "font.family": cfg.font_family,
            "font.size": cfg.font_size,
            "axes.titlesize": cfg.font_size,
            "axes.labelsize": cfg.font_size,
            "lines.linewidth": cfg.line_width,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "savefig.dpi": cfg.dpi,
        }
    )


@contextmanager
def figure_repro_context(config: FigureReproConfig | None = None):
    old = mpl.rcParams.copy()
    apply_matlab_like_style(config)
    try:
        yield
    finally:
        mpl.rcParams.update(old)

