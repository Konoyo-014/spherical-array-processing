"""Covariance-matrix estimators and regularisers for robust SH processing.

Provides three families of tools:

* **Shrinkage** — Ledoit-Wolf and OAS (Chen et al. 2010) data-driven
  shrinkage toward a scaled-identity target.  Indispensable when the
  number of time snapshots is comparable to or smaller than the
  covariance dimension, as in short-duration SH-domain analysis.
* **Forward-backward averaging** — ``R_fb = (R + J R* J) / 2`` which
  halves the variance of the sample covariance for symmetric arrays
  and helps decorrelate coherent sources before MUSIC/ESPRIT.
* **Diagonal loading** — small ``λ I`` add-on used as a crude but
  often sufficient regulariser for MVDR / LCMV weight computation.
"""

from .shrinkage import (
    diagonal_loading,
    forward_backward_average,
    ledoit_wolf_shrinkage,
    oas_shrinkage,
)

__all__ = [
    "diagonal_loading",
    "forward_backward_average",
    "ledoit_wolf_shrinkage",
    "oas_shrinkage",
]
