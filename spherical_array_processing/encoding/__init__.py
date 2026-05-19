"""SH encoding filters for spherical microphone arrays.

This module provides radial equalization filters that invert the
frequency-dependent modal gain ``B_n(kr)`` of the array.  Three
regularization strategies are exposed:

* ``regularization="tikhonov"`` — classical ``1/B_n`` inverse with a
  frequency-independent noise-floor ``λ``,
  ``H_n = B_n* / (|B_n|² + λ²)``.
* ``regularization="wng_limit"`` — Bernschütz-style soft limit that caps
  the per-order magnitude at ``G_max`` dB.  Matches the ``radInverse``
  family of scripts in Politis' MATLAB toolkit.
* ``regularization=None`` — unregularized ``1/B_n`` (for comparison and
  analytic checks; will blow up at ``kr → 0`` for ``n ≥ 1``).

A convenience function :func:`apply_radial_equalizer` applies the
per-frequency per-order filter to a multi-bin SH coefficient tensor.

For real microphone arrays where a *measured* steering matrix is
available, :func:`measured_array_equalizer` constructs a full per-bin
encoding matrix ``(F, Q, M)`` via regularized least-squares; apply it
with :func:`apply_measured_equalizer`.
"""

from .measured_filters import (
    apply_measured_equalizer,
    measured_array_diagnostics,
    measured_array_equalizer,
)
from .radial_filters import (
    apply_radial_equalizer,
    radial_equalizer,
    radial_equalizer_tikhonov,
    radial_equalizer_wng_limited,
)

__all__ = [
    "apply_measured_equalizer",
    "apply_radial_equalizer",
    "measured_array_diagnostics",
    "measured_array_equalizer",
    "radial_equalizer",
    "radial_equalizer_tikhonov",
    "radial_equalizer_wng_limited",
]
