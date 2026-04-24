"""Binaural rendering filters from spherical-harmonic ambisonic signals.

The v0.4.0b3 release ships two complementary routines.  Magnitude Least
Squares (MagLS, Schörkhuber & Zotter 2018) is the baseline low-order
Ambisonic binaural renderer, and Bilateral Ambisonics MagLS (BiMagLS,
Engel et al. 2021) extends it with explicit per-ear delay alignment for
better low-order ITD behaviour.
"""

from .bimagls import bimagls_binaural_filters
from .magls import magls_binaural_filters
from .pipeline import ambi_to_binaural_time_domain

__all__ = [
    "ambi_to_binaural_time_domain",
    "bimagls_binaural_filters",
    "magls_binaural_filters",
]
