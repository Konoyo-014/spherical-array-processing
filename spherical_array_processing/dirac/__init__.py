"""Directional Audio Coding (DirAC) analysis and synthesis.

DirAC (Pulkki 2007) describes a sound field in the time-frequency
domain by two parameters per bin:

* the **direction of arrival** of the dominant sound component, and
* the **diffuseness** ``ψ ∈ [0, 1]`` that quantifies how much of the
  bin's energy is isotropic.

Synthesis re-renders the signal to a chosen loudspeaker layout by
amplitude-panning the *direct* part toward the estimated DOA (via VBAP
over the loudspeaker convex hull) and spreading the *diffuse* part
uniformly (optionally decorrelated) across the loudspeakers.  The
framework is the standard parametric spatial audio pipeline for low
order ambisonic systems.

This module assumes the input is an STFT with ACN / real SH layout of
at least first order (``(N+1)² ≥ 4`` so that W/X/Y/Z are present).
For textbook DirAC diffuseness values the first-order channels should
use **SN3D** scaling; with orthonormal FOA the DOA estimate remains
correct but the diffuseness value is biased.
"""

from .analysis import DirACParameters, dirac_analysis
from .pipeline import dirac_render_time_domain
from .synthesis import dirac_synthesize

__all__ = [
    "DirACParameters",
    "dirac_analysis",
    "dirac_render_time_domain",
    "dirac_synthesize",
]
