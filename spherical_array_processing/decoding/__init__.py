"""Ambisonic decoders — map SH-domain signals to loudspeaker feeds.

Four canonical decoder families are implemented, all producing a
loudspeaker matrix ``D`` of shape ``(n_speakers, (N+1)²)`` such that
``loudspeaker_signal = D @ ambi_signal`` in the ACN / orthonormal SH
convention:

* **SAD** (Sampling Ambisonic Decoder) — ``D = (4π / L) · Y_spk``.
  Correct for regular grids; biased otherwise.
* **MMD** (Mode-Matching Decoder) — ``D = pinv(Y_spk^T)``; minimises
  mode-wise reproduction error.
* **EPAD** (Energy-Preserving Ambisonic Decoder, Zotter & Frank 2012)
  — SVD-based construction that guarantees total energy is preserved
  regardless of loudspeaker layout.
* **AllRAD** (All-Round Ambisonic Decoder, Zotter & Frank 2012) —
  dense virtual-source SAD on a t-design followed by VBAP (Vector Base
  Amplitude Panning) over the convex-hull triangulation of the
  loudspeaker layout.  Robust for irregular layouts.

The public entry point :func:`decoder_matrix` dispatches between these,
and :func:`apply_decoder` is a convenience helper for applying the
resulting matrix to multi-channel / multi-frequency tensors.
"""

from .decoders import (
    allrad_decoder,
    apply_decoder,
    apply_decoder_taper,
    apply_dual_band_decoder,
    check_layout_coverage,
    decoder_matrix,
    decoder_diagnostics,
    decoder_taper_weights,
    dual_band_decoder_matrix,
    epad_decoder,
    frequency_dependent_decoder_matrix,
    in_phase_sh_weights,
    layout_from_directions,
    layout_from_directions_deg,
    layout_itu_5_1,
    layout_itu_7_1_4,
    layout_t_design,
    max_re_sh_weights,
    mmd_decoder,
    sad_decoder,
    suggest_imaginary_loudspeakers,
    vbap_gains,
)

__all__ = [
    "allrad_decoder",
    "apply_decoder",
    "apply_decoder_taper",
    "apply_dual_band_decoder",
    "check_layout_coverage",
    "decoder_matrix",
    "decoder_diagnostics",
    "decoder_taper_weights",
    "dual_band_decoder_matrix",
    "epad_decoder",
    "frequency_dependent_decoder_matrix",
    "in_phase_sh_weights",
    "layout_from_directions",
    "layout_from_directions_deg",
    "layout_itu_5_1",
    "layout_itu_7_1_4",
    "layout_t_design",
    "max_re_sh_weights",
    "mmd_decoder",
    "sad_decoder",
    "suggest_imaginary_loudspeakers",
    "vbap_gains",
]
