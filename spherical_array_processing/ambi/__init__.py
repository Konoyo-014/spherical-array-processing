"""Ambisonic format helpers — normalisation and channel-order conversion.

The core of this package uses **orthonormal real / complex** SH (``∫|Y|²
dΩ = 1``) in **ACN** ordering internally.  Most commodity ambisonic
files instead use one of:

* **AmbiX** — ACN ordering with *SN3D* (Schmidt semi-normalised) or
  *N3D* (fully normalised) scaling.  This is the dominant format for
  modern VR/AR tools (Google/YouTube, Reaper, IEM Plug-in Suite, …).
* **FuMa** — B-format *Furse-Malham* ordering and normalisation,
  defined through 3rd order and used by the legacy B-format ecosystem
  (VVAudio, Bruce Wiggins, older VST plug-ins).

This module provides:

* :func:`convert_ambi_normalization` — rescale between
  ``orthonormal`` / ``n3d`` / ``sn3d`` (all ACN-ordered).
* :func:`acn_to_fuma` / :func:`fuma_to_acn` — reorder and rescale
  between ACN-SN3D and FuMa (through 3rd order).

Higher-order FuMa is not standardised; those entry points raise
``ValueError`` past order 3.
"""

from .encoder import encode_plane_wave, encode_plane_wave_frame
from .format import (
    acn_to_fuma,
    convert_ambi_normalization,
    fuma_to_acn,
)
from .intensity import doa_from_intensity, intensity_vector
from .io import read_ambix_frame, read_ambix_wav, write_ambix_frame, write_ambix_wav
from .nfc import nfc_hoa_distance_filter
from .spec import (
    AmbisonicFrame,
    AmbisonicSignalReport,
    AmbisonicSpec,
    ambisonic_signal_report,
    channel_count,
    infer_order,
    order_channel_mask,
    order_channel_slices,
    per_order_energy,
)
from .translation import translate_foa
from .uhj import uhj_decode, uhj_encode

__all__ = [
    "acn_to_fuma",
    "AmbisonicFrame",
    "AmbisonicSignalReport",
    "AmbisonicSpec",
    "ambisonic_signal_report",
    "channel_count",
    "convert_ambi_normalization",
    "doa_from_intensity",
    "encode_plane_wave",
    "encode_plane_wave_frame",
    "fuma_to_acn",
    "infer_order",
    "intensity_vector",
    "nfc_hoa_distance_filter",
    "order_channel_mask",
    "order_channel_slices",
    "per_order_energy",
    "read_ambix_wav",
    "read_ambix_frame",
    "translate_foa",
    "uhj_decode",
    "uhj_encode",
    "write_ambix_wav",
    "write_ambix_frame",
]
