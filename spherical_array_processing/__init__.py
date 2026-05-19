"""Pythonic spherical array processing toolkit.

Submodules
----------
coords
    Coordinate conversions (spherical ↔ Cartesian, az/el ↔ az/colat).
sh
    Spherical harmonics: basis matrices, transforms, coefficient
    conversion, and Wigner-D rotation.
acoustics
    Radial functions, modal coefficients, Bessel/Hankel wrappers.
array
    Spatial sampling grids, free-field and SH-domain array simulation.
beamforming
    Fixed (cardioid, hypercardioid, Butterworth, Dolph–Chebyshev) and
    adaptive (MVDR, LCMV) beamformers.
doa
    Direction-of-arrival estimation (PWD, MUSIC spectra).
diffuseness
    Diffuseness estimators (IE, TV, SV, CMD).
coherence
    Diffuse-field coherence models.
encoding
    Radial equalization filters (Tikhonov / WNG-limited) for turning
    raw microphone SHT coefficients into plane-wave-steering signals,
    plus measured-steering-matrix encoders (regLS / regLSHD) for real
    microphone arrays.
decoding
    Ambisonic decoders (SAD / MMD / EPAD / AllRAD) mapping SH signals
    to loudspeaker feeds, with a VBAP helper over the loudspeaker
    convex hull.
stft
    Multichannel ``(F, M, T)`` STFT / ISTFT wrappers around
    ``scipy.signal.stft`` that match the layout of
    :func:`doa.srp_map` and :func:`encoding.apply_radial_equalizer`.
binaural
    SH → binaural rendering filters, including the Schörkhuber-Zotter
    Magnitude Least Squares (MagLS) renderer for low-order ambisonic
    binaural reproduction and its bilateral (BiMagLS) extension, plus
    a one-call end-to-end ``ambi_to_binaural_time_domain`` pipeline
    with optional head-tracked rotation.
ambi
    Ambisonic format converters: N3D ↔ SN3D ↔ orthonormal
    normalisation rescalings, ACN ↔ FuMa channel-order and
    B-format normalisation conversion (through third order),
    AmbiX WAV I/O, and NFC-HOA near-field distance compensation
    filters.
room
    Shoebox image-source RIR simulator, plus ambisonic convolution
    reverb helpers (``convolve_mono_to_ambi``, ``convolve_sh_to_sh``).
covariance
    Shrinkage covariance estimators (Ledoit-Wolf, OAS),
    forward-backward averaging, and diagonal loading for robust DOA
    / adaptive-beamforming pipelines on short-duration SH signals.
dirac
    Directional Audio Coding (Pulkki 2007) — per-bin DOA and
    diffuseness analysis of SH-domain STFTs plus VBAP-based
    loudspeaker resynthesis.
hrtf
    HRTF dataset container plus a minimal SOFA AES69
    ``SimpleFreeFieldHRIR`` loader (requires ``h5py``, installable as
    the ``[hrtf]`` extra) and an analytic ``rigid_sphere_hrtf``
    generator for testbed scenarios with no measured data.
plotting
    Visualization helpers (array geometry, spatial maps, MATLAB-style figures).

Developer-only submodules
-------------------------
The following submodules ship with the distribution for reproduction
and cross-tool verification, but they are **not** part of the stable
public API.  They are not exposed as attributes on the top-level
``spherical_array_processing`` namespace (``sap.repro`` / ``sap.regression``
/ ``sap.experimental`` all raise ``AttributeError``), and their signatures
may change without notice.  Importing any of them once per Python
session emits a single :class:`FutureWarning` so accidental downstream
dependencies are visible immediately.  For library or production code,
prefer the documented public submodules listed above; if you
intentionally use one of these developer-only areas, pin an exact
package version.

``repro``
    Python ports of the Politis / Rafaely / SHT reference MATLAB code
    used for parity checks and regression baselines.  Consume these
    only if you are specifically reproducing published figures or
    MATLAB numerical traces.
``regression``
    Internal tooling for MATLAB / Octave interop and CLI batch runs.
    Used by the repository's own regression jobs.
``experimental``
    Research prototypes (currently the stereo → partial-FOA estimator
    family).  Interfaces are provisional.

Quick start
-----------
>>> import spherical_array_processing as sap
>>> grid = sap.array.fibonacci_grid(100)
>>> spec = sap.SHBasisSpec(max_order=3)
>>> Y = sap.sh.real_matrix(spec, grid)
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.5.0"

from .types import (
    ArrayGeometry,
    FigureReproConfig,
    FigureStyleConfig,
    SHBasisSpec,
    SHCovariance,
    SHSignalFrame,
    SpatialSpectrumResult,
    SphericalGrid,
)

_SUBMODULES = {
    "acoustics",
    "ambi",
    "array",
    "beamforming",
    "binaural",
    "coherence",
    "coords",
    "covariance",
    "decoding",
    "diffuseness",
    "dirac",
    "doa",
    "encoding",
    "hrtf",
    "plotting",
    "room",
    "sh",
    "stft",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

__all__ = [
    # submodules
    "acoustics",
    "ambi",
    "array",
    "beamforming",
    "binaural",
    "coherence",
    "coords",
    "covariance",
    "decoding",
    "diffuseness",
    "dirac",
    "doa",
    "encoding",
    "hrtf",
    "plotting",
    "room",
    "sh",
    "stft",
    # types
    "ArrayGeometry",
    "FigureReproConfig",
    "FigureStyleConfig",
    "SHBasisSpec",
    "SHCovariance",
    "SHSignalFrame",
    "SpatialSpectrumResult",
    "SphericalGrid",
]
