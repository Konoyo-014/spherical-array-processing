# v0.6.0 Fivefold Expansion Audit

This document records the release boundary for `spherical-array-processing`
0.6.0.  Version 0.5.0 was already published to GitHub and PyPI, so the
fivefold corrective expansion is released as 0.6.0 rather than by trying
to overwrite an immutable PyPI artifact.

## Baseline and Count Method

The original local package baseline is
`/Users/konoyo/Desktop/CodexWorkspace/Spherical_Array_Process_Python`.
Developer-only reproduction, regression, experimental, and example
modules are excluded from the stable implementation count.  The same
exclusion rule is applied to the 0.6.0 tree.

| Count metric | Original package | v0.6.0 | Multiplier |
| --- | ---: | ---: | ---: |
| Stable top-level implementation definitions | 95 | 481 | 5.06x |
| Stable implementation definitions plus public dataclass methods | 100 | 503 | 5.03x |
| Public exported names summed across stable modules | 89 | 587 | 6.60x |

These are mechanical counts, not a scientific claim about equivalent
feature difficulty.  They are used here as a release-readiness guardrail:
the public surface and the implementation surface both now pass the
fivefold threshold against the original package snapshot.

## Scope Added After the v0.5.0 Publication

The largest change is the Ambisonics expansion.  The package now has a
channel-semantics layer for ACN/FuMa labels, FuMa permutations, mixed
order masks, channel selection, channel energy, active-channel detection,
and per-order summaries.  It also has signal-level Ambisonic processing
helpers for order truncation/padding, per-channel and per-order gains,
W-channel extraction, covariance/correlation, normalization, mixing, and
crossfading.

Decoder work moved beyond constructing a matrix.  The new diagnostics
cover singular values, rank, condition number, row/column power,
diffuse speaker level spread, `D^H D` energy preservation,
mode-leakage ratio, mode-matching response, probe loudspeaker responses,
energy-vector and velocity-vector behavior, vector angular error, and
compact health reports.  This aligns the API with the classical
Ambisonics distinction between pressure/velocity behavior at low
frequency and energy-vector behavior at high frequency.

The traditional acoustics layer was expanded so Ambisonics and array
work can share a common measurement vocabulary.  It now covers SPL, SIL,
PWL, energetic averaging, fractional-octave bands, IEC-style A/C/Z
weighting curves, dry-air sound speed and impedance, plane-wave
pressure/intensity/particle-velocity relations, propagation delay,
phase wrapping, Doppler shifts, pitch/frequency scales, phon/sone
conversion, and band-edge/Q conversion.

The real-measurement path now has measured transfer-function containers,
gain/phase mismatch diagnostics, steering-matrix rank and condition
reports, transfer normalization, regularized inverse banks, fitted
sphere calibration, rigid alignment, loudspeaker layout presets and
diagnostics, cross-module localization/spectral/phase metrics, and
JSON-safe interop for core data containers.

## Ecosystem Targets Used for Scope

The Ambisonics and spatial-audio package comparison used the following
external targets as reference points:

| Area | Reference target | API consequence in 0.6.0 |
| --- | --- | --- |
| Ambisonic decoders | spaudiopy decoder APIs, IEM AllRADecoder, SPARTA/IEM practical decoder workflows | Keep SAD/MMD/EPAD/AllRAD, then add diagnostics for energy preservation, mode matching, speaker loading, and vector behavior. |
| Spherical-array processing | spharpy, sound_field_analysis-py, Rafaely-style SHT workflows | Preserve SH transforms and radial filters, then add measured-array diagnostics and calibration utilities. |
| File/container interop | SOFA/AES69, AmbiX/ACN-SN3D practice, RFC 8486 channel identification | Keep SOFA and AmbiX helpers, then add JSON-safe package-object interchange. |
| Room acoustics | ISO 3382-family metrics and classical Sabine/Eyring/Schroeder formulas | Keep RIR metrics, then add design and classical acoustics utilities used before measured validation. |

## Literature and Standards Anchors

The traditional literature boundary for this release is Gerzon's
periphony/Ambisonics work, Daniel's HOA and near-field compensation
framework, Rafaely's spherical-array formulation, Poletti's unified
spatial sound-field recording/reproduction view, Zotter and Frank's
Ambisonics decoder theory, Allen-Berkley shoebox RIR modelling,
ISO 3382 room-acoustics metrics, IEC 61260 fractional-octave filters,
IEC 61672 sound-level weighting, ISO 266 preferred frequencies, AES69
SOFA, ITU-R BS.2051 immersive loudspeaker layouts, EBU ADM HOA
metadata, and RFC 8486 Ambisonics channel mapping.

The package does not claim instrument compliance with IEC/ISO
procedures.  It implements mathematical building blocks and diagnostics:
complete standard-compliant measurements still require calibrated
hardware, measurement procedures, filter tolerances, uncertainty
budgets, and reporting rules outside this Python package.

## Validation Boundary

The 0.6.0 release requires the full repository test suite to pass, plus
wheel/sdist build and source-distribution file checks.  The new tests
added for the fivefold expansion cover classical acoustics formulas,
Ambisonics channel semantics, Ambisonic processing, decoder diagnostics,
array diagnostics, calibration, measurement, metrics, layouts, room
design, room metrics, and interop round trips.

## Useful External Links

spaudiopy decoder documentation: https://spaudiopy.readthedocs.io/en/latest/spaudiopy.decoder.html

IEM AllRADecoder documentation: https://plugins.iem.at/docs/allradecoder/

pyroomacoustics room simulation documentation: https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html

AES69 SOFA standard page: https://www.aes.org/publications/standards/search.cfm?docID=99

ISO 3382-1 room acoustic parameters: https://www.iso.org/standard/40979.html

ISO 266 preferred frequencies: https://www.iso.org/standard/17426.html

IEC 61672 sound level meters: https://webstore.iec.ch/en/publication/26771

RFC 8486 Ambisonics channel mapping: https://www.rfc-editor.org/rfc/rfc8486
