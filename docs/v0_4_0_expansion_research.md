# v0.4.0 Expansion Research

This note prepares the consolidated **spherical-array-processing 0.4.0** line.
The public duplication between `spherical-array-processing` and `spharray` has a
clear content answer: `spherical-array-processing` is the feature-complete
0.4-series codebase, while `spharray` contributed cleaner governance,
getting-started material, and a few convenience APIs. The merged 0.4.0 release
therefore keeps the feature-complete package name and folds the useful
open-source packaging material into it.

## Baseline Judgment

The package should grow from a **spherical-array math toolkit** into a
**publicly verifiable Ambisonics and spherical microphone array workflow**. The
current 0.4.0 draft covers the first engineering layer: **ACN/SN3D convention
objects**, **source-to-HOA encoding**, **SAD/MAD/EPAD/AllRAD-style loudspeaker
decoding**, **array-to-HOA frequency-domain encoding**, **rotation**, **decoder
metrics**, and a simple **HRIR-to-SH binaural path**. This is enough to expose a
coherent Ambisonics namespace, but it is not enough to match the mature research
and production ecosystems around HOA recording, decoding, and binaural
rendering.

The strongest expansion target is therefore not a loose pile of new functions.
It is a set of tested pipelines that make convention handling, measured arrays,
decoder design, SOFA HRIR rendering, and real-data validation hard to misuse.
The scientific boundary is also important: ideal spherical-array formulas,
measured steering-matrix inversion, and signal-dependent binaural matching are
different evidence classes. They should sit in the same package only if the API
makes those assumptions explicit.

## Ecosystem Lessons

[spaudiopy](https://spaudiopy.readthedocs.io/en/latest/spaudiopy.decoder.html)
is the closest Python reference for user-facing Ambisonics workflows. Its
decoder API exposes **LoudspeakerSetup**, VBAP/VBIP, AllRAD/AllRAD2, SAD, MAD,
EPAD, SH-to-binaural, and MagLS binaural decoding in one visible namespace. The
main lesson for `spherical-array-processing` is that users need a path from matrix construction to
audible rendering and diagnostics. A decoder is not complete just because it
returns a matrix; it should also expose layout checks, energy-vector behavior,
and examples that make the result listenable.

[SpharPy](https://spharpy.readthedocs.io/en/develop/modules/spharpy.spherical.html)
is the strongest reference for spherical-harmonic conventions and sampling
objects. It treats **ACN/FuMa index conversion**, **N3D/SN3D/maxN
renormalization**, **modal strength**, **real and complex SH bases**, gradients,
and sampling layouts as first-class material. The main lesson is that convention
helpers should be boring, explicit, and heavily tested. A package that handles
both spherical arrays and Ambisonics should not let users silently mix
azimuth/elevation with azimuth/colatitude, or N3D with SN3D.

[sound_field_analysis](https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py)
matters because it is a Python port of the SOFiA MATLAB toolbox and is centered
on measured spherical microphone array recordings. Its README describes the
toolbox as a way to analyze, visualize, and process sound fields recorded by
spherical microphone arrays, and as a building block for real-time binaural
rendering. The main lesson is that `spherical-array-processing` needs at least one
publicly reproducible measured-data workflow. Synthetic plane waves are
necessary for tests, but they are not enough to prove that array-to-HOA
encoding is useful on real measurements.

[pyfar](https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/pyfar_coordinates.html)
and [sofar](https://www.sofaconventions.org/mediawiki/index.php/SOFA_specifications)
are important even though they are not HOA-only packages. They show how much
engineering value comes from stable **Signal**, **FrequencyData**,
**Coordinates**, and **SOFA** data models. `spherical-array-processing` can stay lightweight and
NumPy-first, but the Ambisonics layer should carry sample rate, frequency bins,
channel convention, coordinate convention, and HRIR metadata in explicit
dataclasses rather than relying on ndarray shapes alone.

[pyroomacoustics](https://pyroomacoustics.readthedocs.io/) is the broadest
Python reference for room simulation, beamforming, DOA, and adaptive filtering.
It is not an Ambisonics package, but its object model is useful: rooms, sources,
microphone arrays, simulation, and localization are composed as a workflow. This
suggests a future `spherical-array-processing` example layer where Ambisonic receivers, spherical
arrays, and loudspeaker or binaural decoders are used in full acoustic scenes,
not only isolated matrix demos.

The mature non-Python references set the production bar. The
[IEM Plug-in Suite](https://plugins.iem.at/) supports open-source Ambisonic
plug-ins up to seventh order, including encoder, room, decoder, and binaural
workflows. [SPARTA](https://leomccormack.github.io/sparta-site/docs/plugins/overview/)
documents a production suite with **ambiBIN**, **ambiDEC**, **ambiENC**,
**array2sh**, beamforming, SOFA loading, AllRAD, EPAD, MMD, SAD, MagLS, and
head tracking. [libspatialaudio](https://github.com/videolan/libspatialaudio)
shows the value of a unified renderer spanning HOA, object, speaker, and
binaural streams. Politis'
[Higher-Order-Ambisonics](https://github.com/polarch/Higher-Order-Ambisonics)
and spherical-array MATLAB work remain important numeric references for ACN/N3D
conventions, HOA rotation, decoding, and measured array filters.

## Literature Anchors

The core mathematical references for this release should be **Rafaely** for
spherical arrays and **Zotter/Frank/Daniel** for Ambisonics. Rafaely's
[Fundamentals of Spherical Array Processing](https://link.springer.com/book/10.1007/978-3-319-99561-8)
gives the package its foundation in spherical Fourier transforms, spatial
sampling, rigid-sphere arrays, radial modal functions, directivity index, white
noise gain, maximum-directivity beamforming, Dolph-Chebyshev beamforming, MVDR,
and LCMV. Zotter and Frank's open-access
[Ambisonics](https://link.springer.com/book/10.1007/978-3-030-17207-7) book
connects the signal-processing math to encoding, panning, decoding, production,
HOA microphones, compact arrays, psychoacoustic limits, and practical software.
Daniel's thesis entry on
[theses.fr](https://theses.fr/2000PA066581) documents the older but still
central high-order Ambisonics framing: variable-resolution sound-field
representation, geometry-variable loudspeaker or headphone decoding, and
generalized higher-order encoding and decoding families.

For loudspeaker decoding, Zotter, Frank, Pomberger, Noisternig, Epain, and Jin
are the main references. The AES page for
[All-Round Ambisonic Panning and Decoding](https://secure.aes.org/forum/pubs/journal/?elib=16554)
states the key engineering idea: combine Ambisonics with an extended VBAP layer
to work on arbitrary loudspeaker arrangements. The Acta Acustica paper
[Energy-Preserving Ambisonic Decoding](https://doi.org/10.3813/AAA.918490)
addresses non-uniform loudspeaker layouts by preserving decoded energy. The
paper
[Ambisonic Decoding With Constant Angular Spread](https://doi.org/10.3813/AAA.918772)
adds another quality target: keep angular spread nearly constant while
maintaining energy and low energy-vector direction mismatch. These sources map
directly to API concepts such as basic/MAD, SAD, EPAD, AllRAD, AllRAD2-style
projection, max-rE weighting, in-phase weighting, energy-vector metrics, and
spread diagnostics.

For binaural rendering, the baseline should be **SOFA/AES69-compatible HRIR
loading**, **SH-domain HRTF projection**, **time alignment**, and **MagLS**.
SOFA's specification defines FIR data with `Data.IR`, `Data.SamplingRate`, and
mandatory delay fields, so the package must preserve those meanings when it
loads or exports HRIR material. Ahrens' arXiv review
[Binaural Audio Rendering in the Spherical Harmonic Domain](https://arxiv.org/abs/2202.04393)
is useful because it focuses on definitions and pitfalls rather than only final
metrics. Engel, Goodman, and Picinali's
[BiMagLS DAGA paper](https://pub.dega-akustik.de/DAGA_2021/data/articles/000533.pdf)
explains why limited-order Ambisonics struggles with high-order HRTFs, why
time-aligned and magnitude least-squares preprocessing reduce audible
coloration, and why bilateral Ambisonics is a distinct method rather than a
drop-in replacement for a head-centered decoder.

For exchange format, the package should keep modern Ambisonics defaulting to
**ACN/SN3D AmbiX**. The IETF Ambisonics channel-mapping draft describes ACN
ordering as `ACN = n * (n + 1) + m`, mixed-order signaling through inactive
channels, and SN3D normalization for Ambisonic channels. That standard framing
supports the current choice to treat ACN/SN3D AmbiX as the modern exchange
target, while first-order legacy FuMa should remain an explicit conversion path
rather than a hidden default.

## Development Direction

The first development surface is **conventions and containers**. The existing
`ambi`, `hrtf`, `decoding`, `binaural`, and typed grid/frame helpers should
remain the single source of truth for order, normalization, channel ordering,
domain, frequency bins, sample rate, and mixed-order masks. This is the layer
that prevents silent errors in angle convention, axis choice, normalization,
and channel count.

The second development surface is **encoding**. Point-source encoding should
stay simple and exact for ACN/SN3D real Ambisonics. The array-to-HOA path should
grow beyond the current weighted SHT plus radial inverse. It needs explicit
regularization modes, white-noise-gain limits, measured steering matrix
encoders, regLS/regLSHD-style objectives, and validation metrics for aliasing,
condition number, and modal noise amplification. Ideal open or rigid sphere
models and measured transfer-function models should share shape conventions but
remain scientifically separate.

The third development surface is **decoding**. The local draft already has SAD,
MAD/MMD, EPAD, and a deterministic AllRAD-style virtual projection. The mature
0.4.0 target should add max-rE and in-phase weighting as named policies, VBAP
projection for AllRAD, imaginary-loudspeaker closure for hemispherical layouts,
dual-band or frequency-dependent decoder matrices, decoder quality metrics, and
layout diagnostics. Decoder construction should be accompanied by metrics that
answer whether a layout is underdetermined, poorly conditioned, energy-unstable,
or missing vertical coverage.

The fourth development surface is **binaural rendering**. The current
`hrir_to_sh_decoder` path is a useful first pass, but the research bar requires
SOFA metadata handling, delay semantics, time alignment, MagLS, diffuse-field or
spectral equalization options, and an end-to-end Ambisonic-array to stereo
pipeline. BiMagLS can be introduced as an advanced method only if the API
states its bilateral or ear-centered assumptions clearly.

The fifth development surface is **analysis and reproducibility**. DOA,
beamforming, diffuseness, decoder metrics, and room or dataset examples should
be tied together with reproducible tutorials. A serious 0.4.0 should include
synthetic regression tests, a small public measured-data or generated-data
recipe, cross-checks against at least one external library or reference formula,
and install-safe examples that can run after `pip install spherical-array-processing`.

## Validation Strategy

The release gate should begin with convention invariants. ACN index formulas,
channel counts, order inference, SN3D/N3D/orthonormal conversion, FuMa first
order conversion, mixed-order masking, and axis restoration must be tested as
pure functions. These tests catch the errors that are hardest for users to hear
immediately but easiest to propagate through a whole spatial-audio pipeline.

The next gate should test physics-facing invariants. SHT round trips, weighted
quadrature behavior, modal radial coefficients, array simulation, radial
regularization, and array-to-HOA encoding should be tested against analytic
plane waves and high-density numerical references. The package should report
condition number, expected aliasing frequency, and maximum modal gain so users
can see when the requested order or frequency range is outside the credible
region.

The listening-facing gate should focus on decoder behavior. Loudspeaker decoder
tests should verify shape, active-speaker masking, energy-vector direction,
diffuse-level behavior, white-noise gain, and robustness on irregular layouts.
Binaural tests should verify SOFA shape interpretation, HRIR projection, delay
handling, time alignment, MagLS transition behavior, and stable stereo output
from known Ambisonic impulses.

The documentation gate should remain part of release readiness. The new
Ambisonics examples must run from a clean checkout and, where possible, from an
installed wheel. The README should claim only APIs that are present in
`spherical_array_processing.__all__` and covered by tests. The release should avoid describing
advanced research methods such as BiMagLS, ASM, BSM, or signal-dependent
wearable-array rendering as implemented unless the corresponding code, tests,
and assumption docs are in the public package.

## Immediate Preparation

The current uncommitted 0.4.0 draft is a sensible seed release if its scope is
framed as **first Ambisonics public surface**, not as a complete Ambisonics
research stack. Before adding another major algorithm block, the branch should
be made internally consistent: the changelog, README, docs, tests, examples,
package metadata, and `spherical_array_processing.__version__` should all describe the same
release. After that, the expansion can proceed in the order that reduces user
confusion first: convention containers, decoder diagnostics, SOFA/HRIR
robustness, measured array encoding, and finally advanced binaural or
signal-dependent array methods.

This gives the fivefold expansion a concrete shape. The package grows in
capability, but more importantly it grows in **evidence quality**: each added
method comes with an assumption boundary, a known literature anchor, a minimal
reproducible example, and a test that catches the most likely convention or
shape failure.
