# Ambisonics Concepts

Ambisonics represents a sound field with **spherical-harmonic channels**. The
modern file-exchange default is usually **ACN/SN3D AmbiX**: ACN fixes channel
order with `q = n(n+1)+m`, while SN3D fixes the per-degree scaling used by many
VR and DAW toolchains. The package's mathematical core uses orthonormal SH
bases, so `spherical_array_processing.ambi.convert_ambi_normalization` exists
to make format boundaries explicit.

**FuMa/B-format** is the legacy first-order layout `[W, X, Y, Z]`. AmbiX first
order is ACN ordered as `[W, Y, Z, X]`. The package exposes
`ambi.fuma_to_acn` and `ambi.acn_to_fuma` for that compatibility path, while
higher-order workflows should stay in ACN ordering.

**HOA** means higher-order Ambisonics. Order `N` has `(N+1)^2` channels. Higher
orders increase spatial detail, but they require more microphones, more stable
sampling grids, higher loudspeaker counts, and stronger regularization in the
radial filters.

There are three main Ambisonic paths in this package. The source path uses
`ambi.encode_plane_wave` to turn monaural source signals into Ambisonic
channels. The microphone-array path uses `encoding.radial_equalizer` or
`encoding.measured_array_equalizer` after an SH transform to compensate spherical
array modal responses. The playback path uses `decoding.decoder_matrix` plus
`decoding.apply_decoder` for loudspeakers, or `binaural.ambi_to_binaural_time_domain`
for headphones.

Loudspeaker decoding supports **SAD**, **MMD**, **EPAD**, and **AllRAD**.
**max-rE** weighting is available through the dual-band decoder helpers and
improves perceptual localization stability by tapering higher orders. Partial
or hemispherical loudspeaker layouts should be treated carefully; use the
imaginary-loudspeaker and coverage helpers before trusting an irregular layout.

Binaural rendering uses HRIR/HRTF data. The package can load SOFA
`SimpleFreeFieldHRIR` files through `hrtf.load_sofa`, project HRIRs into SH
filters, and run MagLS-style binaural rendering. SOFA delay metadata and
normalization choices are part of the signal contract, not incidental file
details.
