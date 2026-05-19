# v0.5.0 Ambisonics Release Plan

Version 0.5.0 is the first release that treats Ambisonics as a typed workflow
rather than a set of unrelated matrices. The stable scope is deliberately
evidence-facing: source encoding, loudspeaker decoding, measured-array encoding,
SOFA-backed binaural paths, room helpers, and diagnostic reports should agree on
channel count, normalization, axis placement, frequency layout, and failure
messages.

The API centerpieces are `ambi.AmbisonicSpec`, `ambi.AmbisonicFrame`,
`decoding.decoder_diagnostics`, `decoding.frequency_dependent_decoder_matrix`,
and `encoding.measured_array_diagnostics`. They do not replace the lower-level
NumPy APIs. They make the assumptions visible so a caller can tell whether a
signal is ACN/SN3D or orthonormal, whether a decoder is max-rE or in-phase
tapered, whether a loudspeaker layout leaves a large uncovered region, and
whether a measured array inversion is numerically fragile.

The release gate is stricter than the 0.4.0 gate. The version string in
`pyproject.toml`, `spherical_array_processing.__version__`, the changelog, the
README, the GitHub tag, and the PyPI package must all say `0.5.0`. The complete
repository test suite must pass before tagging. The built wheel must keep the
developer-only `repro`, `regression`, and `experimental` layers out of the
runtime install, while the sdist may keep them for reproduction.

The PyPI boundary is operational rather than scientific. GitHub release
publication can use the authenticated `gh` CLI on this machine. PyPI publication
requires a valid upload token or equivalent trusted-publisher configuration. If
no credential is present at release time, the package can be built, verified,
tagged, and attached to GitHub, but PyPI cannot honestly be called synchronized
until upload succeeds and the PyPI JSON endpoint reports version `0.5.0`.
