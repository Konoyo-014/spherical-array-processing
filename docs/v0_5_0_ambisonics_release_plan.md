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

The post-audit expansion also adds non-matrix layers that make the release more
useful in real work. `ambi.ambisonic_signal_report` and `ambi.per_order_energy`
give a convention-aware health check for Ambisonic tensors before they are
decoded or written to disk. `ambi.read_ambix_frame` and `ambi.write_ambix_frame`
connect the typed workflow to AmbiX WAV files without discarding convention
metadata. `decoding.DecoderConfig` gives decoder matrices a JSON-serialisable
configuration object rather than leaving reproducibility to unnamed arrays.
`room.statistical` adds traditional room-acoustics prediction formulas: Sabine,
Eyring, Millington-Sette, Arau-Puchades, Schroeder frequency, critical distance,
rectangular-room modes, and ISO 9613-1 atmospheric absorption coefficients. The
documentation deliberately calls these prediction utilities rather than
standards-compliant measurement functions.

The literature and package comparison for follow-on 0.5.x work points to
configuration and interoperability as the next high-yield layer: AllRAD2 /
AllRAP-style decoding, AmbDec and IEM-style JSON import/export, SOFA validation
via an optional `sofar` backend, HRTF interpolation and diffuse-field
equalization, richer decoder-performance objects, and NFC-HOA driving functions.
IEC 60268-16 STI/STIPA remains out of stable scope until the complete
modulation-transfer and standard-test-signal chain is implemented and validated.
