# Ambisonics Codecs

The codec API is intentionally matrix oriented. Plane-wave source encoding
creates Ambisonic channels from source signals and directions. Loudspeaker
decoding maps Ambisonic channels to loudspeaker feeds. Array encoding maps
microphone-array STFT bins to SH/Ambisonic channels after angular SHT and radial
or measured-array equalization.

The source encoder is `spherical_array_processing.ambi.encode_plane_wave`.
Inputs are monaural signals plus azimuth/elevation or azimuth/colatitude
directions. The output follows the requested SH order, ACN ordering, and
normalization convention. Since 0.5.0, `ambi.AmbisonicSpec` and
`ambi.AmbisonicFrame` provide a typed container path for code that needs to
preserve order, normalization, domain, sample rate, frequency axis, and
mixed-order masks. `ambi.encode_plane_wave_frame` is the frame-returning variant
of the low-level encoder.

The loudspeaker decoder entry point is
`spherical_array_processing.decoding.decoder_matrix`. It supports `sad`, `mmd`,
`mad`/`mmd`, `epad`, and `allrad`. The matrix is applied with
`decoding.apply_decoder`. Layouts are ordinary `SphericalGrid` objects, so the
same coordinate convention rules used by the SH module apply here. The decoder
module also provides `layout_itu_5_1`, `layout_itu_7_1_4`, `layout_t_design`,
`frequency_dependent_decoder_matrix`, `decoder_taper_weights`, and
`decoder_diagnostics`. The diagnostics report is the recommended pre-flight
check for layout-specific decoders because it exposes rank, condition number,
diffuse-level error, coverage gaps, and velocity/energy-vector behavior.

The spherical-array encoder path starts with microphone data projected into SH
coefficients. `encoding.radial_equalizer` builds regularized open, rigid,
cardioid, or directional-array radial filters. `encoding.measured_array_equalizer`
is the measured steering-matrix path for anechoic transfer data. These two paths
should not be conflated: the first is an analytic sphere model, while the second
is a calibration-data inversion problem. Use
`encoding.measured_array_diagnostics` to report condition numbers, WNG ranges,
maximum filter gain, and SH reconstruction residuals for measured encoders.

File and headphone helpers live in separate modules. `ambi.read_ambix_wav` and
`ambi.write_ambix_wav` cover WAV interop when the `audio` extra is installed.
`hrtf.load_sofa` and `hrtf.save_sofa` cover SOFA HRIR files when the `hrtf`
extra is installed. `binaural.magls_binaural_filters`,
`binaural.bimagls_binaural_filters`, and `binaural.ambi_to_binaural_time_domain`
cover SH-to-stereo rendering workflows.
