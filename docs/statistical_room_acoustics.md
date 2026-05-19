# Statistical Room Acoustics

This page documents the traditional room-acoustics prediction helpers added in
0.5.0. These functions are **engineering estimates**, not a claim of
ISO/IEC-compliant measurement software. Use `room.rir_metrics` when you have a
measured or simulated impulse response. Use the formulas here when you have
geometry, surface areas, absorption coefficients, and atmospheric conditions.

## Diffuse-Field RT60 Prediction

`room.sabine_rt60(volume_m3, surface_areas_m2, absorption)` implements the
classical Sabine estimate from equivalent absorption area. It is best treated as
a low-absorption, approximately diffuse-field baseline.

`room.eyring_rt60(...)` keeps the logarithmic energy-loss term and is usually
better behaved at higher absorption, while still assuming a diffuse statistical
field.

`room.millington_sette_rt60(...)` applies the logarithmic loss surface by
surface. It is useful when absorption varies strongly across surfaces, but the
same diffuse-reflection assumptions still matter.

`room.arau_puchades_rt60(dimensions_m, absorption)` is a shoebox-room helper for
asymmetric absorption. It computes three directional Eyring times from opposite
wall pairs and combines them as an area-weighted geometric mean. With uniform
absorption it reduces to Eyring exactly.

## Shoebox Helpers

`room.shoebox_surface_areas(dimensions_m)` returns the six wall areas in
`[-x, +x, -y, +y, -z, +z]` order. `room.shoebox_axis_surface_areas(...)` groups
opposite walls by axis, and `room.shoebox_volume(...)` returns the volume.

`room.shoebox_acoustic_stats(...)` bundles the common summary values:
equivalent absorption area, mean absorption, room constant, Sabine/Eyring/
Millington-Sette/Arau-Puchades RT60, Schroeder frequency, and critical distance.

## Modal And Distance Utilities

`room.rectangular_room_modes(dimensions_m, max_frequency_hz)` returns
`nx, ny, nz, frequency_hz` rows for rigid rectangular-room modes up to a cutoff.
`room.classify_room_modes(...)` labels the corresponding indices as axial,
tangential, or oblique.

`room.schroeder_frequency(rt60_s, volume_m3)` implements the common empirical
transition estimate `2000 * sqrt(T60 / V)`. Below this region, individual room
modes dominate. Above it, statistical assumptions become more plausible.

`room.critical_distance(...)` estimates the distance where direct and diffuse
reverberant energy are equal from room constant and source directivity.
`room.critical_distance_from_rt60(...)` uses the Sabine-equivalent absorption
area implied by RT60. Both are diffuse-field approximations and should not be
treated as hard boundaries in small or strongly non-diffuse rooms.

## Atmospheric Absorption

`room.air_absorption_coefficient_iso9613(frequencies_hz, temperature_c=20,
relative_humidity=0.5, pressure_kpa=101.325)` returns an ISO 9613-1-style
atmospheric absorption coefficient in **dB/m**. Keeping it as a standalone
coefficient avoids mixing pressure attenuation, energy attenuation, and room
reverberation units in one opaque formula.

`room.air_absorption_attenuation_iso9613(...)` multiplies that coefficient by a
distance in metres and returns the corresponding loss in dB.

## Explicit Non-Scope

0.5.0 does **not** expose a stable `sti` or `stipa` function. IEC 60268-16 STI
requires octave-band filtering, modulation-transfer functions, apparent SNR,
speech weighting, redundancy correction, masking/noise/level handling, and
standard test-signal constraints. A partial helper with a compliant-looking name
would be misleading, so STI belongs in a future experimental module until the
full chain is implemented and validated.
