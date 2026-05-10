# Concepts

This page explains the terms that appear in the API. It is deliberately short:
the goal is to make examples readable, not to replace a textbook.

## Directions And Angle Conventions

A spherical array measures a sound field over directions. The package supports
two angle conventions. **Azimuth/elevation** uses azimuth around the horizontal
plane and elevation above that plane. **Azimuth/colatitude** uses the same
azimuth, but the second angle is measured downward from the positive z-axis.
Colatitude is common in spherical harmonics because the mathematical polar angle
is usually written as `theta`.

Most examples use `angle_convention="az_colat"` because SciPy spherical
harmonics and many array-processing formulas use colatitude. If your input data
uses elevation, construct grids with `convention="az_el"` or convert with
`sap.coords.azel_to_az_colat`.

## Spherical Harmonics

**Spherical harmonics** are basis functions on the sphere. They play the same
role for directions that sine and cosine functions play for time signals. A
field sampled over many directions can be represented as a vector of
coefficients. The maximum degree is called the **SH order** `N`, and the number
of coefficients is `(N+1)^2`.

The package stores channels in **ACN order**, short for Ambisonic Channel
Numbering. The index formula is `q = n(n+1) + m`, where `n` is degree and `m` is
order. The helper `sap.sh.acn_index(n, m)` maps degree and order to a channel
index.

## Sampling And Quadrature

A **SphericalGrid** is a collection of directions and optional integration
weights. The weights approximate integration over the full sphere, so their sum
should be close to `4*pi`. Dense Fibonacci grids are convenient and stable for
examples. Exact or near-exact quadrature grids are better when you need high
accuracy in spherical-harmonic transforms.

The forward transform `sap.sh.direct_sht(samples, Y, grid)` projects sampled
values into SH coefficients. The inverse transform `sap.sh.inverse_sht(coeffs,
Y)` synthesizes samples back from coefficients. If the grid is too sparse for
the requested order, the transform becomes an approximation because the samples
do not contain enough spatial information.

## Radial Modal Coefficients

A spherical microphone array does not observe plane-wave SH amplitudes directly.
The sphere radius, frequency, and array type change each SH order by a
frequency-dependent factor called a **modal coefficient** or **radial
coefficient**. The package computes these terms with `sap.acoustics.bn_matrix`
and `sap.acoustics.sph_modal_coeffs`.

When modal coefficients are small, direct inversion amplifies noise. The helper
`sap.acoustics.equalize_modal_coeffs` uses regularization, which means it trades
a small amount of bias for numerical stability. This is necessary near modal
zeros and rigid-sphere resonances.

## Beamforming

An axisymmetric fixed beamformer is described by one weight per SH degree. The
package evaluates its pattern with

```text
B(theta) = sum_n b_n * (2n+1)/(4*pi) * P_n(cos(theta)).
```

All bundled fixed beamformer weights use **unit front gain**, so `B(0) = 1`.
The cardioid family gives broad smooth patterns, the hypercardioid maximizes
directivity index, the supercardioid emphasizes front-to-back energy ratio, and
MaxEV trades directivity for smoother perceptual behavior.

## DOA Estimation

**DOA** means direction of arrival. The package exposes PWD and MUSIC spectra in
the SH domain. PWD is direct and easy to interpret. MUSIC uses a signal/noise
subspace separation, so it can produce sharper peaks when the number of sources
is known and the covariance matrix is well conditioned.

The usual data path is to estimate an SH covariance matrix, evaluate a spatial
spectrum on a search grid, then pick peaks. The examples use synthetic
covariances so the true source direction is known, which makes it easy to check
whether the pipeline is working.

## Diffuseness And Coherence

**Diffuseness** measures whether energy is directionally concentrated or spread
across many directions. **Diffuse-field coherence** predicts how omnidirectional
microphones correlate in a spatially diffuse field. These modules are useful
when you want descriptors of the sound field rather than a single source
direction.
