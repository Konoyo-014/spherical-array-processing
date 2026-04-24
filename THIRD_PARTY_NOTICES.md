# Third-Party Notices

This repository contains a Python package plus repository-only reference assets
used during the MATLAB-to-Python migration.  The runtime package distributed as
`spherical-array-processing` does not require the MATLAB source trees below.

## MATLAB Reference Materials

### `src/Array-Response-Simulator`

Source: Archontis Politis, Sensor Array Response Simulator.

Purpose in this repository: reference implementation for array response,
spherical/cylindrical scattering, modal coefficients, and regression checks.

The original source tree includes its own README and license file:
`src/Array-Response-Simulator/README.md` and
`src/Array-Response-Simulator/LICENSE`.

### `src/Spherical-Harmonic-Transform`

Source: Archontis Politis, Spherical Harmonic Transform Library.

Purpose in this repository: reference implementation for SHT utilities,
spherical harmonic bases, sampling resources, and regression checks.

The original source tree includes its own README and license file:
`src/Spherical-Harmonic-Transform/README.md` and
`src/Spherical-Harmonic-Transform/LICENSE`.

### `src/Spherical-Array-Processing-MATLAB`

Source: Archontis Politis, Spherical-Array-Processing MATLAB routines.

Purpose in this repository: reference implementation for spherical array
processing algorithms, beamforming, diffuseness, coherence, and DOA parity
checks.

The original source tree includes its own README and license file:
`src/Spherical-Array-Processing-MATLAB/README.md` and
`src/Spherical-Array-Processing-MATLAB/LICENSE.md`.

### `src/Rafaely`

Source: Rafaely-style MATLAB math, plotting, and figure scripts used as
reference material for spherical array processing examples.

Purpose in this repository: reproduction and figure-regression workbench for
book-style examples and spherical harmonics/radial-function parity checks.

## Distribution Boundary

Only the Python package under `spherical_array_processing/` is included in the
runtime wheel.  Source distributions include a package-only test subset and do
not include the large MATLAB source trees or generated regression artifacts.

If third-party reference trees are published in a public Git repository, keep
their license files and README files intact.  Prefer git submodules or an
external data-fetch step for large reference assets instead of committing
nested `.git` directories or generated artifacts into the main repository.
