# spherical-array-processing

A self-contained Python toolkit for spherical microphone array signal processing.

This repository is organised as an open-source Python package plus a
repository-only reproduction workbench for MATLAB parity checks.  The stable
installable package lives under `spherical_array_processing/`.  The
`scripts/`, `examples/`, `docs/`, and `src/` directories are development and
reproduction assets used by the repository test suite; they are intentionally
not part of the runtime wheel.

## Features

- **Spherical harmonics** — complex and real SH basis matrices, ACN/orthonormal normalisation, ACN↔degree/order helpers, forward and inverse SHTs, coefficient conversions, and **Wigner-D rotation of SH-domain signals** with both the classical Sakurai summation and a numerically-stable ``J_y``-eigendecomposition backend (`method="jy"`) for high-order rotations.
- **Spatial sampling** — Fibonacci, Gauss-Legendre (exact quadrature), and t-design grids with integration weights.
- **Array simulation** — free-field plane-wave transfer functions **and closed-form modal-coefficient simulators for open / rigid / cardioid / arbitrary-α directional spherical arrays** (`simulate_sh_array_response`).
- **Array presets** — ready-to-use geometries for the **Eigenmike-32** rigid-sphere array, **tetrahedral A-format** 4-mic, cubic 8-mic, and evenly-spaced circular arrays.
- **Fixed beamforming** — cardioid, hypercardioid (max DI), supercardioid (max front-back ratio), MaxEV, Butterworth, and spherical Dolph-Chebyshev weight vectors.
- **Adaptive beamforming** — MVDR and LCMV in SH domain.
- **DOA estimation** — narrow-band PWD and MUSIC spatial spectra, broadband SRP / SRP-PHAT (mic-domain and covariance-based), **closed-form spherical ESPRIT** (`esprit_doa`), and AIC / MDL **source-count estimators** (`estimate_n_sources`).
- **Ambisonic encoding** — Tikhonov and WNG-limited inverse filters (`encoding.radial_equalizer`) for turning raw microphone SHT into plane-wave-steering signals, plus **measured-steering-matrix encoders** (`encoding.measured_array_equalizer`, regLS / regLSHD) when an anechoic-grid array measurement is available.
- **Ambisonic decoding** — `sad`, `mmd`, `epad`, and `allrad` (VBAP-based) loudspeaker decoders plus the unified `decoder_matrix` entry point, **dual-band `(D_lf, D_hf)` decoders** with max-rE tapering (`dual_band_decoder_matrix` / `apply_dual_band_decoder`), and **imaginary-loudspeaker hull closure** (`suggest_imaginary_loudspeakers`) for hemispherical / partial layouts.
- **Binaural rendering** — Magnitude Least Squares (MagLS, Schörkhuber-Zotter 2018) and its bilateral-Ambisonics extension (BiMagLS, Engel et al. 2021) that time-aligns HRTFs around each ear for better low-order reproduction (`binaural.magls_binaural_filters`, `binaural.bimagls_binaural_filters`), plus an end-to-end one-call renderer `binaural.ambi_to_binaural_time_domain` with optional head-tracking.  HRTFs can be loaded from SOFA AES69 `SimpleFreeFieldHRIR` files via `hrtf.load_sofa` (requires the optional `[hrtf]` extra) or synthesised analytically with `hrtf.rigid_sphere_hrtf` for testbed scenarios.
- **Ambisonic format converters** — `ambi.convert_ambi_normalization` (N3D ↔ SN3D ↔ orthonormal) and `ambi.acn_to_fuma` / `ambi.fuma_to_acn` for the AmbiX ↔ legacy B-format round trip through third order, plus `ambi.read_ambix_wav` / `ambi.write_ambix_wav` for WAV file interop and `ambi.nfc_hoa_distance_filter` for near-field distance compensation.
- **Room convolution** — `room.convolve_mono_to_ambi` (dry-mono → ambi-reverb via an SH-RIR, e.g. from `shoebox_sh_rir`) and `room.convolve_sh_to_sh` (channel-diagonal SH-RIR application).
- **Acoustic metrics** — ISO 3382 descriptors (RT60 via T20/T30/T60 regression, EDT, C50, C80, D50, EDC) via `room.rir_metrics` or the individual `room.reverberation_time` / `room.clarity` / `room.definition` / `room.early_decay_time` / `room.energy_decay_curve`.
- **Frequency-dependent shoebox** — `room.shoebox_rir_banded` with per-wall per-octave reflection magnitudes for realistic spectral decay modelling.
- **Plane-wave encoding** — `ambi.encode_plane_wave` to turn one or more monaural source signals into an ambisonic mixdown at the chosen directions.
- **DirAC parametric audio** — per-bin DOA / diffuseness analysis and loudspeaker resynthesis of SH-domain STFTs (`dirac.dirac_analysis`, `dirac.dirac_synthesize`).
- **Room simulation** — shoebox image-source monaural and Ambisonic RIR generator (`room.shoebox_rir`, `room.shoebox_sh_rir`) for reverberant DOA / beamforming test vectors.
- **Head-tracking** — time-varying Wigner-D rotation of an ambisonic signal with linear crossfade between keyframes (`sh.rotate_ambi_over_time`).
- **Covariance regularisers** — Ledoit-Wolf and OAS shrinkage, forward-backward averaging, and trace-relative diagonal loading for robust DOA / adaptive beamforming on short SH records (`covariance.*`).
- **STFT utilities** — multichannel `(F, M, T)` layout wrappers compatible with the SRP and encoding APIs (`sap.stft`).
- **Diffuseness** — IE, TV, SV, and CMD estimators from FOA / SH covariance.
- **Coherence** — sinc-based diffuse-field coherence models.
- **Acoustics** — spherical Bessel, Neumann, and Hankel functions, wavenumber helpers, modal coefficients for open/rigid/cardioid/directional arrays.
- **Plotting** — 3-D array geometry, 2-D spatial maps, MATLAB-like figure style.

## Installation

```bash
pip install spherical-array-processing        # from PyPI (when published)
pip install -e ".[dev]"                        # editable install for development
```

Python >= 3.9. Core dependencies: `numpy`, `scipy`.
Install the `plotting` extra if you want `sap.plotting` entry points.

| Extra      | Adds                          |
|------------|-------------------------------|
| `audio`    | `soundfile` for WAV I/O       |
| `hrtf`     | `h5py` for SOFA HRTF I/O      |
| `plotting` | `matplotlib` for visualisation (`sap.plotting`) |
| `image`    | `scikit-image` for image utils |
| `dev`      | `build`, `pytest`, `pytest-cov` |

## Quick start

```python
import numpy as np
import spherical_array_processing as sap

# -- Spatial sampling --
grid = sap.array.gauss_legendre_sampling(3)
print(f"weights sum: {grid.weights.sum():.4f}")  # 4*pi

# -- SH basis and transforms --
spec = sap.SHBasisSpec(max_order=3)  # defaults: complex basis, az_colat convention
Y = sap.sh.matrix(spec, grid)            # (32, 16) complex SH matrix
nm = np.random.randn(spec.n_coeffs) + 1j * np.random.randn(spec.n_coeffs)
f = sap.sh.inverse_sht(nm, Y)            # synthesize a band-limited field
nm_rec = sap.sh.direct_sht(f, Y, grid)   # weighted least-squares recovery
print(np.max(np.abs(nm - nm_rec)))       # ~ machine precision on exact quadrature grids

# -- Fixed beamforming --
b = sap.beamforming.beam_weights_hypercardioid(3)
theta = np.linspace(0, np.pi, 181)
pattern = sap.beamforming.axisymmetric_pattern(theta, b)

# -- Array simulation (open-sphere free-field) --
sensor_grid = sap.array.fibonacci_grid(32)
geometry = sap.ArrayGeometry(radius_m=0.042, sensor_grid=sensor_grid)
freqs, H = sap.array.simulate_plane_wave_array_response(
    fft_len=256, fs=16000.0, geometry=geometry,
    source_grid=sap.array.fibonacci_grid(4),
)
# H.shape == (129, 32, 4) — (n_bins, n_mics, n_sources)

# -- Rigid-sphere array simulation via modal coefficients --
em32 = sap.array.em32_eigenmike()
freqs, H_rigid = sap.array.simulate_sh_array_response(
    fft_len=256, fs=16000.0, geometry=em32,
    source_grid=sap.array.fibonacci_grid(1),
    max_order=6, array_type="rigid",
)

# -- SH-domain signal rotation (head-tracking, scene rotation) --
alpha, beta, gamma = 0.3, 0.2, 0.1                    # ZYZ Euler angles
nm_rot = sap.sh.rotate_sh_coeffs(nm, spec.max_order, alpha, beta, gamma)

# -- Radial encoding filters (Tikhonov / WNG-limited) --
from spherical_array_processing.encoding import radial_equalizer
kr_vec = sap.acoustics.kr(freqs, radius_m=em32.radius_m)
eq = radial_equalizer(6, kr_vec, array_type="rigid",
                      regularization="tikhonov", tikhonov_lambda=0.01)

# -- DOA estimation (broadband SRP-PHAT) --
scan = sap.array.fibonacci_grid(2000)
result = sap.doa.srp_map(H[:, :, 0], freqs, geometry, scan,
                          weighting="phat", freq_range_hz=(500, 6000))
print("SRP-PHAT peak direction (rad):", result.peak_dirs_rad[0])
```

## Module reference

| Module             | Key symbols |
|--------------------|-------------|
| `sap.sh`           | `matrix`, `complex_matrix`, `real_matrix`, `direct_sht`, `inverse_sht`, `complex_to_real_coeffs`, `real_to_complex_coeffs`, `acn_index`, `acn_to_nm`, `degree_order_pairs`, `replicate_per_order`, `wigner_small_d` (`method={"sakurai","jy"}`), `wigner_D`, `sh_rotation_matrix_complex`, `sh_rotation_matrix_real`, `rotate_sh_coeffs`, `rotate_ambi_over_time` |
| `sap.array`        | `fibonacci_grid`, `gauss_legendre_sampling`, `equiangle_sampling`, `get_tdesign_fallback`, `simulate_plane_wave_array_response`, `simulate_sh_array_response`, `em32_eigenmike`, `tetrahedral_array`, `cubic_array`, `circular_array` |
| `sap.acoustics`    | `besseljs`, `besseljsd`, `besselys`, `besselysd`, `besselhs`, `besselhsd`, `besselhs2`, `besselhs2d`, `wavenumber`, `kr`, `plane_wave_radial_bn` (open/rigid/cardioid/directional with `dir_coeff`), `bn_matrix`, `sph_modal_coeffs` |
| `sap.beamforming`  | `beam_weights_cardioid`, `beam_weights_hypercardioid`, `beam_weights_supercardioid`, `beam_weights_maxev`, `beam_weights_butterworth`, `beam_weights_dolph_chebyshev`, `normalize_axisymmetric_weights`, `axisymmetric_pattern`, `mvdr_weights`, `lcmv_weights` |
| `sap.doa`          | `pwd_spectrum`, `music_spectrum`, `esprit_doa`, `srp_map`, `srp_map_from_covariance`, `estimate_n_sources`, `peak_pick_spectrum`, `peak_pick_spectrum_nms`, `spatial_spectrum_from_map` |
| `sap.encoding`     | `radial_equalizer`, `radial_equalizer_tikhonov`, `radial_equalizer_wng_limited`, `apply_radial_equalizer`, `measured_array_equalizer`, `apply_measured_equalizer` |
| `sap.room`         | `ShoeboxRoom`, `shoebox_rir`, `shoebox_sh_rir`, `shoebox_rir_banded`, `convolve_mono_to_ambi`, `convolve_sh_to_sh`, `rir_metrics`, `RIRMetrics`, `energy_decay_curve`, `reverberation_time`, `early_decay_time`, `clarity`, `definition`, `fdn_reverb`, `fdn_sh_tail` |
| `sap.decoding`     | `decoder_matrix`, `apply_decoder`, `sad_decoder`, `mmd_decoder`, `epad_decoder`, `allrad_decoder`, `vbap_gains`, `suggest_imaginary_loudspeakers`, `check_layout_coverage`, `dual_band_decoder_matrix`, `apply_dual_band_decoder`, `max_re_sh_weights` |
| `sap.binaural`     | `magls_binaural_filters`, `bimagls_binaural_filters`, `ambi_to_binaural_time_domain` |
| `sap.hrtf`         | `HRTFDataset`, `load_sofa`, `save_sofa`, `rigid_sphere_hrtf` |
| `sap.ambi`         | `convert_ambi_normalization`, `acn_to_fuma`, `fuma_to_acn`, `read_ambix_wav`, `write_ambix_wav`, `nfc_hoa_distance_filter`, `encode_plane_wave`, `uhj_encode`, `uhj_decode`, `intensity_vector`, `doa_from_intensity`, `translate_foa` |
| `sap.covariance`   | `ledoit_wolf_shrinkage`, `oas_shrinkage`, `forward_backward_average`, `diagonal_loading` |
| `sap.dirac`        | `dirac_analysis`, `dirac_synthesize`, `dirac_render_time_domain`, `DirACParameters` |
| `sap.stft`         | `stft`, `istft` |
| `sap.diffuseness`  | `diffuseness_ie`, `diffuseness_tv`, `diffuseness_sv`, `diffuseness_cmd`, `intensity_vectors_from_foa` |
| `sap.coherence`    | `diffuse_coherence_matrix_omni`, `diffuse_coherence_from_weights` |
| `sap.coords`       | `sph_to_cart`, `cart_to_sph`, `azel_to_az_colat`, `az_colat_to_azel`, `unit_sph_to_cart` |
| `sap.plotting`     | `plot_mic_array`, `plot_directional_map_from_grid`, `apply_matlab_like_style`, `figure_repro_context` |

## SH conventions

Channel ordering follows ACN (Ambisonic Channel Numbering):

```
index q = n(n + 1) + m,   n = 0..N,   m = -n..n
```

Complex SH use orthonormal normalisation. Real SH use the tesseral form. Round-trip `complex -> real -> complex` is lossless for conjugate-symmetric inputs.

`direct_sht` returns the weighted least-squares projection onto the chosen SH basis. On exact quadrature grids such as `gauss_legendre_sampling`, this coincides with the usual weighted projection formula. On approximate grids such as Fibonacci sampling, it is more accurate than treating `Y^H W` as an exact inverse.

## Running tests

For a full repository checkout, run the complete suite:

```bash
pytest tests/ -q
```

The source distribution published to PyPI contains a smaller test subset that
does not depend on repository-only MATLAB reference sources or example scripts.
Use the full repository checkout when validating MATLAB parity, image
regression, or migration tooling.

## Running the bundled examples (after `pip install`)

Since 0.4.0b15, a small set of self-contained, stable-API-only demos ships
inside the wheel as a regular sub-package at
``spherical_array_processing.examples``.  Wheel users can therefore run them
directly without cloning the repository:

```bash
python -m spherical_array_processing.examples.plane_wave_doa
python -m spherical_array_processing.examples.binaural_em32_to_ears
```

Programmatic use:

```python
from spherical_array_processing.examples.plane_wave_doa import run_example
out = run_example(az_deg=73.0, col_deg=62.0)
print(out["pwd_error_deg"], out["music_error_deg"])
```

The repo-side ``examples/`` tree is still the right place for longer,
research-flavoured scripts (Rafaely chapter figures, Politis comparisons) that
depend on repository-only reference assets and the ``spherical_array_processing.repro``
developer-only layer.  Those are intentionally **not** part of the runtime wheel.

## Development and release checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q --tb=short
python -m build
```

Before publishing, verify both the wheel and source distribution from a clean
environment.  The CI workflow runs the full repository suite on supported
Python versions and separately checks that the source distribution can run its
packaged test subset.

An optional external numeric audit is available for cross-checking core
mathematical contracts against independent open-source packages:

```bash
python scripts/external_numeric_audit.py --repo-root /tmp/sap-cross-repos
```

## MATLAB source provenance

Some repository-only reproduction tests compare Python behavior against
Rafaely and Politis MATLAB reference materials.  Those files are kept outside
the runtime package and are documented in `THIRD_PARTY_NOTICES.md`.  They are
reference and parity assets, not required for normal `pip install` usage.

## License

MIT. See [LICENSE](LICENSE).

## References

- Rafaely, B. (2015). *Fundamentals of Spherical Array Processing*. Springer.
- Zotter, F. & Frank, M. (2019). *Ambisonics*. Springer.
- Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. *IEEE Trans. Antennas Propag.*, 34(3), 276-280.
