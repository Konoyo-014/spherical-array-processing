# Cross-Reference Numeric Audit

This document records the external numeric checks used to validate
`spherical-array-processing` against independently maintained open-source
packages. The external repositories are used only as audit inputs. They are not
runtime dependencies and are not vendored into this repository.

## Reference Projects

The current audit downloads or installs these projects:

| Project | Role | Source |
| --- | --- | --- |
| `spaudiopy` | Spatial audio SH basis, modal beamforming weights, spherical mode strength | https://github.com/chris-hld/spaudiopy |
| `sound_field_analysis-py` | Spherical microphone array analysis, SH basis, radial functions, modal coefficients | https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py |
| `spharpy` | Spherical harmonics, sampling, beamforming, modal strength | https://github.com/pyfar/spharpy |
| `sphericart` | Fast real spherical harmonics in Cartesian coordinates | https://github.com/lab-cosmo/sphericart |
| `pyroomacoustics` | General room-acoustics and array-processing reference for future DOA/beamforming audits | https://github.com/LCAV/pyroomacoustics |

`spherical` was also considered as a spherical-harmonics reference, but its
`spinsfast` dependency needs `fftw3` at build time on this machine. It is not
used in the automated audit until a reliable wheel or system FFTW dependency is
available.

## Reproduction Command

The audit script expects the optional reference packages to be installed in the
active environment. If external git checkouts are available, pass their root
directory so the report records commit hashes.

```bash
python scripts/external_numeric_audit.py \
  --repo-root /tmp/sap-cross-repos \
  --json /tmp/sap-cross-audit.json
```

## Current Result

The latest local run used deterministic seed `20260414`, SH order `4`, and 37
random directions for SH basis checks. External repository heads were
`spaudiopy@8995ca8`, `sound_field_analysis-py@4b03ee1`, `spharpy@883b341`,
`sphericart@1c36b59`, and `pyroomacoustics@c7365b8`.

| Check | Status | Max absolute error | Interpretation |
| --- | --- | ---: | --- |
| `scipy.complex_spherical_harmonics` | pass | `0.000e+00` | Confirms SciPy-compatible complex orthonormal SH basis in ACN order. |
| `spaudiopy.complex_sh_matrix` | pass | `0.000e+00` | Confirms complex SH basis and angle convention. |
| `spaudiopy.real_sh_matrix` | pass | `0.000e+00` | Confirms real SH basis convention. |
| `spaudiopy.hypercardioid_weights` | pass | `0.000e+00` | Confirms max-DI hypercardioid weights. |
| `spaudiopy.cardioid_weights` | pass | `3.997e-15` | Confirms cardioid/in-phase modal weights. |
| `spaudiopy.butterworth_modal_weights` | pass | `2.220e-16` | Confirms spatial Butterworth modal taper weights. |
| `spaudiopy.open_rigid_mode_strength` | pass | `0.000e+00` | Confirms open and rigid sphere mode-strength formulas. |
| `sound_field_analysis.complex_sh_all` | pass | `0.000e+00` | Confirms complex SH basis. |
| `sound_field_analysis.real_sh_convention_delta` | info | `1.409e+00` | Real SH definition differs; complex basis is the comparable contract. |
| `sound_field_analysis.radial_special_functions` | pass | `4.677e-09` | SFA uses recurrence formulas for some derivatives; tolerance is set to `1e-8`. |
| `sound_field_analysis.normalized_modal_coefficients` | pass | `2.326e-16` | Confirms normalized open and rigid modal coefficients. |
| `spharpy.complex_harmonic_basis` | pass | `7.473e-16` | Confirms complex SH basis. |
| `spharpy.real_harmonic_basis` | pass | `9.714e-16` | Confirms real SH basis. |
| `spharpy.supercardioid_front_back_ratio` | pass | `1.072e-06` | Confirms supercardioid max front-back ratio weights. |
| `spharpy.maxre_weight_family_delta` | info | `1.478e-02` | Both are max-rE-style tapers, but the published parameterizations differ slightly. |
| `spharpy.dolph_chebyshev_weights` | pass | `1.038e-13` | Confirms spherical Dolph-Chebyshev modal weights. |
| `spharpy.open_modal_strength` | pass | `0.000e+00` | Confirms open sphere modal strength. |
| `spharpy.rigid_modal_strength_global_sign_adjusted` | pass | `5.385e-15` | Rigid sphere formula matches after a global sign-convention adjustment. |
| `sphericart.real_spherical_harmonics` | pass | `4.802e-15` | Confirms real SH values from Cartesian unit vectors. |

## Finding Fixed From This Audit

The audit exposed a real issue in `beam_weights_supercardioid`. The public API
claimed a maximum front-back ratio design, but orders 2 through 4 were served
from old tabulated coefficients that did not match the generalized eigenvalue
solution used by `spharpy`. Orders 5 and above already used the numerical
solution and matched `spharpy`, which isolated the problem to the low-order
table. The main API now solves the max front-back ratio design uniformly for
all orders. The legacy Politis/MATLAB tables remain available in the
repository-only reproduction layer where MATLAB parity, rather than Pythonic
API semantics, is the intended contract.

## Function Gaps Closed From Reference Projects

The reference packages also exposed several core API gaps that were in scope
for this package. `spharpy` provides direct ACN-to-degree/order helpers, so the
package now exposes `sap.sh.acn_to_nm` and `sap.sh.degree_order_pairs`.
`sound_field_analysis` exposes spherical Neumann and second-kind Hankel radial
wrappers, so `sap.acoustics` now includes `besselys`, `besselysd`,
`besselhs2`, and `besselhs2d`, along with `wavenumber` and `kr` convenience
helpers. `spaudiopy` and `spharpy` include additional fixed modal beamforming
tapers, so `sap.beamforming` now includes `beam_weights_butterworth`,
`beam_weights_dolph_chebyshev`, and `normalize_axisymmetric_weights`.

Large systems implemented by `pyroomacoustics`, such as full room simulation,
image-source modeling, and full-featured broadband DOA classes, remain outside
the current package boundary. They should be treated as future roadmap items
rather than copied into a spherical-array core library without a matching
architecture and test-data layer.
