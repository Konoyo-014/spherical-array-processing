# CODEX_REVIEW_B6_2

## Pytest

`PYTHONPATH=. python3 -m pytest -q` completed with the expected result:

`392 passed, 1 warning in 4.91s`

The single warning comes from `scripts/run_rafaely_ch1_pn.py` during `tests/test_rafaely_example_scripts.py::test_run_rafaely_ch1_pn_script`, where Matplotlib reports that `tight_layout()` could not fully satisfy the figure margins. This does not affect the b6 codepaths under review.

## Sdist

`tar tzf dist/spherical_array_processing-0.4.0b6.tar.gz | grep -E "(ambi|rigid_sphere|pipeline)"` shows the expected new packaged files for b6. In particular, the three newly added tests are present in the source distribution:

`spherical_array_processing-0.4.0b6/tests/test_ambi_format.py`

`spherical_array_processing-0.4.0b6/tests/test_ambi_to_binaural.py`

`spherical_array_processing-0.4.0b6/tests/test_rigid_sphere_hrtf.py`

The corresponding shipped modules are also present, including `spherical_array_processing/ambi/format.py`, `spherical_array_processing/binaural/pipeline.py`, and `spherical_array_processing/hrtf/rigid_sphere.py`.

## Code Review

I re-read `spherical_array_processing/ambi/format.py`, `spherical_array_processing/binaural/pipeline.py`, and `spherical_array_processing/hrtf/rigid_sphere.py` end-to-end.

The **`4π` fix is still intact** in `spherical_array_processing/hrtf/rigid_sphere.py:129`. The implementation builds `bn` directly from `bn_matrix(..., sphere="rigid")`, which already includes the **`4π·iⁿ`** modal prefactor documented in `spherical_array_processing/acoustics/radial.py:186`. The HRTF assembly therefore uses the correct orthonormal-basis sum `H = Σ_q b_q · Y_q(ear) · Y_q*(src)` without reintroducing an extra `4π` factor. The DC handling remains correct as well: `np.nan_to_num(...)` zeroes invalid higher-order DC entries and `bn[0, 0] = 4.0 * np.pi` restores the exact omnidirectional limit.

The `binaural` pipeline dtype cleanup is also still present in `spherical_array_processing/binaural/pipeline.py:131` and `spherical_array_processing/binaural/pipeline.py:184`. The function now forwards the SH signal directly into `oaconvolve` without the older real/complex branch, and it returns a clean `float64` stereo result.

The new `sap.ambi` code is coherent with the shipped API. `spherical_array_processing/ambi/__init__.py:27` exports `convert_ambi_normalization`, `acn_to_fuma`, and `fuma_to_acn`, and the top-level lazy loader exposes `sap.ambi` through `spherical_array_processing/__init__.py:89`.

## Docs Spot-Check

The README updates accurately describe the shipped code. The feature summary mentions the new one-call renderer `binaural.ambi_to_binaural_time_domain`, the analytic generator `hrtf.rigid_sphere_hrtf`, and the new `ambi` format-conversion module, and the module reference table lists the same exported symbols that are actually present in the package.

The CHANGELOG `Fixed (codex b6 review)` section is materially accurate. It correctly describes the **`4π`** amplitude correction, the `binaural` dtype-path cleanup, and the new FuMa / complex-basis regression tests.

I made one small inline fix before sign-off: the b6 header still claimed `Full test suite: 388 / 388`, but the current shipped state is `392 / 392`. That line is now updated in `CHANGELOG.md:7`.

## Findings

No new code defects turned up in this second pass. Beyond the factual test-count drift in the changelog header, which is now fixed, I found **nothing new worth blocking the release**.

## Verdict

**TAG**
