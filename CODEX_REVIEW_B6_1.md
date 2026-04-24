# b6 External Review

## Scope and outcome

I read `spherical_array_processing/ambi/format.py`, `spherical_array_processing/binaural/pipeline.py`, and `spherical_array_processing/hrtf/rigid_sphere.py` end-to-end, then ran the requested focused test command and the full regression suite. During review I found one **real mathematical bug** in the new rigid-sphere HRTF path, plus one small cleanup in the binaural pipeline. I fixed both directly and added regression tests that lock the corrected behaviour down. With those fixes in place, my final verdict is **TAG**.

## What I changed

The main fix is in `spherical_array_processing/hrtf/rigid_sphere.py:132`. The original implementation multiplied the SH addition-theorem sum by an extra **`4π`** even though `bn_matrix(..., sphere="rigid")` already returns modal coefficients carrying the **`4π i^n`** factor. Under this package's default **orthonormal** SH convention, the correct identity is `H = Σ_q b_q Y_q(r_ear) Y_q*(r_src)`, not `4π · Σ_q ...`. I therefore removed the global `4π` factor at `spherical_array_processing/hrtf/rigid_sphere.py:140`, corrected the module-level formula at `spherical_array_processing/hrtf/rigid_sphere.py:6`, and fixed the DC limit at `spherical_array_processing/hrtf/rigid_sphere.py:138` to **`b_0(0) = 4π`** rather than `1`. That DC value is the one that makes the orthonormal SH sum collapse to unit gain at zero frequency.

While fixing that path, I also made the one-sided spectrum explicitly compatible with a real `irfft` representation at `spherical_array_processing/hrtf/rigid_sphere.py:144`. For an even-length real HRIR, the **DC** and **Nyquist** bins must be purely real. `irfft` was already enforcing that implicitly by discarding any unsupported imaginary Nyquist part; the code now makes that projection explicit before transforming back to time domain.

In `spherical_array_processing/binaural/pipeline.py:184` I removed the suspicious dtype branch and now pass `sig_cf` directly into `oaconvolve`. The old code used `sig_cf.real if sig_cf.dtype.kind != "c" else sig_cf`, which was functionally harmless for ordinary real arrays but logically unnecessary and confusing during review. The current form is cleaner and handles both real and complex SH signals without hidden branching.

To keep these fixes from regressing, I added three focused tests. `tests/test_rigid_sphere_hrtf.py:87` now checks that the analytic rigid-sphere HRTF, after undoing the intentional time-domain `fftshift`, matches `simulate_sh_array_response` bin-for-bin. `tests/test_ambi_format.py:146` now drives the FuMa round-trip with deterministic **all-channels-active** order-2 and order-3 fields, and `tests/test_ambi_format.py:166` adds the specific **T/V/M/Q** spot-checks you asked for, including the expected FuMa weights. `tests/test_ambi_to_binaural.py:123` now checks that the one-call renderer gives the same binaural result for a physically identical field expressed in the **real** and **complex** SH bases, which covers the complex-signal code path in `ambi_to_binaural_time_domain`.

## What I checked and did not change

`_scaling_vector` in `spherical_array_processing/ambi/format.py` is mathematically correct as written. The package's SH basis uses `orthonormal` by default, with `Y_n^{m,\mathrm{N3D}} = \sqrt{4\pi} Y_n^{m,\mathrm{ortho}}` and `Y_n^{m,\mathrm{SN3D}} = \sqrt{4\pi/(2n+1)} Y_n^{m,\mathrm{ortho}}`, so the coefficient scaling must be the inverse of those basis scalings. The implementation at `spherical_array_processing/ambi/format.py:74` through `spherical_array_processing/ambi/format.py:90` matches that exactly, and the ACN per-order lookup `n(q) = floor(sqrt(q))` is correct.

`_FUMA_WEIGHTS_FROM_SN3D` in `spherical_array_processing/ambi/format.py:60` is also correct. The **W** channel uses the expected `1/√2`, the first-order channels are unity, the second-order **S/T/U/V** channels use `2/√3`, and the third-order weights `√(45/32)`, `3/√5`, and `√(8/5)` match the standard FuMa-to-AmbiX conversion table derived from the Malham / Nachbar conventions. I did not change these constants.

`_FUMA_TO_ACN` in `spherical_array_processing/ambi/format.py:33` is correct as well. The spot checks you called out all line up with the ACN formula `q = n(n+1)+m`: **X ↔ 3**, **Y ↔ 1**, **T ↔ 5**, **V ↔ 4**, **M ↔ 11**, and **Q ↔ 9**. I verified these against the code table and then locked several of them in with new tests.

`ambi_to_binaural_time_domain` is mathematically correct in its convolution logic. After the MagLS filter is converted to a centred FIR, the output really is `y_e[t] = Σ_q (x_q * h_{q,e})[t]`, and the `oaconvolve(..., axes=-1)` plus `sum(axis=0)` implementation at `spherical_array_processing/binaural/pipeline.py:181` through `spherical_array_processing/binaural/pipeline.py:190` computes exactly that sum over SH channels. The `fftshift` centring is also internally consistent: it produces a **zero-phase centred FIR** and therefore a full convolution whose physically aligned `T` samples are obtained by slicing from `fft_len // 2`, exactly as the docstring says.

## Flags that remain but are not blockers

The one API point I would still call out is `spherical_array_processing/binaural/pipeline.py:49`. `_detect_sh_axis` necessarily becomes ambiguous when both dimensions happen to equal `(N+1)^2`, and in that case it silently chooses **channels-first**. That behaviour is defensible, and I did not change it, but it is still worth documenting because it can surprise a caller passing a square `(Q, Q)` buffer.

I also think the FuMa API could be more future-proof than the current boolean flags `from_sn3d` and `to_sn3d` in `spherical_array_processing/ambi/format.py:149` and `spherical_array_processing/ambi/format.py:213`. They are acceptable for the current scope because the only supported non-FuMa normalisations here are **SN3D** and the package's internal **orthonormal** basis, but an explicit normalization enum would scale better if the API grows. I did not change this because it is an ergonomics question, not a correctness bug.

## Validation

The requested focused command `PYTHONPATH=. python3 -m pytest tests/test_ambi_to_binaural.py tests/test_ambi_format.py tests/test_rigid_sphere_hrtf.py -q` passes after the fixes, with **29 passing tests**.

I also ran the full suite with `PYTHONPATH=. python3 -m pytest -q`. That now completes successfully with **392 passed** and one unrelated plotting-layout warning.

## Verdict

**TAG**.

Before the review fixes, `rigid_sphere_hrtf` had a genuine amplitude error from a duplicated **`4π`** factor in the modal series. After correcting that, making the DC/Nyquist handling explicit, and tightening the tests around FuMa mapping and complex-basis binaural rendering, the b6 additions look mathematically sound and releaseable.
