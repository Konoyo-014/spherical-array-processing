# External review of b8

## Scope

I read `spherical_array_processing/room/metrics.py`, `spherical_array_processing/ambi/encoder.py`, and `spherical_array_processing/room/banded.py` end to end, then ran `PYTHONPATH=. python3 -m pytest tests/test_room_metrics.py tests/test_ambi_encoder.py tests/test_room_banded.py -q` and a full `PYTHONPATH=. python3 -m pytest -q`. The final post-fix state is **459 / 459 passing**.

## What I changed

I found one real correctness bug in the new banded-RIR path and patched it in `spherical_array_processing/room/banded.py:64`. The original `_fir_from_bands` construction did **not** encode piecewise-constant band gains. Because each interior band edge appeared only once, `scipy.signal.firwin2` linearly interpolated from one band gain to the next across the whole following band. In a simple two-band target `[1, 0]`, that means the upper half of the spectrum becomes an average ramp instead of a stopband, with an average magnitude around `0.5` instead of near `0`.

The fix duplicates every **interior** band edge with the gain on both sides of the discontinuity, which is the standard `firwin2` way to realize a stepped magnitude response. That change is in `spherical_array_processing/room/banded.py:72`. I also tightened the docstring at `spherical_array_processing/room/banded.py:152` so the API now states the real onset-timing tradeoff of the centered linear-phase FIR.

To lock that bug down, I added a regression test at `tests/test_room_banded.py:18`. It designs a two-band `[1, 0]` target and verifies that the realized stopband stays low instead of ramping across the upper band.

## Math review

In `spherical_array_processing/room/metrics.py:46`, the **Schroeder-integrated EDC** is implemented with the correct reverse cumulative energy integral, namely `10 log10(sum_{m>=n} h[m]^2 / sum_m h[m]^2)`. The regression anchors in `spherical_array_processing/room/metrics.py:109` are also correct for **T20**, **T30**, and **T60**. The code fits between `[-5, -25]`, `[-5, -35]`, and `[-5, -65]` dB respectively, then extrapolates the fitted slope to 60 dB, which is the ISO 3382 convention. The failure mode is clean as well: if the EDC does not reach the required lower anchor, `_rt_from_edc` raises a clear `ValueError` from `spherical_array_processing/room/metrics.py:86`, and `rir_metrics()` propagates that cleanly through `spherical_array_processing/room/metrics.py:231`.

The **plane-wave encoder** is mathematically consistent. At `spherical_array_processing/ambi/encoder.py:82`, `sh_matrix(...)` returns shape `(K, Q)`, which matches the contract documented in `spherical_array_processing/sh/basis.py:213`. The contraction `y.T @ sig` at `spherical_array_processing/ambi/encoder.py:84` therefore computes `c_q(t) = Σ_k Y_q(d_k) s_k(t)` exactly, with the expected linear ambisonic summation semantics when multiple monaural sources are passed in together. I do not see a regression here relative to b7.

The **banded image-source model** is structurally sound after the FIR fix. Grouping images by identical six-wall bounce vectors is mathematically valid because the per-band gain factor is `∏_w β_w(f)^{n_w}`, so two images with the same bounce-count vector do share the same reflection filter. The geometric spreading term remains separated as `1 / r`, mirroring the scalar model in `spherical_array_processing/room/shoebox.py:165`. I also spot-checked the scalar reduction case: when all wall gains are frequency-flat, `shoebox_rir_banded` collapses numerically to the same impulse train as `shoebox_rir`, which is exactly what you want.

## Flagged items

There is still one **modeling caveat** in `spherical_array_processing/room/banded.py:155`. The synthesized FIR is linear phase and is scattered **centered** on the geometric image delay. That preserves the nominal reflection time at the FIR midpoint, but it also introduces noncausal pre-ringing of up to `fir_taps // 2` samples before each reflection. With the default `129` taps at `16 kHz`, that is `64` samples or about `4 ms`. On a representative high-frequency-loss profile like `[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.05]`, roughly **4.2%** of the FIR energy lies before the nominal arrival, and the largest pre-lobe is about **15%** of the center coefficient. I do not think this invalidates the feature for late-reverberation coloration, but it **can smear early reflections audibly** when onset timing matters. I therefore consider this acceptable for b8 only if it is understood as a deliberate approximation rather than a physically causal wall-filter model.

There is also a smaller API ergonomics mismatch around `band_edges_hz`. The docstring at `spherical_array_processing/room/banded.py:139` says the edges start at `0`, while validation in `spherical_array_processing/room/banded.py:55` currently accepts any first edge `>= 0`. That is not a mathematical bug because `_fir_from_bands` extends the first band gain down to DC anyway, but the documented contract and the validator are not perfectly aligned.

## Regressions

I do not see a regression against the b7 surface. The new metrics code is self-contained and does not perturb old room paths. The encoder is additive and follows existing SH conventions. The banded RIR path had the FIR-shape bug described above, but after the patch it integrates cleanly and the full suite still passes.

## Verdict

My verdict is **TAG** for b8 **after** the `_fir_from_bands` correction in `spherical_array_processing/room/banded.py:64`. Without that fix I would have called it **NEEDS-WORK**, because the realized reflection spectrum was not actually piecewise-constant by band.
