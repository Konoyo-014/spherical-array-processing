# External review of b9

## Scope

I read `spherical_array_processing/ambi/uhj.py` and `spherical_array_processing/ambi/intensity.py` end to end, then read `tests/test_ambi_uhj.py` and `tests/test_ambi_intensity.py` end to end. I ran the focused suite with `pytest -q tests/test_ambi_uhj.py tests/test_ambi_intensity.py` and the full suite with `pytest -q`. The final post-fix state is **487 / 487 passing**.

This workspace does not include a `.git` directory, so I could not diff directly against the b8 tag. My regression judgment therefore comes from static review of the new b9 surface plus a full-suite run on the current tree.

## What I changed

I found one real correctness bug and one smaller coefficient typo in the new UHJ encoder. The conversion into FuMa was already right at `spherical_array_processing/ambi/uhj.py:82`: the code correctly converts ACN input to SN3D and then applies the required **`W_FuMa = W_SN3D / √2`** scaling at `spherical_array_processing/ambi/uhj.py:95`. That part did not need a fix.

The actual encode matrix at `spherical_array_processing/ambi/uhj.py:144` had two problems. First, the quadrature `X` coefficient was written as `0.5098022`, while the classical Gerzon UHJ-2 references give **`0.5098604`**. Second, the implementation emitted `L_T = S + D` and `R_T = S - D` instead of the classical **`L_T = (S + D)/2`** and **`R_T = (S - D)/2`**. That missing half-scale is the big issue: it makes the encoded stereo pair **6.02 dB too hot** and breaks the published Lt/Rt definition even though the decoder still “sort of works” on relative direction cues. I corrected both issues at `spherical_array_processing/ambi/uhj.py:148` and `spherical_array_processing/ambi/uhj.py:151`, and I updated the module docstring to state the classical equations explicitly at `spherical_array_processing/ambi/uhj.py:13`.

To keep that locked down, I added two regression tests. The first one at `tests/test_ambi_uhj.py:59` checks the pure-`Y` path, where the classical half-gain is algebraically exact and easy to verify. The second one at `tests/test_ambi_uhj.py:69` uses a bin-centered sinusoid in SN3D and checks the full published UHJ-2 encoding formula, including the **FuMa `W/√2`** conversion and the Hilbert-shifted term.

On the intensity side, the implementation itself was already correct. The sign convention in `doa_from_intensity` matched the package’s SH encoder, but the module-level prose in `spherical_array_processing/ambi/intensity.py:11` said the active vector pointed opposite to the source while the function docstring at `spherical_array_processing/ambi/intensity.py:149` correctly said it points toward the source. I fixed that documentation contradiction at `spherical_array_processing/ambi/intensity.py:11`.

I also tightened the standing-wave regression in `tests/test_ambi_intensity.py:54`. The original test used `100 Hz` in a `512`-sample FFT frame at `16 kHz`, which is not bin-centered and therefore leaks active-energy residue across bins. The phenomenon being tested was still real, but the setup was messier than it needed to be. I changed it to a bin-centered tone at `3 * fs / T = 93.75 Hz` in `tests/test_ambi_intensity.py:59`, so the active cancellation now collapses to numerical noise while the reactive term remains dominant by many orders of magnitude.

## Math review

The **UHJ-2 encode matrix** is now aligned with the classical references. I cross-checked the constants against the standard formulas reproduced by the Ambisonic UHJ pages on Wikipedia and Xiph, and against the implementation notes in OpenAL Soft’s `uhjfilter.cpp`. Those sources agree on `S = 0.9396926 W + 0.1855740 X`, `D = j(-0.3420201 W + 0.5098604 X) + 0.6554516 Y`, and `Left/Right = (S ± D)/2`. In other words, the code path at `spherical_array_processing/ambi/uhj.py:144` is now the classical UHJ-2 matrix, and the already-existing FuMa conversion at `spherical_array_processing/ambi/uhj.py:95` is exactly the right thing to do for SN3D input.

The **UHJ-2 decoder** at `spherical_array_processing/ambi/uhj.py:206` is also fine. The coefficients `0.982`, `0.164`, `0.419`, `0.828`, `0.763`, and `0.385` are the usual rounded Gerzon two-channel approximation, returning horizontal **E-format-like** recovery with `Z = 0`. I did not replace them because the published references match the current values. What matters here is that the module describes this as an approximate FOA recovery, which is the correct contract for UHJ-2.

The **Hilbert transform sign convention** is correct as implemented. `scipy.signal.hilbert(x).imag` returns the standard quadrature signal `H{x}`, and in the context of the published UHJ equations it gives the right wideband `j` operator. The new regression at `tests/test_ambi_uhj.py:69` is the practical proof: for a bin-centered cosine carrier, the implementation reproduces the published Lt/Rt formula within numerical tolerance. If the sign had been flipped, that test would fail immediately because the stereo image would rotate the wrong way.

The **intensity DOA sign** is also correct for this package. The encoder defines a plane wave as `c_q(t) = Y_q(û) s(t)` in `spherical_array_processing/ambi/encoder.py:1`, so under the real first-order orthonormal basis the coefficient triplet `(X, Y, Z)` is proportional to the **source direction vector `û` itself**, not to the propagation vector `-û`. That makes `Re{W^* (X, Y, Z)}` proportional to `|s|^2 û`, so the DOA estimator in `spherical_array_processing/ambi/intensity.py:156` should indeed be **`+I / ||I||`**. My independent numerical check on a `45°` plane wave gave a mean intensity proportional to `[+x, +y, 0]`, not the negative of that vector.

The **reactive-intensity standing-wave test** is physically legitimate. Two equal-amplitude counter-propagating monochromatic plane waves with a relative phase `φ` still form a standing wave; the phase just shifts the node pattern in space. Algebraically, `e^{-jkx} + e^{j(kx+φ)} = 2 e^{jφ/2} cos(kx + φ/2)`. In FOA terms, the equal powers from `+x` and `-x` cancel the active part while the cross-term leaves a reactive component proportional to `sin φ`. With the cleaned-up bin-centered test at `tests/test_ambi_intensity.py:59`, the active energy drops to about `2.7e-24` while the reactive energy stays around `3.26e8`, so this is plainly a standing-wave signature rather than an FFT artefact.

The **normalization-invariance checks** are sound. For UHJ, `_to_fuma` in `spherical_array_processing/ambi/uhj.py:82` first canonicalizes to SN3D and only then applies the FuMa `W/√2` rule, so correctly declared orthonormal, N3D, and SN3D inputs land on the same FuMa triplet before the Gerzon matrix. For intensity, `spherical_array_processing/ambi/intensity.py:103` canonicalizes everything to orthonormal before forming `W^* (X, Y, Z)`. My independent numerical checks gave a max difference of `0.0` for UHJ orthonormal-vs-SN3D output and about `4.2e-17` for intensity, which is exactly the floating-point behavior I would expect.

## Flagged items

The remaining caveat is mostly about **API expectations**, not correctness. The UHJ decoder in `spherical_array_processing/ambi/uhj.py:157` is the classical two-channel approximation, so an `encode → decode` round trip is **not** a transparent recovery of the original FOA. It is intrinsically an approximate horizontal reconstruction, with `Z` discarded and the recovered `W/X/Y` corresponding to Gerzon’s two-channel decode, often described as **E-format** rather than exact original B-format. The current docstring already says “approximate FOA” and “`Z = 0`”, which is good, but downstream users should not interpret this API as lossless.

There is also one **ergonomic caveat** in both `uhj_encode` and `uhj_decode`: the wideband quadrature operator is implemented with `scipy.signal.hilbert`, which is FFT-based and therefore has whole-block semantics. That is perfectly reasonable for offline waveform conversion, but it will show edge behavior if someone applies it to short independent blocks in a streaming context. I do not think this blocks b9, but it is worth documenting if these functions are expected to be used in chunked real-time pipelines.

## Regressions

I do not see a regression against the pre-b9 surface. The new functionality is self-contained in the added ambisonics modules, and after the fixes the full suite passes at **487 / 487**. The one concrete regression risk I did find was the UHJ encoder gain/coefficient issue described above, and that is now patched and covered by explicit regression tests.

## Verdict

My verdict is **TAG** for b9 **after** the UHJ encoder correction in `spherical_array_processing/ambi/uhj.py:144` and the associated regression tests in `tests/test_ambi_uhj.py:59`. Without that fix I would have called b9 **NEEDS-WORK**, because the published UHJ-2 Lt/Rt matrix was not being implemented faithfully.
