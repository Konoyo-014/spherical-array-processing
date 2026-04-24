# Codex Review b11 Round 3

I re-checked the `translate_foa` entry in `CHANGELOG.md` and the module documentation in `spherical_array_processing/ambi/translation.py`.

The `Added` entry no longer claims the method is mathematically exact. It now states that the implementation reproduces the leading-order geometric advance `û · r / c` only in the small-translation regime `|k r| ≪ 1`, and it explicitly acknowledges two limitations: angular smearing from the first-order plane-wave decomposition and periodic-extension artefacts from the FFT-based fractional-delay step.

That wording is consistent with the module-level documentation, which already describes `translate_foa` as an approximation away from the `r → 0` limit and says the leading-order advance is reproduced only for `|k r| ≪ 1`, with larger translations becoming increasingly smoothed and frequency-dependent. The extra sentence about FFT periodic-extension artefacts is also consistent with the implementation note in the same module, which says the zero-pad reduces but does not eliminate those artefacts.

Final verdict: **TAG**.
