# Codex Review — 0.4.0b14 Final Aggregate Sign-off

## Verdict

**TAG.** I re-ran the full release validation for `0.4.0b14` and the aggregate state is internally consistent. The shipped A1, A2, and A3 changes still hold together under cross-check, and I did not find a silent regression introduced by the final A3 boundary cleanup.

## Validation

I ran `PYTHONPATH=. python3 -m pytest -q` at the package root. The suite completed at **557 passed** with no test failures, which matches the expected post-b14 count. The warning output is consistent with the intended contract: direct imports of the developer-only `repro`, `regression`, and `experimental` surfaces emit `FutureWarning`, while stable public imports continue to pass.

I also verified the rebuilt source distribution with `tar tzf dist/spherical_array_processing-0.4.0b14.tar.gz | grep -E "(ambi/intensity|dirac/analysis|hrtf/sofa|encoding/measured_filters|_measured_sht_filters|repro/__init__|regression/__init__|experimental/__init__)"`. All expected files are present in the sdist, including the new internal `_measured_sht_filters` module and the three warning-emitting developer-only `__init__` files.

## A1 / A2 / A3 Cross-check

For **A1**, the shared **`_canonical_foa_pv`** path is wired exactly where it needs to be: `ambi.intensity_vector(..., physical_units=True)` and `dirac.dirac_analysis` both canonicalise FOA inputs through the same pressure/velocity conversion. The backward-compatibility branch remains intact because the default `physical_units=False` path still forms the historical coefficient-space intensity directly. The full suite includes the explicit backward-compat lock and the pressure/velocity equivalence checks, so there is no sign that the later A3 work disturbed this behavior.

For **A2**, the **`preserve_zero_delay`** implementation matches the intended semantics. The loader still folds all-zero `Data.Delay` blocks to `None` by default, and with `preserve_zero_delay=True` it preserves explicit zeros in memory. The important shape rule is present in code and tests: file-side **`(1, 2)` normalises to dataset-side `(2,)`**, while **`(M, 2)` remains `(M, 2)`**. The byte-level round-trip test remains green, so the representation-fidelity guarantee is intact.

For **A3**, the boundary leak fix is sound. Stable `encoding.measured_filters` now imports the shared implementation from **`spherical_array_processing._measured_sht_filters`** instead of reaching into `repro.politis.functions`. At the same time, the repro layer now delegates to that same internal implementation, so the numerical path is shared rather than duplicated. That matters because it means the A3 refactor changed the import boundary without changing the algorithm. This is exactly what the measured-equalizer tests want: `measured_array_equalizer` and `apply_measured_equalizer` still reconstruct the expected SH patterns, keep axis handling correct, and preserve exact zero output for silent input. The dedicated import-contract test also confirms that importing stable `sap.encoding` emits **no** developer-only `FutureWarning`.

## Changelog accuracy

The release notes were almost correct as written, but they did not explicitly mention the **file-side `(1, 2)` to memory-side `(2,)`** normalisation for `Data.Delay`. I updated `CHANGELOG.md` so the `0.4.0b14` entry now describes the shipped behavior precisely: default all-zero folding to `None`, `preserve_zero_delay=True` preserving explicit zeros in memory, `(1, 2)` normalising to `(2,)`, `(M, 2)` staying `(M, 2)`, and `save_sofa` preserving byte-level round-trip behavior on disk.

## Final sign-off

My aggregate review outcome is **TAG**. The package state, tests, sdist contents, warning boundaries, and release notes now line up with what `0.4.0b14` actually ships.
