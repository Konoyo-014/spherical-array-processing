# Changelog

All notable changes to `spherical-array-processing` are documented here.

## [0.4.0] - 2026-05-10

### Changed

- Consolidated the public project line around `spherical-array-processing` as
  the canonical package and repository, absorbing the open-source governance,
  getting-started documentation, and tutorial material that had previously lived
  in the separate `spharray` line.
- Promoted the 0.4.0 beta surface to a stable 0.4.0 release version.

### Added

- Added `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, and a `docs/`
  starter documentation set.
- Added beginner core/tutorial examples from the `spharray` line, adapted to
  import `spherical_array_processing`.
- Added small compatibility conveniences from the `spharray` line:
  `coords.angular_distance`, `coords.angular_distance_deg`,
  `array.spatial_aliasing_frequency`, `array.max_sh_order`,
  `acoustics.equalize_modal_coeffs`, `beamforming.steer_sh_weights`,
  `beamforming.beamform_sh`, and DOA covariance helpers.

## [0.4.0b15] - 2026-04-24

Fifteenth beta — closes the remaining API-symmetry, packaging-boundary,
and example-installation follow-up items from the previous beta cycle.
The release was validated with the full test suite: **592 / 592** (with
a built wheel in `dist/`).

### Changed

- **Wheel boundary is now dev-layer-free.** `pyproject.toml` adds
  `[tool.setuptools] include-package-data = false` and switches
  `[tool.setuptools.packages.find]` to an explicit
  ``include = ["spherical_array_processing*"]`` plus an
  ``exclude = ["...repro*", "...regression*", "...experimental*"]``
  list.  The built wheel now contains 0 files from the developer-only
  layers (down from 28) while the sdist keeps them for reproduction
  and regression workflows.  The invariant is locked by two new
  contract tests in `tests/test_import_contract.py`.
- **Plotting tests gate cleanly on matplotlib.** Both
  `tests/test_plotting_helpers.py` and
  `tests/test_rafaely_plot_wrappers.py` now use
  `pytest.importorskip("matplotlib")` at module level, so a clean
  venv with only the ``.[dev]`` extras skips them instead of crashing
  collection.
- **`directional` first-order arrays are first-class.**
  `ArrayGeometry.array_type`, `encoding.radial_equalizer_*`,
  `array.simulate_sh_array_response`, and `acoustics.bn_matrix` all
  accept ``"directional"`` with a ``dir_coeff`` ∈ [0, 1].  The α=0.5
  and α=1.0 limits are **bitwise-identical** to the ``cardioid`` and
  ``open`` paths respectively.  The equalizer family and the
  simulation layer have matching three-case validation (missing /
  stray / out-of-range `dir_coeff`) with identical error messages.
  New `tests/test_directional_array_plumbing.py` locks the full
  contract (13 tests).
- **FOA intensity is now single-track.**
  `diffuseness.intensity_vectors_from_foa` is a thin wrapper over
  `ambi.intensity_vector`; it gains ``normalization`` and
  ``physical_units`` keyword arguments that forward to the canonical
  implementation.  Default behaviour is byte-level identical to prior
  releases (including the historical ``match="4 channels"`` error
  string and the FuMa layout path).  `tests/test_diffuseness_intensity_wrapper.py`
  adds 7 equivalence + contract tests.

### Added

- **`spherical_array_processing.examples`** sub-package shipping
  inside the wheel.  Two install-safe demos:
  ``plane_wave_doa`` (PWD + MUSIC DOA recovery) and
  ``binaural_em32_to_ears`` (Eigenmike → MagLS binaural render).  Both
  are runnable as ``python -m spherical_array_processing.examples.<name>``
  after ``pip install`` — no repo checkout required.  The repo-side
  ``examples/binaural_em32_to_ears.py`` is now a compatibility shim
  that re-exports from the sub-package.  Verified in an isolated
  venv from outside the repo root.
- **README**: new "Running the bundled examples (after `pip install`)"
  section documenting the ``python -m`` invocation and the
  programmatic ``run_example(...)`` entry point.
- **Acoustics-layer `dir_coeff` validation is now symmetric.**
  `plane_wave_radial_bn`, `bn_matrix`, and `sph_modal_coeffs` share a
  new private `_validate_sphere_and_dir_coeff` helper and reject
  stray `dir_coeff` on non-directional sphere kinds, missing
  `dir_coeff` on the directional branch, and out-of-range α at the
  same layer as the equalizer and simulation entry points.
  `sph_modal_coeffs` reports errors in its `array_type` vocabulary
  rather than leaking the internal integer ``sphere`` code.  Four
  new tests in `tests/test_directional_array_plumbing.py`.
- **`ArrayGeometry` is now the single source of truth for the array
  spec in `simulate_sh_array_response`.**  The function's
  `array_type` parameter defaults to `None` and falls back to
  `geometry.array_type`; when the resolved type is `"directional"`
  and `dir_coeff` is omitted, it is read from
  `geometry.metadata["dir_coeff"]`.  Explicit kwargs always win, so
  existing callers are unaffected — a geometry with its default
  `"rigid"` array type and a caller that also omits the argument
  observes exactly the pre-b15 output.  Four new tests lock the
  precedence contract in `tests/test_directional_array_plumbing.py`.
- **`dir_coeff` validation is now single-sourced.**  The three-case
  contract (missing / stray / out-of-range) lives in one private
  helper `_validate_sphere_and_dir_coeff` in
  `spherical_array_processing/acoustics/radial.py`; the equalizer
  (`_validate_dir_coeff`) and simulation layers are thin façades
  over it.  The helper takes an `arg_name` kwarg so each call site
  gets error messages in its own vocabulary (`sphere='directional'`
  for the raw acoustics API, `array_type='directional'` elsewhere),
  and the duplicate in-branch checks inside `plane_wave_radial_bn`
  were dropped in favour of the upfront contract call.
- **`ArrayGeometry.sensor_kind` now participates in validation.**
  The field changed from a free-form informational tag to a
  constraint the dataclass actively enforces.  It defaults to `None`
  and is auto-derived from `array_type` in `__post_init__`
  (`"pressure"` for open/rigid, `"directional"` for
  cardioid/directional).  An explicit value that disagrees with
  what `array_type` implies raises `ValueError` immediately, so
  e.g. `ArrayGeometry(array_type="rigid", sensor_kind="directional")`
  can no longer slip through.  Six new tests in
  `tests/test_directional_array_plumbing.py`.

Fourteenth beta — third-tier polish around API boundaries and
representation fidelity.  Three focused changes address residual
release-readiness items from the earlier beta cycle.  Full test suite:
**557 / 557**.

### Changed

- **`ambi.intensity_vector`** gained a ``physical_units=False``
  keyword.  Default is backward compatible (coefficient-space
  intensity ``Re{W^* · (X, Y, Z)}``).  Setting ``True`` yields the
  textbook pressure-velocity intensity, matching what
  :func:`dirac.dirac_analysis` uses internally — both entry points
  now share one private helper (``_canonical_foa_pv``), so they
  cannot drift apart.
- **`hrtf.load_sofa`** gained a ``preserve_zero_delay=False`` keyword.
  Default folds an all-zero ``Data.Delay`` block to
  ``data_delay_samples=None`` as before.  Setting ``True`` keeps the
  explicit zero block in memory; SOFA's file-side per-ear shape
  ``(1, 2)`` is normalised to dataset-side ``(2,)`` while ``(M, 2)``
  stays ``(M, 2)``, and downstream ``save_sofa`` still round-trips the
  file byte-for-byte.
- **`encoding.measured_array_equalizer`** now depends on a private
  ``_measured_sht_filters`` module instead of reaching into the
  developer-only ``repro`` subpackage.  Normal users of the stable
  ``encoding`` API are no longer hit by the developer-only
  ``FutureWarning`` introduced below.

### Deprecated

- **``spherical_array_processing.repro`` / ``regression`` /
  ``experimental``** now emit a one-time ``FutureWarning`` on first
  import, calling out that they are developer-only reproduction /
  tooling / research surfaces and not part of the stable public
  API.  Each warning tells the user which stable module to prefer
  and recommends pinning an exact version if they genuinely need
  the unstable surface.  The top-level ``sap`` namespace continues
  to hide these submodules (``sap.repro`` etc. raise
  ``AttributeError``), so the cleaner signal only shows up when
  users explicitly reach for them.  The new
  ``test_developer_only_submodules_emit_future_warning_on_import``
  and ``test_public_encoding_does_not_trigger_developer_warnings``
  regressions lock both halves of this contract in place.

---

## [0.4.0b13] - 2026-04-24

Thirteenth beta — second-tier cleanup from the b11 independent
audit.  Three focused changes target the "research-code smell" areas
that were still visible in the production modules.  Full test suite:
**547 / 547**.

### Added

- **`shoebox_rir` / `shoebox_sh_rir`** gained an ``interpolation``
  keyword.  The default ``"nearest"`` preserves the existing
  round-to-sample Allen–Berkley semantics; ``"sinc"`` scatters a
  Kaiser-windowed fractional-delay FIR kernel (configurable
  ``fir_taps``, default ``21``), pushing the timing accuracy of each
  image source from ±0.5 samples down to ~±0.15 samples.  High-frequency
  comb structure around each reflection is noticeably cleaner at the
  cost of ``O(K · fir_taps)`` instead of ``O(K)`` writes.

### Changed

- **`fdn_reverb` / `fdn_sh_tail`** now validate that the supplied
  ``mixing_matrix`` is orthogonal (``M·Mᵀ = I`` within ``1e-6``) by
  default.  Passing a non-orthogonal matrix silently broke the
  RT60 calibration; it now raises ``ValueError`` up front.  Set
  ``check_orthogonality=False`` to opt out (e.g. for intentionally
  coloured / lossy feedback).
- **`dirac.dirac_analysis`** is now **normalisation-invariant**.
  The function gained a ``normalization`` keyword (default
  ``"orthonormal"``, matching the rest of the package); internally
  it canonicalises the input to **orthonormal** FOA coefficients
  and applies the ``1/√3`` velocity-channel scaling needed by
  DirAC's textbook ``ψ = 1 − ||I||/E`` formula.  Feeding the same
  physical field in any of the three supported normalisations now
  gives **bit-for-bit identical** ψ and DOA, and a coherent plane
  wave yields ``ψ ≈ 0`` exactly (previously a stray SN3D-vs-orthonormal
  scale error left a ``0.134`` residual).

### Fixed (release validation)

- `dirac.dirac_analysis` internal canonicalisation direction
  corrected.  An earlier b13 draft converted input to SN3D before
  applying ``1/√3`` on the velocity channels; the right canonical
  form for this package's coefficient-rescaling normalisation
  family is **orthonormal**, and a coherent plane wave now
  satisfies ``||I|| = E`` exactly.  Regression test at
  ``tests/test_dirac.py::test_plane_wave_diffuseness_is_low``.
- `shoebox_rir` / `shoebox_sh_rir` sinc mode now keeps image
  sources whose fractional-delay kernel support overlaps the
  output buffer, even when the kernel centre is just past the
  last sample.  An earlier b13 draft dropped those images using
  the nearest-sample mask, truncating legitimate tail energy.
  The new ``_write_mask`` helper picks the correct mask per
  interpolation mode.  Regression test at
  ``tests/test_room.py::test_sinc_keeps_kernel_that_overlaps_buffer_end``.

---

## [0.4.0b12] - 2026-04-24

Twelfth beta — API consistency sweep in response to the b11
independent audit.  Four focused changes close the remaining
"easy to use wrong" corners of the public surface.  Full test
suite: **537 / 537**.

### Fixed

- **`diffuseness.intensity_vectors_from_foa`** — the function used
  to assume the FuMa channel order ``[W, X, Y, Z]`` while every
  other module in the package (``ambi.intensity_vector``,
  ``dirac_analysis``, the plane-wave encoder, etc.) uses ACN
  ``[W, Y, Z, X]``.  Feeding an ACN FOA therefore silently produced
  a 90°-rotated DOA.  The function now defaults to ``channel_order="acn"``
  and accepts ``channel_order="fuma"`` for legacy B-format input.
  New regression tests lock the **direction** semantic (not just
  shape / range) so this cannot drift again.

### Changed (breaking defaults)

- **`dirac.dirac_analysis`** default ``coeff_axis`` changed from
  ``-1`` to ``-2`` so the function works out-of-the-box on STFT
  output in the package-wide ``(F, Q, T)`` layout.  Callers who were
  passing ``(F, T, Q)`` must now pass ``coeff_axis=-1`` explicitly.
- **`sh.wigner_small_d` / `wigner_D` / `sh_rotation_matrix_*` /
  `rotate_sh_coeffs`** default small-``d`` backend changed from
  ``"sakurai"`` to ``"jy"``.  The Sakurai summation starts losing
  precision past ``n ≈ 30``; the ``J_y``-diagonalisation backend
  stays unitary to ``~1e-14`` through at least ``n = 80`` and is
  now the safe default.  Pass ``method="sakurai"`` explicitly when
  you need the last few percent of speed at small ``n``.

### Packaging

- **`matplotlib` moved out of the hard runtime dependencies to the
  new ``[plotting]`` extra.**  Headless / server pipelines no longer
  need to install it.  ``sap.plotting`` entry points now raise a
  clean ``ImportError`` with install instructions if matplotlib is
  absent.  Install with
  ``pip install 'spherical-array-processing[plotting]'``.  A new
  regression test locks the "``import sap.plotting`` must succeed
  without matplotlib; only calling functions triggers the import"
  contract.
- Version bumped to `0.4.0b12`.

---

## [0.4.0b11] - 2026-04-24

Eleventh beta — sweeps up the "known shortcomings" list identified in
the b10 session review: **SOFA ``Data.Delay`` round-trip**,
**streaming-safe UHJ via FIR Hilbert**, **FOA scene translation**,
and a **saner default `seed`** for `fdn_sh_tail`.  Full test suite:
**533 / 533**.

### Added

- **`HRTFDataset.data_delay_samples`** — new optional field carrying
  SOFA ``Data.Delay`` values (shape ``(2,)`` per-ear or ``(M, 2)``
  per-direction).  ``load_sofa`` now preserves non-zero delays from
  the file and ``save_sofa`` writes them back, closing the SOFA
  round-trip that previously zeroed delay information.
- **`sap.ambi.translate_foa`** — virtual-listener translation for
  first-order ambisonic scenes via plane-wave decomposition →
  per-direction frequency-domain phase shift → re-encoding.
  Reproduces the leading-order geometric advance ``û · r / c`` for
  small translations (``|k r| ≪ 1``); for larger translations the
  first-order PWD point-spread function smears the result and the
  FFT-based fractional-delay step introduces periodic-extension
  artefacts that the zero-pad only mitigates, not eliminates.
- **`uhj_encode` / `uhj_decode`** gained a ``hilbert_method`` kwarg.
  ``"fft"`` (default) keeps the existing whole-block
  ``scipy.signal.hilbert`` behaviour; ``"fir"`` uses a Kaiser-
  windowed linear-phase FIR of configurable length (``fir_taps``,
  default ``513``) that has a fixed group delay and is therefore
  safe to use in block-by-block / streaming pipelines.

### Changed

- `fdn_sh_tail` default ``seed`` is now ``None`` (fresh directions
  each call).  Pass an explicit integer for reproducibility — the
  earlier hard-coded ``seed=0`` was surprising for users who wanted
  statistically independent diffuse tails on each render.
- Version bumped to `0.4.0b11`.

### Fixed (release validation)

- ``uhj_encode(hilbert_method="fir")`` now returns an output that
  strictly matches the input length for **every** input size.  The
  previous ``np.convolve(..., mode='same')`` did not preserve the
  input length when ``fir_taps > len(signal)`` — NumPy returns the
  longer array in that regime.  The implementation now performs
  a full convolution and trims it by the group delay
  ``(N-1)/2`` so the guarantee holds regardless of input size.
  Regression test ``test_fir_hilbert_short_signal_still_matches_input_length``.
- ``load_sofa`` tightens ``Data.Delay`` validation.  A
  non-zero ``(k, 2)`` block is only accepted when ``k`` equals
  ``1`` or ``hrirs.shape[0]`` — previously any ``k`` silently
  populated ``data_delay_samples``, which could produce a dataset
  whose delay table disagreed with ``source_grid`` length.
- ``translate_foa`` documentation tightened: the per-direction
  phase shift approximates the leading-order ``û · r / c`` geometric
  advance only for ``|k r| ≪ 1``, and the FFT zero-pad reduces but
  does not eliminate the periodic-extension artefact from fractional
  delays.  The earlier "exact" wording overclaimed the math.

---

## [0.4.0b10] - 2026-04-24

Tenth beta — rounds out the room-acoustics story with a
**Feedback-Delay-Network diffuse reverberator** and closes the HRTF
I/O loop with a **SOFA ``SimpleFreeFieldHRIR`` writer**.  Full test
suite: **510 / 510**.

### Added

- **`spherical_array_processing.hrtf.save_sofa`** — write an
  :class:`HRTFDataset` to a standards-compliant
  ``SimpleFreeFieldHRIR`` SOFA file.  Covers the mandatory
  ``Data.IR``, ``Data.SamplingRate``, ``Data.Delay``,
  ``SourcePosition`` (spherical in degrees), ``ReceiverPosition``
  (Cartesian (2, 3, 1)), listener / emitter blocks, and all the
  expected global attributes.  Extra ``dataset.metadata`` keys round-
  trip through ``load_sofa``.  Requires the ``[hrtf]`` extra.
- **`spherical_array_processing.room.fdn_reverb`** — Jot-style FDN
  monaural reverberator with configurable delay lengths, mixing
  matrix, and target ``RT60``.  The default 8-line Hadamard FDN
  reproduces a requested ``RT60`` within ~5 %.
- **`spherical_array_processing.room.fdn_sh_tail`** — ambisonic
  extension that scatters the FDN taps across random directions on
  the sphere and encodes each as a plane wave, producing a dense,
  spatially diffuse late-reverb tail ready to be added to an
  image-source early-reflection RIR.

### Changed

- Version bumped to `0.4.0b10`.
- `__init__.py` docstring notes the new `save_sofa` / `fdn_*`
  helpers.

### Fixed (release validation)

- `hrtf.save_sofa` now writes the SimpleFreeFieldHRIR-mandatory
  global attributes (``Version="2.1"``, ``AuthorContact``,
  ``Organization``, ``ListenerShortName``) plus a float64
  ``Data.Delay``, and drops the spurious ``Type``/``Units``
  attributes on ``ListenerUp``.  The output now passes the
  third-party ``sofar.verify()`` and opens cleanly in
  ``netCDF4.Dataset()``, not only the in-tree ``load_sofa``.
- `room.fdn_sh_tail` no longer forces the output dtype to
  ``float64`` — the ``basis="complex"`` path returns a
  ``complex128`` array as the API advertises.  Regression test
  ``test_complex_basis_returns_complex_output`` pins this.

---

## [0.4.0b9] - 2026-04-24

Ninth beta — adds **UHJ stereo interoperability** with the legacy
B-format ecosystem and **per-bin intensity analysis** for DirAC-style
source-activity / diffuseness research.  Full test suite: **487 /
487**.

### Added

- **`spherical_array_processing.ambi.uhj_encode`** and
  **`uhj_decode`** — Gerzon 1985 2-channel UHJ codec.  Encodes an FOA
  signal to stereo-compatible ``(L_T, R_T)`` via Hilbert-shifted
  linear combinations, and approximately recovers ``(W, X, Y)`` on
  decode (``Z = 0``, since UHJ-2 carries no vertical information).
  Handles arbitrary input normalisation by converting transparently
  through Furse-Malham internally.
- **`spherical_array_processing.ambi.intensity_vector`** and
  **`doa_from_intensity`** — active / reactive intensity
  decomposition from an FOA STFT.  ``intensity_vector`` returns the
  Cartesian ``Re{W^* (X, Y, Z)}`` (and optionally the reactive
  ``Im{W^* (X, Y, Z)}``) for each frequency / time bin;
  ``doa_from_intensity`` normalises it to a unit DOA vector.

### Changed

- Version bumped to `0.4.0b9`.
- `__init__.py` docstring notes the new UHJ and intensity helpers.

### Fixed (release validation)

- `ambi.uhj_encode` now applies the classical ``(S ± D)/2`` factor of
  ``1/2`` in the UHJ-2 ``L_T / R_T`` equations.  The previous
  implementation emitted ``L_T = S + D`` / ``R_T = S − D``, which was
  ``6.02 dB`` hotter than the published Gerzon matrix.  The quadrature
  ``X`` coefficient was also corrected from ``0.5098022`` to the
  standard-reference ``0.5098604``.  Regression tests
  ``test_y_only_uses_classical_half_gain`` and
  ``test_matches_classical_reference_formula_on_tone`` pin the output
  against the published UHJ-2 formulas.
- `ambi.intensity` module prose now matches the function docstring:
  the active intensity points **toward** the encoded source under
  the package's SH convention (previously the module-level narrative
  wrongly said "opposite to").
- `tests/test_ambi_intensity` standing-wave test switched to a
  bin-centered carrier; the previous tone leaked active-energy
  residue across bins and only passed because the reactive part was
  much larger.  Active/reactive now separate cleanly.

---

## [0.4.0b8] - 2026-04-24

Eighth beta — rounds out acoustic analysis and source-encoding
ergonomics: the package can now **measure standard acoustic
descriptors** from any room IR, **encode monaural sources as plane
waves** in one call, and **run a shoebox simulation with
octave-band wall absorption** for realistic frequency-dependent
reverb.  Full test suite: **459 / 459**.

### Added

- **`spherical_array_processing.room.rir_metrics`** and its per-metric
  components (`energy_decay_curve`, `reverberation_time`,
  `early_decay_time`, `clarity`, `definition`) — standard ISO 3382
  acoustic descriptors (EDC, RT60 via T20/T30/T60 regression, EDT,
  C50/C80, D50) from a monaural room impulse response.
- **`spherical_array_processing.ambi.encode_plane_wave`** — monaural
  → SH plane-wave encoder.  Accepts a single mono signal + single
  direction, or ``K`` mono signals + ``K`` directions (summed into
  one SH output).  Outputs any of the three normalisation conventions
  via the `normalization=` keyword.
- **`spherical_array_processing.room.shoebox_rir_banded`** —
  frequency-dependent shoebox RIR.  Takes per-wall octave-band
  reflection magnitudes ``(6, B)`` plus band edges ``(B+1,)`` and
  synthesises a linear-phase FIR per distinct bounce-count vector
  (grouped for efficiency) so the RIR's spectral shape evolves with
  reflection order.  Integrates with ``rir_metrics`` for objective
  measurement of the resulting decay.

### Changed

- Version bumped to `0.4.0b8`.
- `__init__.py` docstring notes the new `room.rir_metrics` +
  ``shoebox_rir_banded`` additions and the new ``encode_plane_wave``
  entry in `ambi`.

### Fixed (release validation)

- `room.banded._fir_from_bands` now realises **piecewise-constant**
  per-band gains as intended.  The previous construction passed only
  one sample per interior band edge to ``scipy.signal.firwin2``, so
  the solver linearly interpolated across the following band instead
  of honouring the step — a two-band ``[1, 0]`` target produced an
  average gain of ~0.5 in the upper half of the spectrum rather than
  a stopband.  Interior edges are now duplicated with the gain on
  each side, which is the standard ``firwin2`` idiom for stepped
  responses.  Regression test
  ``test_band_fir_respects_piecewise_constant_target`` pins this
  behaviour with a ``freqz`` passband / stopband mean check.
- Docstring tightened to note the centred-FIR pre-ringing tradeoff
  (up to ``fir_taps // 2`` samples of symmetric ringing around each
  reflection delay) so downstream callers can match
  ``fir_taps`` to their onset-timing requirements.

---

## [0.4.0b7] - 2026-04-24

Seventh beta — closes the "interoperability and convolution" gap: the
package can now **read and write AmbiX WAV files**, **convolve a dry
signal with a shoebox SH-RIR** to produce a reverberant ambisonic
capture, and generate **NFC-HOA distance filters** for near-field
rendering.  Full test suite: **428 / 428**.

### Added

- **`spherical_array_processing.ambi.read_ambix_wav`** and
  **`write_ambix_wav`** — AmbiX WAV I/O via ``soundfile`` (the
  ``[audio]`` extra).  Handles arbitrary ambisonic orders
  (``N = √n_ch − 1`` inferred from the channel count), converts
  between file and internal normalisation conventions on load/save,
  and supports both ``channels_first`` and ``channels_last`` layouts.
- **`spherical_array_processing.room.convolve_mono_to_ambi`** and
  **`convolve_sh_to_sh`** — ambisonic convolution reverb.  The
  mono-to-ambi variant convolves a dry monaural signal with each SH
  channel of a room-impulse response (e.g. from
  ``shoebox_sh_rir``); the channel-diagonal variant convolves a
  pre-encoded SH signal with a matching SH-RIR per-channel.
- **`spherical_array_processing.ambi.nfc_hoa_distance_filter`** —
  per-order near-field compensation filter
  ``F_n(k) = h_n^{(2)}(k·d_src) / h_n^{(2)}(k·R_ref)``.  Stabilised at
  DC via the closed-form ``(R/d)^{n+1}`` asymptote; integrates with
  :func:`encoding.apply_radial_equalizer` for per-SH-channel use.

### Changed

- Version bumped to `0.4.0b7`.
- `__init__.py` docstring notes the new `ambi` submodule additions
  (WAV I/O, NFC-HOA) and the convolution-reverb helpers under `room`.

### Fixed (release validation)

- `room.convolve_mono_to_ambi` now restores the user's axis layout
  correctly for any ``axis`` value, not just ``axis=-1``.  The
  previous ``np.moveaxis(out, -1, axis)`` pattern mis-placed the
  time axis for 1-D signals with ``axis=0`` and for multi-axis
  signals with a non-last time axis.  Two regression tests
  (``test_axis_zero_for_1d_signal_keeps_q_before_time``,
  ``test_non_last_time_axis_inserts_sh_axis_before_time``) lock the
  inserted-SH-axis convention.
- `room.convolve_sh_to_sh` now accepts arbitrary-rank inputs (batch
  axes before the SH/time axes); the earlier implementation assumed
  2-D inputs and silently convolved along the wrong axis for
  higher-rank signals.  Regression test
  ``test_batched_signal_with_non_default_axes`` pins the
  ``(batch, T, Q)`` use case.
- `room.convolve_sh_to_sh` now also rejects malformed scalar / 1-D
  inputs with a clean `ValueError` instead of leaking a
  `ZeroDivisionError` or an axis-configuration error from the internal
  axis-normalisation path.
- `ambi.read_ambix_wav` / `write_ambix_wav` now validate the
  ``axis`` literal on both paths and raise cleanly for unknown
  values.  A previous silent-coerce path was exposed by the test
  suite.
- `ambi.write_ambix_wav` stops forcing the output to ``float32``
  when ``subtype="DOUBLE"``.  The array is now handed to
  ``soundfile`` as-is so double-precision samples survive the
  round-trip bit-for-bit.
- `ambi.nfc_hoa_distance_filter` validates ``c > 0``; docstring now
  describes the correct ``(R/d)·exp(-ik(d-R)) + O(1/k)``
  high-frequency asymptote instead of the earlier "per-order
  amplitude roll-off" claim.

---

## [0.4.0b6] - 2026-04-24

Sixth beta — closes the binaural loop end-to-end, adds ambisonic
format interoperability, and provides an analytic HRTF generator for
testbeds without measured data.  Full test suite: **392 / 392**.

### Added

- **`spherical_array_processing.binaural.ambi_to_binaural_time_domain`** —
  one-call SH → stereo pipeline.  Takes a ``(Q, T)`` ambisonic signal
  plus an :class:`HRTFDataset`, builds a MagLS rendering filter,
  inverse-FFTs it to a real FIR, and convolves each SH channel into
  the two ears.  Accepts optional ZYZ Euler keyframes to drive
  :func:`sh.rotate_ambi_over_time` for head-tracked playback.
- **`spherical_array_processing.ambi`** (new module) — ambisonic
  format converters.  ``convert_ambi_normalization`` rescales SH
  coefficients between ``orthonormal`` / ``n3d`` / ``sn3d``
  conventions; ``acn_to_fuma`` / ``fuma_to_acn`` reorder and rescale
  between the AmbiX ACN basis and the legacy B-format FuMa ordering
  through third order.
- **`spherical_array_processing.hrtf.rigid_sphere_hrtf`** — closed-form
  rigid-sphere HRTF generator (Rayleigh / Duda-Martens).  Returns an
  :class:`HRTFDataset` consumable by the MagLS / BiMagLS renderers
  and the new ``ambi_to_binaural_time_domain`` pipeline.  Peak-to-peak
  ITD matches the Woodworth ``(r/c)(θ + sin θ)`` estimate within 2 %
  at ``head_radius_m = 0.085``.

### Changed

- Version bumped to `0.4.0b6`.
- `__init__.py` registers the new `ambi` lazy submodule and updates
  the `binaural` / `hrtf` docstrings.

### Fixed (release validation)

- `hrtf.rigid_sphere_hrtf` dropped a duplicated ``4π`` factor in the
  modal-series amplitude.  ``bn_matrix(sphere="rigid")`` already
  carries the ``4π·iⁿ`` prefactor, so the orthonormal SH addition
  theorem only needs ``H = Σ_q b_q · Y_q(ear) · Y_q*(src)``.  The DC
  bin now uses ``b_0(0) = 4π`` to give unit omnidirectional gain, and
  DC / Nyquist bins are explicitly real before ``irfft``.  Regression
  test ``test_frequency_response_matches_modal_simulator`` locks the
  output against ``simulate_sh_array_response`` to machine precision.
- `binaural.ambi_to_binaural_time_domain` drops the unnecessary
  ``sig.real if not complex else sig`` branch; `oaconvolve` now takes
  the SH signal directly.  A new test renders the same physical field
  from real-basis and complex-basis coefficient vectors and checks the
  binaural output matches to ``1e-10``.
- `tests/test_ambi_format.py` gained all-channels-active FuMa
  round-trip tests at orders 2 and 3 plus per-channel weight spot
  checks (T, V, M, Q).

---

## [0.4.0b5] - 2026-04-23

Fifth beta — rounds out the toolkit with a parametric **shoebox room
simulator**, a **head-tracked ambisonic rotation helper**, and a
public wrapper around Politis' **measured-steering-matrix encoding
filters**.  Full test suite: **363 / 363**.

### Added

- **`spherical_array_processing.room`** (new module) — rectangular-room
  image-source RIR generator à la Allen & Berkley (1979).
  `ShoeboxRoom` describes the geometry plus per-wall reflection
  coefficients, `shoebox_rir` returns a monaural IR plus the
  per-image arrival-direction / delay table, and `shoebox_sh_rir`
  encodes the same image sources into an `((N+1)², T)` ambisonic
  RIR ready for reverberant DOA / beamforming test vectors.
- **`spherical_array_processing.sh.rotate_ambi_over_time`** —
  head-tracking rotation helper.  Takes a ``(Q, T)`` (or ``(T, Q)``)
  ambisonic signal plus ``(K, 3)`` ZYZ Euler keyframes, splits the
  signal into ``K`` equal-length blocks, and applies the matching
  Wigner-D rotation to each block with a configurable linear
  matrix-lerp crossfade at every block boundary.  Backed by the
  ``method="jy"`` small-d solver so it stays unitary past ``N ≈ 25``.
- **`spherical_array_processing.encoding.measured_array_equalizer`**
  and **`apply_measured_equalizer`** — public API around
  `repro.politis.arraySHTfiltersMeas_{regLS, regLSHD}`.  Accepts a
  measured ``(F, M, G)`` complex steering matrix plus az/el grid and
  returns an ``(F, Q, M)`` per-bin encoding filter with regularised
  least-squares or SH-decomposed least-squares, optionally also the
  length-``n_fft`` time-domain FIR form.  `apply_measured_equalizer`
  multiplies this filter bin-by-bin into an arbitrary-layout
  multichannel STFT.

### Changed

- Version bumped to `0.4.0b5`.
- `__init__.py` registers the new `room` lazy submodule and notes the
  measured-filter API in the package docstring.

---

## [0.4.0b4] - 2026-04-15

Completes the "packaged-pipeline" story: b4 turns the individual
building blocks into a self-contained EM32 → SH → MagLS workflow and
closes the last b3 follow-ups (SOFA I/O, DirAC time-domain wrapper,
layout-coverage diagnostic).  Full test suite: **332 / 332**.

### Added

- **`spherical_array_processing.hrtf`** (new module) — `HRTFDataset`
  dataclass plus `load_sofa()` for SOFA AES69
  `SimpleFreeFieldHRIR` files.  Built on ``h5py`` as an optional
  dependency (``pip install 'spherical-array-processing[hrtf]'``);
  supports both ``spherical`` and ``cartesian`` ``SourcePosition``
  blocks and recovers ear positions from ``ReceiverPosition`` when
  present.  A ``to_frequency_domain()`` helper hands the HRTFs
  straight to the MagLS / BiMagLS renderers.
- **`spherical_array_processing.dirac.dirac_render_time_domain`** —
  end-to-end time-domain wrapper around `stft` →
  `dirac_analysis` → `dirac_synthesize` → `istft`, so users can go
  from a (Q, T) ambisonic signal to loudspeaker feeds in one call.
- `spherical_array_processing.decoding.check_layout_coverage` — a
  diagnostic that reports ``max_gap_deg`` / ``mean_gap_deg`` /
  ``uncovered_fraction_above_30deg`` so users can sanity-check a
  loudspeaker layout before wiring it into VBAP / AllRAD.
- `suggest_imaginary_loudspeakers` gained `max_imaginary`, `strict`,
  and `n_probe_points` keyword arguments.  ``strict=True`` raises
  ``ValueError`` when the residual gap after placing
  ``max_imaginary`` speakers still exceeds the requested
  ``min_cap_half_width_deg``.
- **`examples/binaural_em32_to_ears.py`** — worked pipeline script
  that simulates an Eigenmike-32 capture of a plane-wave, SH-encodes
  with Tikhonov radial equalisation, and renders MagLS binaural audio
  using a synthetic HRTF built in-process (no external data required).
  Covered end-to-end by ``tests/test_example_end_to_end.py``.

### Changed

- Version bumped to `0.4.0b4`.
- `__init__.py` registers the new `hrtf` lazy submodule.
- `pyproject.toml` declares the optional ``[hrtf]`` dependency group
  (``h5py>=3.6``).

### Fixed (release validation)

- `dirac_render_time_domain` no longer mis-classifies short
  channels-first inputs (e.g. ``(Q=4, T=2)``) as time-major.  The new
  heuristic prefers the axis whose length matches an ``(N+1)²`` SH
  channel count; falls back to ``shape[0] ≤ shape[1]`` only when
  both axes are ambiguous.  Regression at
  ``tests/test_dirac.py::test_dirac_render_time_domain_accepts_short_channels_first_input``.
- `hrtf.load_sofa` now strictly validates ``SourcePosition.Type``
  against ``{"spherical", "cartesian"}`` and accepts only the exact
  ``(2, 3, 1)`` or ``(2, 3)`` shapes for ``ReceiverPosition``.
  Previously an unknown ``Type`` silently defaulted to ``spherical``
  and unusual ear-position shapes were forced through
  ``.reshape(-1, 3)``; both paths could feed nonsense geometry into
  BiMagLS.  Two regression tests added.
- Example `examples/binaural_em32_to_ears.py` switched to ceil-framing
  with tail padding so the overlap-add convolution is robust to
  durations that are not an integer multiple of the hop.

---

## [0.4.0b3] - 2026-04-15

Focused on *closing practical pipeline gaps* left open by b2 — mostly
user-visible ergonomics around partial loudspeaker layouts, low-order
binaural rendering, robust adaptive-beamforming covariance, dual-band
decoding, and parametric spatial coding (DirAC).  All A-tier items
shipped with cross-module regression tests; full suite is at **314 /
314** passing.

### Added

- **`spherical_array_processing.decoding.suggest_imaginary_loudspeakers`** —
  heuristic auxiliary-speaker placement that closes the convex hull
  of hemispherical / partial layouts.  The returned grid is consumed
  by `vbap_gains` and `allrad_decoder`; imaginary rows are dropped
  from the final decoder matrix after the VBAP solve, so the caller
  still gets ``L`` real-speaker gains.
- `vbap_gains(..., imaginary_loudspeakers=)` and
  `allrad_decoder(..., imaginary_loudspeakers=, auto_close_hull=)` —
  first-class support for non-enclosing layouts, addressing a
  release-validation finding that the previous centroid-fallback could
  produce ~20° angular errors on a pure upper-dome.
- **`spherical_array_processing.decoding.dual_band_decoder_matrix`** and
  **`apply_dual_band_decoder`** — Daniel-style dual-band ambisonic
  decoding.  ``D_lf`` is the base decoder (SAD / MMD / EPAD / AllRAD);
  ``D_hf`` is the same matrix post-multiplied by the max-rE per-SH
  taper with an RMS-preserving scalar.  The apply helper crossfades
  between them with a power-complementary Butterworth response.
  Also exposes the underlying `max_re_sh_weights` helper.
- **`spherical_array_processing.binaural.bimagls_binaural_filters`** —
  bilateral-Ambisonics MagLS (Engel, Goodwin, Alon 2021): time-aligns
  the HRTF phases around each ear before MagLS, returning
  delay-aligned SH filters alongside SH-encoded per-direction ear
  delays so users can reattach the ITD at render time.
- **`spherical_array_processing.covariance`** (new module) — covariance
  regularisers for robust SH-domain adaptive beamforming and DOA:
  Ledoit-Wolf data-driven shrinkage (numerically verified bit-exact
  against ``sklearn.covariance.ledoit_wolf``), Oracle-Approximating
  Shrinkage (Chen et al. 2010), forward-backward averaging, and a
  trace-relative diagonal loader.
- **`spherical_array_processing.dirac`** (new module) — parametric
  Directional Audio Coding (Pulkki 2007).  ``dirac_analysis`` derives
  per-bin DOA + diffuseness from an FOA STFT with a single-pole IIR
  time-smoothing; ``dirac_synthesize`` re-renders to any VBAP-capable
  loudspeaker layout (with optional per-bin phase decorrelation for
  the diffuse component).

### Changed

- Version bumped to `0.4.0b3` in `pyproject.toml` and
  `spherical_array_processing.__version__`.
- `__init__.py` registers the new lazy submodules `covariance` and
  `dirac`.
- `dirac_analysis` docstring now clarifies that the diffuseness
  calibration assumes **ACN / SN3D FOA** scaling: the DOA estimate
  remains correct under other conventions (including the package
  default orthonormal SH), but ``ψ`` values are biased away from the
  textbook ``[0, 1]`` range.

### Fixed (release validation)

- `bimagls_binaural_filters` no longer forces `delay_sh_coeffs` to
  ``float`` — complex-basis inputs with off-axis ear positions now
  keep their imaginary parts, so the SHT round-trip of the delay
  table is exact.  Real-basis callers still get real coefficients via
  ``np.real_if_close``.
- `covariance.diagonal_loading` now Hermitian-symmetrises its output
  (``0.5·(out + outᴴ)``) so that nearly-Hermitian inputs come out
  exactly Hermitian, matching the contract of the other regularisers
  in the module.
- `apply_dual_band_decoder` raises `ValueError` when ``coeff_axis``
  would collide with the leading frequency axis, and validates
  ``crossover_hz > 0`` / ``crossover_order > 0`` up front rather than
  letting a zero divisor leak into the magnitude response.
- `covariance.ledoit_wolf_shrinkage` — tightened the Frobenius-inner-
  product derivation of ``π̂``.  The optimised form now reproduces
  the per-snapshot loop version to the numerical noise floor for
  complex data and stays bit-exact against
  ``sklearn.covariance.ledoit_wolf`` on the real path.

---

## [0.4.0b2] - 2026-04-15

The second beta closes the Ambisonic encode–decode loop and adds
closed-form, subspace, and model-selection DOA tooling.  All seven
bullets below arrived with cross-module regression tests; the full
suite is at **282 / 282** passing.

### Added

- **`spherical_array_processing.decoding`** (new module) — ambisonic
  loudspeaker decoders: `sad_decoder`, `mmd_decoder`, `epad_decoder`,
  `allrad_decoder`, plus the unified dispatch entry point
  `decoder_matrix(method={"sad","mmd","epad","allrad"})` and an
  `apply_decoder` tensor helper.  Includes a `vbap_gains` utility that
  amplitude-pans virtual sources onto the loudspeaker convex hull via
  :class:`scipy.spatial.ConvexHull`.
- **`spherical_array_processing.binaural`** (new module) — MagLS
  (Schörkhuber & Zotter 2018) SH → binaural rendering filters with
  optional phase-continuation seeding.  Below a user-specified cutoff
  frequency it is exact complex least-squares; above it is
  alternating-projection magnitude LS.
- **`spherical_array_processing.stft`** (new module) — thin
  scipy.signal.stft / istft wrappers that reshape the output into the
  ``(F, M, T)`` layout consumed by :func:`doa.srp_map` and the
  encoding filters.
- `spherical_array_processing.sh.rotation` — new ``method="jy"``
  backend on `wigner_small_d`, `wigner_D`, `sh_rotation_matrix_*`,
  and `rotate_sh_coeffs`.  Implements ``d^n(β) = exp(-i β J_y^{(n)})``
  by Hermitian diagonalisation of the tridiagonal angular-momentum
  operator; unitarity stays below ``1e-14`` through ``n = 80`` while
  the classical Sakurai summation degrades to ``~1e-6`` past ``n = 40``.
- `spherical_array_processing.doa.esprit_doa` — closed-form
  Eigenbeam-ESPRIT DOA estimator based on the Jo & Choi (2019)
  three-recurrence formulation (originally translated into the
  `repro.politis` layer; now promoted to first-class public API).
- `spherical_array_processing.doa.estimate_n_sources` — AIC / MDL
  model-selection source counter following Wax & Kailath (1985).
  Accepts either an SH covariance matrix or its eigenvalues directly.
- `spherical_array_processing.acoustics.plane_wave_radial_bn` (and the
  propagated `bn_matrix`, `sph_modal_coeffs`,
  `simulate_sh_array_response`) now accept ``sphere="directional"``
  with a ``dir_coeff`` keyword argument, implementing the general
  first-order capsule ``α + (1-α) cos θ``.  ``"cardioid"`` is now a
  documented alias for ``α = 0.5``.

### Changed

- `simulate_sh_array_response` no longer forces ``H[0, :, :] = 1`` when
  ``array_type="directional"``; the DC bin is set analytically to
  ``α + (1-α) cos γ_{ms}``.  (Non-breaking for the pre-existing
  ``open`` / ``rigid`` / ``cardioid`` types — their DC logic is
  unchanged.)
- `sph_modal_coeffs` now **raises** `ValueError` when given an unknown
  ``array_type``, instead of silently falling back to rigid-sphere
  coefficients.  Misspellings such as ``"rigidd"`` or ``"direction"``
  surface immediately as exceptions.

### Fixed

- **`magls_binaural_filters(phase_continuation=True)`** now actually
  uses the seeded phase: in v0.4.0b1 the seed was overwritten on the
  first iteration of the alternating-projection loop, so the option
  was a silent no-op.  The first LS solve at every high-frequency bin
  now consumes ``|H[f]|·exp(j·prev_phase)`` and the bin-to-bin phase
  is propagated through the final projection.
- **`vbap_gains` and `allrad_decoder`** raise an informative
  `ValueError` (instead of a raw ``scipy.spatial.QhullError``) when
  the loudspeaker layout is coplanar / degenerate — for example a
  purely horizontal ring without any elevated speakers — and now warn
  (or raise, with ``strict=True``) when individual virtual-source
  directions fall outside the convex hull, rather than silently
  panning to the closest centroid.

### Repository and packaging

- Version bumped to `0.4.0b2` in `pyproject.toml` and
  `spherical_array_processing.__version__`.
- `__init__.py` exposes two new lazy submodules, `decoding` and
  `binaural`, plus the `stft` module.

---

## [0.4.0b1] - 2026-04-15

This is the first beta of the 0.4.0 release.  The package surface has
grown substantially beyond the 0.3.0 open-source baseline, along with a
handful of **intentional, behavior-changing** fixes to align sign and
normalization conventions across modules.  See the "Changed" section
below before upgrading.

### Added

- `spherical_array_processing.array.simulate_sh_array_response` — closed-form
  plane-wave array response that handles **open**, **rigid**, and
  **cardioid** spherical arrays via the modal-coefficient / Legendre
  addition-theorem form.  Matches `simulate_plane_wave_array_response`
  for `array_type="open"` to machine precision at sufficiently high
  SH order.
- New `spherical_array_processing.sh.rotation` module — Wigner small-d
  (`wigner_small_d`), full Wigner-D (`wigner_D`), block-diagonal
  coefficient rotation matrices for complex and real SH
  (`sh_rotation_matrix_complex`, `sh_rotation_matrix_real`), and a
  high-level `rotate_sh_coeffs` helper.  Validated to machine precision
  against direct SH re-evaluation at rotated points.
- New `spherical_array_processing.encoding` module — radial equalization
  filters for SH-domain encoding / DOA pipelines: Tikhonov
  (`radial_equalizer_tikhonov`), WNG-limited tanh soft-limit
  (`radial_equalizer_wng_limited`), and a unified entry point
  `radial_equalizer` plus `apply_radial_equalizer` for broadcasting
  over a frequency / coefficient tensor.
- New DOA estimators `srp_map` and `srp_map_from_covariance` in
  `spherical_array_processing.doa`, covering classical mic-domain
  SRP and SRP-PHAT with optional band-limit, PHAT weighting, and
  time-frame accumulation.
- Canonical array presets in `spherical_array_processing.array`:
  `em32_eigenmike` (mh acoustics Eigenmike-32 rigid sphere),
  `tetrahedral_array` (A-format 4-mic tetrahedron with front/upright
  orientations), `cubic_array`, and `circular_array`.
- `spherical_array_processing.doa.peak_pick_spectrum_nms` helper and a
  new `min_separation_deg` keyword argument on `pwd_spectrum`,
  `music_spectrum`, `srp_map`, `srp_map_from_covariance`, and
  `spatial_spectrum_from_map`, enabling angular non-maximum suppression
  during multi-source peak picking.  Defaults preserve 0.3.0 top-N
  behaviour when the keyword is omitted.
- Cross-module consistency tests (`tests/test_cross_module_consistency.py`)
  that pin down the sign / normalization / convention contract between
  the simulator, `bn_matrix`, the SHT, and DOA estimation.
- Expanded docstrings for `pwd_spectrum` describing the covariance
  construction convention (users must conjugate SHT output before
  building the outer-product covariance when using the complex SH
  basis; the real basis is self-conjugate and needs no transform).

### Changed (breaking)

- **Phase convention of `simulate_plane_wave_array_response`** now
  matches the DOA convention used everywhere else in the package
  (`exp(+j k · û_s · r)` instead of `exp(-j k · û_s · r)`).  Down-stream
  code that interpreted `source_grid` as *propagation direction* will
  see a sign flip in the complex transfer function; code that consumed
  only `abs(H)` or DOA estimates is unaffected.
- **Cardioid convention of `plane_wave_radial_bn(sphere="cardioid")`**
  now returns `2π iⁿ (j_n − j·j_n')`, matching the standard unit-front-gain
  cardioid pattern `0.5·(1 + cos θ)`.  The previous 4π factor
  corresponded to an un-normalised `1 + cos θ` capsule and was only
  present in the pre-release `[Unreleased]` branch.

### Fixed

- Corrected the docstring comment in `plane_wave_radial_bn` that claimed
  `h_n^(2)` matches the `e^{-iωt}` engineering convention — it is in
  fact `e^{+jωt}` (the convention produced by `numpy.fft.ifft` on a
  conjugate-symmetric spectrum).
- Clarified the `beam_weights_dolph_chebyshev(design_criterion="mainlobe")`
  parameter semantics: it is the main-lobe *half-width* (first-null
  location from the main-beam axis), not the full null-to-null width.
- **`simulate_sh_array_response(array_type="cardioid")`** no longer
  forces the DC bin to unity.  The hard-coded override erased the
  capsule's own directional response at DC; the new code preserves the
  analytic limit ``0.5·(1 + cos γ_{ms})`` for cardioid and keeps the
  omnidirectional ``1`` for open / rigid.
- **`radial_equalizer_wng_limited`** at exact modal zeros now saturates
  at the advertised ``max_gain_db`` ceiling.  The previous
  ``np.isfinite → 0`` mask collapsed the filter to silence wherever
  ``|B_n(kr)| = 0``, which contradicted the docstring.
- **`tetrahedral_array(orientation="upright")`** now actually rotates
  the first capsule to ``+z`` via the same Rodrigues construction used
  for ``"front"``.  Previously the function returned the identity
  rotation, contradicting the docstring.
- **`pwd_spectrum` and `music_spectrum`** now compute ``y(q̂)ᵀ · R ·
  y(q̂)*`` instead of ``y(q̂)ᴴ · R · y(q̂)``.  The new formula matches
  the package's physical SHT covariance ``R = E[c cᴴ]`` with ``c ∝
  Y_n^m*(k̂_src)``; users building ``R`` from ``direct_sht`` output no
  longer need a manual conjugation step.  Existing callers that
  construct ``R`` directly from ``Y(k̂)`` (not the SHT) must now use
  ``outer(conj(Y), Y)`` instead of ``outer(Y, conj(Y))`` to preserve
  the peak at the source direction.  For the real SH basis this change
  is a no-op.

### Repository and packaging

- Version bumped to `0.4.0b1` in `pyproject.toml` and
  `spherical_array_processing.__version__`.
- Added `encoding` to the lazy-loaded submodule surface exposed by
  `spherical_array_processing.__init__`.

---

## [Unreleased — folded into 0.4.0b1]

### Added

- Added ACN inverse/index-list helpers, spherical Neumann and second-kind
  Hankel radial wrappers, acoustic wavenumber helpers, Butterworth modal
  weights, Dolph-Chebyshev modal weights, and shared axisymmetric weight
  normalization to close core API gaps found in the reference packages.
- Added an optional external numeric audit against `spaudiopy`,
  `sound_field_analysis`, `spharpy`, and `sphericart`.

### Fixed

- Fixed `beam_weights_supercardioid` so all orders use the documented maximum
  front-back ratio design instead of mixing in incompatible low-order tables.

### Repository and packaging

- Added source distribution controls so the packaged test subset does not rely
  on repository-only MATLAB sources or example scripts.
- Added CI coverage for the declared Python support range and a package build
  job that verifies wheel and source distribution behavior.
- Added open-source governance files and third-party provenance notes.
- Made top-level submodule access lazy so `import spherical_array_processing`
  no longer imports plotting dependencies until `sap.plotting` is requested.

## [0.3.0] - 2026-03-27

### Added

- Core Python package for spherical harmonics, spatial sampling, array
  simulation, beamforming, DOA estimation, diffuseness, coherence, acoustics,
  plotting, and MATLAB reproduction layers.
- Experimental stereo-to-incomplete-FOA estimators and repository regression
  tooling for MATLAB parity and figure comparison.

### Packaging

- Declared package metadata in `pyproject.toml`.
- Included `py.typed` for typed package consumers.
