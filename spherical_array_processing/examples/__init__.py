"""Install-safe runnable demo scripts.

This subpackage bundles a handful of small, self-contained example
scripts that exercise the **stable public API only** (no dependency on
the developer-only ``repro`` / ``regression`` / ``experimental`` layers
and no reliance on repo-side artefacts such as ``scripts/``,
``artifacts/``, or external ``.sofa`` files).  Everything here is part
of the wheel so that users who ``pip install spherical-array-processing``
immediately have runnable, version-pinned examples on hand.

Running the examples
--------------------

Each script is importable as a regular module and also runnable with
``python -m`` so the wheel-installed form "just works" without any
repo checkout::

    python -m spherical_array_processing.examples.plane_wave_doa
    python -m spherical_array_processing.examples.binaural_em32_to_ears

Programmatic use is also supported — every script exposes a
``run_example(...)`` function that returns a dictionary of
intermediate tensors and sanity metrics::

    from spherical_array_processing.examples.plane_wave_doa import (
        run_example,
    )
    out = run_example(az_deg=73.0, col_deg=62.0)
    print(out["recovered_az_deg"], out["recovered_col_deg"])

Scope
-----

These demos are intentionally small: their goal is to act as
copy-pasteable starting points and as smoke tests for the public API
boundary.  For more elaborate, research-flavoured reproduction scripts
(Rafaely chapter figures, Politis comparisons, etc.) see the
repository's top-level ``scripts/`` and ``examples/rafaely/`` trees —
those are shipped only in the source distribution and developer
workflows, not in the runtime wheel.
"""

__all__ = [
    "binaural_em32_to_ears",
    "plane_wave_doa",
]
