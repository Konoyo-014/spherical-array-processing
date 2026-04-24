"""Repo-side entry point for the Eigenmike → MagLS binaural example.

Since 0.4.0b15 the implementation lives inside the package at
:mod:`spherical_array_processing.examples.binaural_em32_to_ears` so
that wheel users get the example "for free" after ``pip install``.
This file remains as a back-compatible shim so pre-existing invocations
such as ``python examples/binaural_em32_to_ears.py`` keep working
against the source tree.
"""

from __future__ import annotations

from spherical_array_processing.examples.binaural_em32_to_ears import (  # noqa: F401
    _synthetic_sofa,
    main,
    run_example,
)


if __name__ == "__main__":  # pragma: no cover — manual invocation
    main()
