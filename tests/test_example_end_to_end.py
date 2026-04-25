"""Smoke + correctness test for the end-to-end Eigenmike→MagLS example.

Since 0.4.0b15 the example implementation lives inside the wheel as
``spherical_array_processing.examples.binaural_em32_to_ears`` (see the
A5 install-state-examples work).  The repo-side
``examples/binaural_em32_to_ears.py`` is now a thin shim that re-exports
from there, but it is **pruned from the sdist**, so this test must
import the package-side module to remain runnable both in repo
checkouts and in sdist-only test runs (e.g. the GitHub Actions
``package`` job).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def example_module():
    """Import the in-package binaural example module.

    Equivalent to what end users get after ``pip install`` — i.e. via
    ``from spherical_array_processing.examples import binaural_em32_to_ears``.
    """
    from spherical_array_processing.examples import (
        binaural_em32_to_ears as module,
    )
    return module


def test_example_right_source_favours_right_ear(example_module):
    out = example_module.run_example(
        source_az_deg=60.0, source_col_deg=80.0,
        fs=16000.0, duration_s=0.3,
    )
    # Source az=60° is in the +x/+y quadrant — closer to the right
    # ear (x=+0.09) so right-ear RMS energy must exceed the left.
    assert out["right_energy"] > out["left_energy"] * 1.2


def test_example_left_source_favours_left_ear(example_module):
    out = example_module.run_example(
        source_az_deg=240.0, source_col_deg=80.0,
        fs=16000.0, duration_s=0.3,
    )
    assert out["left_energy"] > out["right_energy"] * 1.2


def test_example_binaural_shape(example_module):
    out = example_module.run_example(
        source_az_deg=0.0, source_col_deg=90.0,
        fs=16000.0, duration_s=0.2,
    )
    assert out["binaural"].ndim == 2
    assert out["binaural"].shape[0] == 2
    assert out["binaural"].shape[1] > 0
