"""Smoke + correctness test for the end-to-end Eigenmike→MagLS example."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def example_module():
    """Import ``examples/binaural_em32_to_ears.py`` as a module."""
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        module = importlib.import_module("binaural_em32_to_ears")
    finally:
        sys.path.pop(0)
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
