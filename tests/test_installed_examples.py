"""Contract tests for the ``spherical_array_processing.examples``
sub-package introduced in 0.4.0b15.

The sub-package is what a wheel-installed user sees when they type
``python -m spherical_array_processing.examples.<name>``.  These tests
pin the shape of that surface so we do not accidentally break it when
refactoring the repo-side ``examples/`` tree.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


def test_examples_package_importable():
    import spherical_array_processing.examples as ex
    assert "binaural_em32_to_ears" in ex.__all__
    assert "plane_wave_doa" in ex.__all__


def test_plane_wave_doa_recovers_dense_source_direction():
    from spherical_array_processing.examples.plane_wave_doa import (
        run_example,
    )
    out = run_example(az_deg=73.0, col_deg=62.0)
    # Both estimators should be inside the scan-grid resolution floor.
    floor = out["grid_resolution_deg"]
    assert out["pwd_error_deg"] < 2.0 * floor
    assert out["music_error_deg"] < 2.0 * floor
    # And they should agree with each other on a rank-1 cov.
    np.testing.assert_allclose(
        out["pwd_error_deg"], out["music_error_deg"], atol=1e-6,
    )


def test_binaural_em32_example_importable_from_package():
    """The package-side binaural example must expose the same public
    entry point as the repo-side shim."""
    from spherical_array_processing.examples import (
        binaural_em32_to_ears as pkg,
    )
    assert callable(pkg.run_example)
    assert callable(pkg.main)


def test_repo_side_shim_reexports_package_implementation():
    """``examples/binaural_em32_to_ears.py`` is now a compatibility
    shim; it must delegate to the package-side implementation."""
    import importlib
    import sys
    from pathlib import Path
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        shim = importlib.import_module("binaural_em32_to_ears")
        from spherical_array_processing.examples.binaural_em32_to_ears import (
            run_example as pkg_run_example,
        )
        # Same underlying function object.
        assert shim.run_example is pkg_run_example
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize(
    "module",
    [
        "spherical_array_processing.examples.plane_wave_doa",
    ],
)
def test_example_runs_under_python_m(module):
    """``python -m <module>`` is the canonical recipe documented in
    the examples sub-package; prove it actually executes.

    ``binaural_em32_to_ears`` is intentionally excluded here: it runs
    a full STFT + MagLS binaural render (~1 s wall time) and is
    already covered by ``tests/test_example_end_to_end.py``.  Running
    it again under ``-m`` would only duplicate that coverage at the
    cost of doubling this test's wall time.
    """
    result = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    # Every example prints a sanity summary to stdout.
    assert len(result.stdout.strip().splitlines()) >= 1
