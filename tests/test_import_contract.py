from __future__ import annotations

import subprocess
import sys

import pytest


def test_top_level_import_is_lightweight_until_plotting_is_requested():
    """``matplotlib`` is now an optional dependency behind the
    ``[plotting]`` extra.  Neither ``import
    spherical_array_processing`` nor accessing ``sap.sh`` /
    ``sap.plotting`` pulls it in — only calling a plotting entry
    point does.
    """
    code = """
import sys
import spherical_array_processing as sap
print('matplotlib' in sys.modules)
_ = sap.sh
print('matplotlib' in sys.modules)
_ = sap.plotting
print('matplotlib' in sys.modules)
# Calling a plotting function triggers the import.
sap.plotting.apply_matlab_like_style()
print('matplotlib' in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["False", "False", "False", "True"]


def test_built_wheel_does_not_ship_developer_only_layers():
    """Build a wheel, inspect its contents, confirm ``repro`` /
    ``regression`` / ``experimental`` are **not** packaged.

    Users who ``pip install`` the distribution should see only the
    stable public surface.  Developer-only layers remain in the
    source tree (and sdist) so editable installs and in-tree tests
    keep working, but the wheel boundary is kept clean.
    """
    import pathlib
    import zipfile
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    wheels = sorted(repo_root.glob("dist/*.whl"))
    if not wheels:
        pytest.skip("no wheel in dist/ — run `python3 -m build --wheel` first")
    latest = wheels[-1]
    with zipfile.ZipFile(latest) as zf:
        names = zf.namelist()
    offenders = [
        n for n in names
        if any(
            n.startswith(f"spherical_array_processing/{layer}/")
            for layer in ("repro", "regression", "experimental")
        )
    ]
    assert not offenders, (
        f"wheel {latest.name} unexpectedly ships developer-only "
        f"files: {offenders[:5]}..."
    )


def test_built_wheel_ships_install_safe_examples_subpackage():
    """The ``spherical_array_processing.examples`` sub-package is part
    of the public install surface since 0.4.0b15 — wheel users must be
    able to run ``python -m spherical_array_processing.examples.<name>``
    without any repo checkout.  If a packaging change accidentally
    excludes it, this test makes that regression impossible to miss.
    """
    import pathlib
    import zipfile
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    wheels = sorted(repo_root.glob("dist/*.whl"))
    if not wheels:
        pytest.skip("no wheel in dist/ — run `python3 -m build --wheel` first")
    latest = wheels[-1]
    with zipfile.ZipFile(latest) as zf:
        names = set(zf.namelist())
    required = {
        "spherical_array_processing/examples/__init__.py",
        "spherical_array_processing/examples/binaural_em32_to_ears.py",
        "spherical_array_processing/examples/plane_wave_doa.py",
    }
    missing = sorted(required - names)
    assert not missing, (
        f"wheel {latest.name} is missing install-safe example files: {missing}"
    )


def test_developer_only_submodules_emit_future_warning_on_import():
    """repro / regression / experimental are developer-only.  Importing
    any of them must emit a FutureWarning and they must NOT be
    exposed as attributes on the top-level ``sap`` namespace."""
    code = """
import warnings
import spherical_array_processing as sap

# Top-level namespace must NOT expose these.
for name in ('repro', 'regression', 'experimental'):
    assert not hasattr(sap, name), f"sap.{name} leaked to public API"
    assert name not in sap.__all__, f"{name} listed in sap.__all__"

# Importing them directly must emit FutureWarning.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    import spherical_array_processing.repro  # noqa: F401
    import spherical_array_processing.regression  # noqa: F401
    import spherical_array_processing.experimental  # noqa: F401

fw = [w for w in caught if issubclass(w.category, FutureWarning)]
names_warned = sorted(str(w.message).split()[0].split('.')[-1] for w in fw)
assert names_warned == ['experimental', 'regression', 'repro'], (
    f"expected exactly one FutureWarning from each dev submodule; got {names_warned}"
)
for w in fw:
    msg = str(w.message)
    assert 'not part of the stable public API' in msg
    assert 'pin an exact spherical-array-processing version' in msg
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "OK"


def test_stable_public_submodules_do_not_emit_dev_api_future_warnings():
    """Stable public submodules should not surface developer-only API
    warnings during import."""
    code = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    import spherical_array_processing.encoding  # noqa: F401
    import spherical_array_processing.binaural  # noqa: F401
    import spherical_array_processing.decoding  # noqa: F401
    import spherical_array_processing.room  # noqa: F401

fw = [w for w in caught if issubclass(w.category, FutureWarning)]
assert fw == [], f"stable public imports emitted FutureWarning: {[str(w.message) for w in fw]}"
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "OK"


def test_plotting_submodule_imports_without_matplotlib_until_called():
    """Masking ``matplotlib`` out of ``sys.modules`` must still allow
    importing ``spherical_array_processing.plotting``.  Only calling a
    plotting entry point should fail, with a clean install hint.
    """
    code = """
import sys
sys.modules.pop('matplotlib', None)
sys.modules.pop('matplotlib.pyplot', None)
sys.modules['matplotlib'] = None
sys.modules['matplotlib.pyplot'] = None

import spherical_array_processing.plotting as plotting
print(plotting.__name__)
try:
    plotting.apply_matlab_like_style()
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "spherical_array_processing.plotting",
        "ImportError",
        "matplotlib is required for the plotting submodule; install with `pip install 'spherical-array-processing[plotting]'` or `pip install matplotlib`.",
    ]
