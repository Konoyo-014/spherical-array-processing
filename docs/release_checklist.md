# Release Checklist

Use this checklist from the repository root before tagging or publishing a
release.

## Repository Boundary

The open-source repository root is this directory, not the outer migration
workspace. Do not commit local virtual environments, coverage files, generated
build outputs, generated figures, private migration bundles, or large reference
assets.

## Local Validation

Run the full test suite and all tutorial entry points.

```bash
python -m pytest -q --tb=short
python examples/core/basic_usage.py
python examples/tutorials/01_sht_and_beamforming.py
python examples/tutorials/02_simulated_doa_pipeline.py
python examples/tutorials/03_modal_equalization_pipeline.py
```

Run notebook code cells if notebook content changed. A lightweight check is to
execute the code cells with `exec` in order, or use `jupyter nbconvert --execute`
when Jupyter is installed.

## Build Validation

Build both distribution formats, then verify the source distribution from a
fresh environment.

```bash
python -m build
tmpdir="$(mktemp -d)"
tar -xzf dist/spherical_array_processing-*.tar.gz -C "$tmpdir"
cd "$tmpdir"/spherical_array_processing-*
python -m pip install ".[dev]"
python -m pytest -q --tb=short
```

Install the wheel in a clean environment and verify import, version, and typed
package marker.

```bash
python -m pip install dist/spherical_array_processing-*-py3-none-any.whl
python - <<'PY'
import importlib.resources as ir
import spherical_array_processing as sap
print(sap.__version__)
print(ir.files("spherical_array_processing").joinpath("py.typed").is_file())
PY
```

Run packaging checks before upload.

```bash
python -m twine check dist/*
python -m spherical_array_processing.examples.plane_wave_doa
python -m spherical_array_processing.examples.binaural_em32_to_ears
```

## Metadata

Confirm that `pyproject.toml`, `spherical_array_processing.__version__`,
`CITATION.cff`, `CHANGELOG.md`, and any release tag all agree on the version.
Confirm that `README.md`, `docs/getting_started.md`, `docs/concepts.md`, and
`examples/README.md` still describe runnable entry points.
