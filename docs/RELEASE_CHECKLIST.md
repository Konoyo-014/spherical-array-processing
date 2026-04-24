# Release Checklist

Use this checklist before publishing `spherical-array-processing`.

## Repository Boundary

Confirm that the release root is the top-level package repository and that no
nested git worktrees, generated artifacts, local caches, or internal bundles are
part of the release commit.

Large MATLAB reference assets should be documented as third-party materials and
kept out of the PyPI wheel.  If they must remain available in the public source
repository, prefer submodules or documented download steps.

## Validation

Run the full repository suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q --tb=short
```

Build distributions and verify the source distribution test subset:

```bash
python -m build
tmpdir="$(mktemp -d)"
tar -xzf dist/spherical_array_processing-*.tar.gz -C "$tmpdir"
cd "$tmpdir"/spherical_array_processing-*
python -m pip install ".[dev]"
python -m pytest -q --tb=short
```

Install the built wheel in a clean environment and verify the top-level import:

```bash
python -m pip install dist/spherical_array_processing-*-py3-none-any.whl
python -c "import spherical_array_processing as sap; print(sap.__version__)"
```

## Metadata

Confirm that `pyproject.toml`, `README.md`, `CHANGELOG.md`, CI, and the git tag
all agree on the version and supported Python range.

Confirm that public API changes are documented and that experimental or
repository-only modules are clearly labeled as such.

When changing spherical harmonics, radial functions, or beamforming weights,
run the optional external numeric audit if the reference packages are available:

```bash
python scripts/external_numeric_audit.py --repo-root /tmp/sap-cross-repos
```

Record the audit result in `docs/CROSS_REFERENCE_AUDIT.md` whenever it changes
public numerical contracts or closes reference-package API gaps.
