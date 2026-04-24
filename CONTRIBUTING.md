# Contributing

Thank you for contributing to `spherical-array-processing`.

## Development Setup

Use an isolated virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install optional extras only when working on those areas:

```bash
python -m pip install -e ".[image]"
python -m pip install -e ".[audio]"
```

## Validation

Run the package test suite before submitting changes:

```bash
python -m pytest -q --tb=short
```

Run packaging checks when changing metadata, package layout, examples, or
distribution-related files:

```bash
python -m build
```

The repository contains MATLAB reference and regression tooling under
repository-only directories such as `src/`, `scripts/`, and `artifacts/`.  These
assets support migration and parity work but are not part of the runtime wheel.

## Pull Request Expectations

Keep changes focused and include tests for behavior changes.  Public API
changes should update `README.md` and, when relevant, `CHANGELOG.md`.

If a change touches `spherical_array_processing/repro/`, state whether the goal
is MATLAB semantic compatibility, Pythonic behavior, or a documented expected
difference.
