# Getting Started

This guide assumes you know Python and NumPy, but you do not need to know
spherical microphone arrays before starting. The package works with directions
on a sphere, spherical-harmonic coefficients, and frequency-domain array data.
The quickest way to build confidence is to run the tutorial examples first,
then read the concepts page when a term becomes unfamiliar.

## Install The Public Release

Install the current public release from PyPI:

```bash
pip install spherical-array-processing
```

## Install From A Checkout

Create a virtual environment from the open-source repository root. The repository
root is the directory that contains `pyproject.toml`, `spherical_array_processing/`,
`examples/`, and `tests/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

After installation, verify that the local checkout is importable.

```bash
python - <<'PY'
import spherical_array_processing as sap
print(sap.__version__)
print(sap.array.fibonacci_grid(8).size)
PY
```

## Run The First Example

The basic example builds a spherical grid, computes a spherical-harmonic matrix,
projects samples into SH coefficients, and evaluates a fixed beamformer.

```bash
python examples/core/basic_usage.py
```

The output should include `B(0°)=1.0000`. That line confirms that the example is
using the local checkout and that the fixed beamformer weights are normalized to
unit front gain. If the value is different, you are probably importing an older
installed copy of the package. Running from the repository root fixes this for
the bundled examples because they bootstrap the local checkout before import.

## First Real Workflow

The most useful mental model is a short data path. Start with directions on the
sphere, build an SH basis matrix, transform sampled data into coefficients, then
use those coefficients for beamforming or DOA estimation.

```python
import numpy as np
import spherical_array_processing as sap

order = 3
grid = sap.array.fibonacci_grid(1000)
spec = sap.SHBasisSpec(max_order=order, basis="complex", angle_convention="az_colat")

Y = sap.sh.matrix(spec, grid)
samples = np.cos(grid.angle2)
coeffs = sap.sh.direct_sht(samples, Y, grid)
reconstructed = sap.sh.inverse_sht(coeffs, Y).real

weights = sap.beamforming.beam_weights_hypercardioid(order)
front = sap.beamforming.axisymmetric_pattern(0.0, weights)
print(coeffs.shape, reconstructed.shape, front)
```

The important dimensions are `M` directions and `(N+1)^2` SH channels. With
`order = 3`, there are `16` SH coefficients. Increasing the order gives the model
more spatial detail, but it also needs more sensors, more stable sampling, and
more careful modal equalization.

## Tutorial Scripts

The tutorial scripts are designed to be run directly from a clean checkout.
`examples/tutorials/01_sht_and_beamforming.py` demonstrates an SH transform and
a normalized fixed beam pattern. `examples/tutorials/02_simulated_doa_pipeline.py`
creates a synthetic source covariance and recovers its direction with PWD and
MUSIC. `examples/tutorials/03_modal_equalization_pipeline.py` shows how radial
modal coefficients distort SH-domain pressure data and how regularized modal
equalization recovers plane-wave SH amplitudes.

```bash
python examples/tutorials/01_sht_and_beamforming.py
python examples/tutorials/02_simulated_doa_pipeline.py
python examples/tutorials/03_modal_equalization_pipeline.py
```

If you prefer notebooks, open `examples/notebooks/getting_started.ipynb`. It
uses the same path as the first tutorial script, but breaks the workflow into
small cells.

## What To Read Next

Read `docs/concepts.md` when the words **ACN**, **colatitude**, **modal
coefficient**, **SHT**, or **unit front gain** are unclear. Read
`examples/core/beamforming.py` when you want to compare fixed beamformer
patterns. Read `examples/core/doa_estimation.py` when you want a slightly larger
simulation pipeline.
