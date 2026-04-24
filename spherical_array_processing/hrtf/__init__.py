"""HRTF dataset I/O and preprocessing.

This module provides a lightweight :class:`HRTFDataset` container that
packages the fields required by the :mod:`spherical_array_processing.binaural`
renderers (HRIRs, sample rate, source-direction grid, optional ear
positions) and a SOFA (AES69) ``SimpleFreeFieldHRIR`` reader that
populates it from the widely-used ``.sofa`` format.

The reader is implemented on top of :mod:`h5py` and is therefore an
**optional dependency** — install with ``pip install
'spherical-array-processing[hrtf]'`` or add ``h5py`` to the environment.
The in-memory dataclass has no extra dependencies, so users can still
wire in their own loader (e.g. from MATLAB ``.mat`` files or measured
impulse responses).
"""

from .dataset import HRTFDataset
from .rigid_sphere import rigid_sphere_hrtf
from .sofa import load_sofa, save_sofa

__all__ = ["HRTFDataset", "load_sofa", "rigid_sphere_hrtf", "save_sofa"]
