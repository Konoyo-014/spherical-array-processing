"""MATLAB / Octave interop tooling for repository regression jobs.

.. warning::
   This submodule is **not part of the stable public API** of
   ``spherical-array-processing``.  It exists as internal tooling for
   the repository's own cross-tool regression runs; function
   signatures may change without notice.  A one-time
   :class:`FutureWarning` is emitted on first import.
"""

import warnings as _warnings

_warnings.warn(
    "spherical_array_processing.regression is a developer-only "
    "interop/tooling layer and not part of the stable public API. "
    "Its function signatures may change without notice. End-user "
    "code should avoid depending on it; if you must use it for local "
    "tooling, pin an exact spherical-array-processing version.",
    FutureWarning,
    stacklevel=2,
)

from .matlab import (
    detect_matlab,
    detect_octave,
    matlab_available,
    run_matlab_batch,
    run_octave_eval,
)
from .status import CaseStatus

__all__ = [
    "CaseStatus",
    "detect_matlab",
    "detect_octave",
    "matlab_available",
    "run_matlab_batch",
    "run_octave_eval",
]
