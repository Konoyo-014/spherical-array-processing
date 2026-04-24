"""Research prototypes — provisional APIs.

.. warning::
   This submodule is **not part of the stable public API** of
   ``spherical-array-processing``.  It contains experimental
   estimators (currently the stereo → incomplete-FOA family) whose
   behaviour and signatures are provisional and may change without
   notice.  A one-time :class:`FutureWarning` is emitted on first
   import.
"""

import warnings as _warnings

_warnings.warn(
    "spherical_array_processing.experimental is a research-prototype "
    "area and not part of the stable public API. Its function "
    "signatures and algorithmic behaviour may change without notice. "
    "Prefer documented stable APIs where available; if you evaluate "
    "this prototype, pin an exact spherical-array-processing version.",
    FutureWarning,
    stacklevel=2,
)

from .foa_from_stereo import FOAEstimate, StereoFOAConfig, estimate_incomplete_foa_from_stereo
from .foa_from_stereo_dl import StereoFOADLConfig, estimate_incomplete_foa_from_stereo_dl

__all__ = [
    "FOAEstimate",
    "StereoFOAConfig",
    "StereoFOADLConfig",
    "estimate_incomplete_foa_from_stereo",
    "estimate_incomplete_foa_from_stereo_dl",
]
