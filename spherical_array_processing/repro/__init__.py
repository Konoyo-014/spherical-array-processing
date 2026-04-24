"""Reproduction workflows for source libraries.

.. warning::
   This submodule is **not part of the stable public API** of
   ``spherical-array-processing``.  It exists to reproduce published
   MATLAB figures and numerical baselines; function signatures and
   behaviour may change without notice.  A one-time
   :class:`FutureWarning` is emitted when you first import this
   subpackage — it is intended to make accidental downstream
   dependencies visible.
"""

import warnings as _warnings

_warnings.warn(
    "spherical_array_processing.repro is a developer-only "
    "reproduction layer and not part of the stable public API. "
    "Its function signatures may change without notice. Prefer the "
    "documented top-level public submodules; if you must depend on "
    "this layer, pin an exact spherical-array-processing version.",
    FutureWarning,
    stacklevel=2,
)

from . import array_response_simulator, politis, rafaely, sht

__all__ = ["politis", "rafaely", "sht", "array_response_simulator"]
