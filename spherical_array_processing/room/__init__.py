"""Room acoustic simulators for SH-domain testing.

The shoebox image-source method (Allen & Berkley 1979) produces an
ambisonic room impulse response at a chosen listener position, using
rectangular-room geometry and per-wall reflection coefficients.  It
is the standard baseline for reverberant DOA / beamforming
experiments because it is parametric, reproducible, and fast.
"""

from .banded import shoebox_rir_banded
from .fdn import fdn_reverb, fdn_sh_tail
from .metrics import (
    RIRMetrics,
    clarity,
    definition,
    early_decay_time,
    energy_decay_curve,
    reverberation_time,
    rir_metrics,
)
from .reverb import convolve_mono_to_ambi, convolve_sh_to_sh
from .shoebox import (
    ShoeboxRoom,
    shoebox_rir,
    shoebox_sh_rir,
)

__all__ = [
    "RIRMetrics",
    "ShoeboxRoom",
    "clarity",
    "convolve_mono_to_ambi",
    "convolve_sh_to_sh",
    "definition",
    "early_decay_time",
    "energy_decay_curve",
    "fdn_reverb",
    "fdn_sh_tail",
    "reverberation_time",
    "rir_metrics",
    "shoebox_rir",
    "shoebox_rir_banded",
    "shoebox_sh_rir",
]
