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
from .statistical import (
    ShoeboxAcousticStats,
    air_absorption_attenuation_iso9613,
    air_absorption_coefficient_iso9613,
    arau_puchades_rt60,
    classify_room_modes,
    critical_distance,
    critical_distance_from_rt60,
    equivalent_absorption_area,
    eyring_rt60,
    mean_absorption,
    millington_sette_rt60,
    rectangular_room_modes,
    room_constant,
    sabine_rt60,
    schroeder_frequency,
    shoebox_acoustic_stats,
    shoebox_axis_surface_areas,
    shoebox_surface_areas,
    shoebox_volume,
)

__all__ = [
    "RIRMetrics",
    "ShoeboxAcousticStats",
    "ShoeboxRoom",
    "air_absorption_attenuation_iso9613",
    "air_absorption_coefficient_iso9613",
    "arau_puchades_rt60",
    "classify_room_modes",
    "clarity",
    "convolve_mono_to_ambi",
    "convolve_sh_to_sh",
    "critical_distance",
    "critical_distance_from_rt60",
    "definition",
    "early_decay_time",
    "energy_decay_curve",
    "equivalent_absorption_area",
    "eyring_rt60",
    "fdn_reverb",
    "fdn_sh_tail",
    "mean_absorption",
    "millington_sette_rt60",
    "rectangular_room_modes",
    "reverberation_time",
    "rir_metrics",
    "room_constant",
    "sabine_rt60",
    "schroeder_frequency",
    "shoebox_acoustic_stats",
    "shoebox_axis_surface_areas",
    "shoebox_rir",
    "shoebox_rir_banded",
    "shoebox_sh_rir",
    "shoebox_surface_areas",
    "shoebox_volume",
]
