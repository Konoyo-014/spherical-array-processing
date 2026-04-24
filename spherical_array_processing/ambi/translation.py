"""FOA / ambisonic scene translation (virtual-listener displacement).

The standard ambisonic encoding ``c_q(t) = Y_q(û) s(t)`` assumes the
listener sits at the origin of the coordinate frame where the
plane-wave directions ``û`` are defined.  Moving the listener off the
origin by ``r`` changes the arrival time of every plane wave by
``Δ_k = û_k · r / c`` (seconds): waves whose direction aligns with
``r`` arrive **earlier** at the new position, waves from the
opposite hemisphere arrive later.

For a **far-field** scene (the ambisonic assumption) the translation
can be realised by:

1. **Decomposing** the FOA into a dense set of plane-wave components
   via the pseudo-inverse of the SH encoding matrix (PWD).
2. **Advancing / delaying** each component by ``Δ_k`` via a
   frequency-domain phase shift.
3. **Re-encoding** the shifted components back to SH.

Steps 1 and 3 form an identity on the SH coefficient space when the
decomposition grid spans the sphere densely enough, so in the limit
``r → 0`` the output equals the input.  Away from that limit the
method is an approximation: the intermediate PWD reconstructed from
only four FOA channels is a first-order angular point-spread function,
not an exact delta distribution.  The leading-order geometric advance
``û · r / c`` is reproduced for ``|k r| ≪ 1``; for larger
translations the result remains physically plausible but increasingly
smoothed and frequency-dependent.

Currently only first-order (FOA) translation is implemented — higher
orders would need the full Taylor-series / Hankel-based formulation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..array import fibonacci_grid
from ..coords import unit_sph_to_cart
from ..sh import matrix as sh_matrix
from ..types import NormalizationKind, SHBasisSpec
from .format import convert_ambi_normalization


AxisLayout = Literal["channels_first", "channels_last"]


def translate_foa(
    foa_signal: ArrayLike,
    translation_m: ArrayLike,
    *,
    fs: float,
    normalization: NormalizationKind = "orthonormal",
    axis: AxisLayout = "channels_first",
    n_decomposition_dirs: int = 32,
    c: float = 343.0,
) -> NDArray[np.float64]:
    """Translate the virtual listener by *translation_m* in a FOA scene.

    Parameters
    ----------
    foa_signal : array_like, shape ``(4, T)`` or ``(T, 4)``
        First-order ambisonic signal in ACN order.
    translation_m : array_like, shape ``(3,)``
        Cartesian displacement ``(dx, dy, dz)`` of the virtual
        listener, in metres.
    fs : float
        Sampling rate in Hz.
    normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation of *foa_signal*.  Output keeps the same
        normalisation.
    axis : {"channels_first", "channels_last"}, optional
        Layout of *foa_signal*.  Default ``"channels_first"``.
    n_decomposition_dirs : int, optional
        Number of plane-wave directions used to discretise the
        sphere for the intermediate decomposition.  Default ``32``
        (a Fibonacci grid).  Higher values reduce the high-frequency
        discretisation artefact at the cost of a larger FFT matrix
        multiply.
    c : float, optional
        Speed of sound (m/s).  Default ``343``.

    Returns
    -------
    ndarray
        Translated FOA signal with the same shape and normalisation as
        the input.
    """
    r = np.asarray(translation_m, dtype=float).reshape(-1)
    if r.shape != (3,):
        raise ValueError(
            f"translation_m must have shape (3,); got {r.shape}"
        )
    if fs <= 0:
        raise ValueError("fs must be positive")
    if c <= 0:
        raise ValueError("c must be positive")
    if n_decomposition_dirs < 4:
        raise ValueError(
            "n_decomposition_dirs must be ≥ 4 for a valid FOA PWD"
        )

    sig = np.asarray(foa_signal, dtype=float)
    if sig.ndim != 2:
        raise ValueError(
            f"foa_signal must be 2-D (4, T) or (T, 4); got {sig.shape}"
        )
    if axis == "channels_first":
        if sig.shape[0] != 4:
            raise ValueError(
                f"channels_first expects shape (4, T); got {sig.shape}"
            )
        foa_qt = sig
    elif axis == "channels_last":
        if sig.shape[1] != 4:
            raise ValueError(
                f"channels_last expects shape (T, 4); got {sig.shape}"
            )
        foa_qt = sig.T
    else:
        raise ValueError(
            "axis must be 'channels_first' or 'channels_last'; got "
            f"{axis!r}"
        )

    # Convert to orthonormal for internal processing.
    if normalization != "orthonormal":
        foa_qt = convert_ambi_normalization(
            foa_qt, max_order=1,
            from_=normalization, to="orthonormal", axis=0,
        )

    n_samples = foa_qt.shape[1]
    # Plane-wave decomposition grid.
    grid = fibonacci_grid(int(n_decomposition_dirs))
    Y = np.asarray(
        sh_matrix(SHBasisSpec(max_order=1, basis="real"), grid),
    )  # (M, 4)
    # Decoder: p = pinv(Y.T) @ foa.
    pwd_dec = np.linalg.pinv(Y.T)                           # (M, 4)
    # Cartesian unit vectors for each direction.
    u_xyz = unit_sph_to_cart(
        grid.azimuth, grid.angle2, convention=grid.convention,
    )                                                        # (M, 3)
    advance_s = (u_xyz @ r) / float(c)                      # (M,)

    # Zero-pad so the periodic seam of the FFT-domain phase shift sits
    # outside the retained interval for the common short-translation
    # case.  This reduces, but does not eliminate, periodic-extension
    # artefacts for fractional delays.
    nfft = 1 << max(0, (n_samples - 1).bit_length())
    nfft = max(nfft, n_samples * 2)

    foa_f = np.fft.rfft(foa_qt, n=nfft, axis=-1)            # (4, F)
    plane_f = pwd_dec @ foa_f                                # (M, F)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(fs))         # (F,)
    phase = np.exp(
        1j * 2.0 * np.pi * freqs[None, :] * advance_s[:, None],
    )                                                        # (M, F)
    plane_f_shifted = plane_f * phase
    foa_out_f = Y.T @ plane_f_shifted                        # (4, F)
    foa_out = np.fft.irfft(foa_out_f, n=nfft, axis=-1)[:, :n_samples]

    if normalization != "orthonormal":
        foa_out = convert_ambi_normalization(
            foa_out, max_order=1,
            from_="orthonormal", to=normalization, axis=0,
        )
    return foa_out if axis == "channels_first" else foa_out.T


__all__ = ["translate_foa"]
