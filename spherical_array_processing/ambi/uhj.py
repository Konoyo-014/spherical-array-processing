"""UHJ stereo ↔ ambisonic B-format conversion (Gerzon 1985).

The 2-channel **UHJ** format encodes the horizontal information of a
B-format signal (W, X, Y) into a stereo-compatible pair ``(L_T, R_T)``
via Hilbert-shifted linear combinations.  Playback through ordinary
stereo systems reproduces the source direction cues; dedicated UHJ
decoders recover an approximate B-format for ambisonic rendering.

Encode (Gerzon):

.. math::

    S &= 0.9396926\\,W + 0.1855740\\,X \\\\
    D &= j(-0.3420201\\,W + 0.5098604\\,X) + 0.6554516\\,Y \\\\
    L_T &= (S + D) / 2 \\\\
    R_T &= (S - D) / 2

where ``j`` is the Hilbert transform (``+π/2`` phase shift for
``f > 0``).  The ``W`` used here is the Furse-Malham scaled W
(``W_FuMa = W_SN3D / √2``); the module converts transparently between
the package's supported normalisations so the caller can simply pass
their SN3D, N3D or orthonormal FOA and get standards-compliant UHJ.

Decode (approximate; UHJ-2 carries no vertical information, so the
``Z`` channel is always zero):

.. math::

    W &= 0.982\\,S + 0.164\\,j\\,D \\\\
    X &= 0.419\\,S - 0.828\\,j\\,D \\\\
    Y &= 0.763\\,D + 0.385\\,j\\,S

with ``S = (L_T + R_T)/2`` and ``D = (L_T - R_T)/2``.  Coefficients
are the classical Gerzon decoder choice.  For full 3-channel UHJ (with
a dedicated height carrier) use a separately recorded ``T`` channel;
this module implements the 2-channel variant only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import hilbert

from ..types import NormalizationKind
from .format import convert_ambi_normalization


AxisLayout = Literal["channels_first", "channels_last"]
HilbertMethod = Literal["fft", "fir"]


def _hilbert_fir_taps(numtaps: int, beta: float = 8.6) -> NDArray[np.float64]:
    """Design a Kaiser-windowed linear-phase FIR Hilbert transformer.

    Produces the impulse response ``h[n] = (1 - cos(πn')) / (π n')``
    (with ``n' = n - (N-1)/2``) multiplied by a Kaiser window.  ``N``
    is forced odd so the filter is type III and exactly band-limited
    away from DC and Nyquist.  Group delay is ``(N-1)/2`` samples.
    """
    n = int(numtaps)
    if n < 3:
        raise ValueError("numtaps must be at least 3")
    if n % 2 == 0:
        n += 1
    idx = np.arange(n) - (n - 1) // 2
    h = np.zeros(n, dtype=float)
    mask = idx != 0
    h[mask] = (1.0 - np.cos(np.pi * idx[mask])) / (np.pi * idx[mask])
    return np.asarray(h * np.kaiser(n, beta), dtype=np.float64)


def _apply_hilbert(
    signal: NDArray[np.float64],
    *,
    method: HilbertMethod,
    fir_taps: int,
) -> NDArray[np.float64]:
    """Return ``H{signal}`` aligned in time with the input.

    For ``method="fft"`` the result is zero-phase (whole-block FFT);
    for ``method="fir"`` it is produced by a linear-phase Kaiser-
    windowed FIR, trimmed from the full convolution so that the output
    always has the same length as the input and is centred by the
    group delay ``(fir_taps-1)/2``.  In both cases the output is
    time-aligned with the input at the interior — edge samples within
    ``(fir_taps-1)/2`` of the boundaries leak to zero in the FIR case.
    """
    if method == "fft":
        return np.asarray(np.imag(hilbert(signal)), dtype=np.float64)
    if method == "fir":
        fir = _hilbert_fir_taps(fir_taps)
        full = np.convolve(signal, fir, mode="full")
        start = (fir.size - 1) // 2
        stop = start + signal.size
        return np.asarray(full[start:stop], dtype=np.float64)
    raise ValueError(
        f"hilbert_method must be 'fft' or 'fir', got {method!r}"
    )


def _prepare_foa(
    foa_signal: ArrayLike, axis: AxisLayout,
) -> tuple[NDArray[np.float64], AxisLayout]:
    sig = np.asarray(foa_signal, dtype=float)
    if sig.ndim != 2:
        raise ValueError(
            f"foa_signal must be 2-D (4, T) or (T, 4); got shape {sig.shape}"
        )
    if axis == "channels_first":
        if sig.shape[0] != 4:
            raise ValueError(
                f"channels_first expects shape (4, T); got {sig.shape}"
            )
        qt = sig
    elif axis == "channels_last":
        if sig.shape[1] != 4:
            raise ValueError(
                f"channels_last expects shape (T, 4); got {sig.shape}"
            )
        qt = sig.T
    else:
        raise ValueError(
            "axis must be 'channels_first' or 'channels_last'; got "
            f"{axis!r}"
        )
    return qt, axis


def _to_fuma(
    qt: NDArray[np.float64], normalization: NormalizationKind,
) -> NDArray[np.float64]:
    """Convert FOA from any input convention to Furse-Malham ordering /
    normalisation used by the Gerzon UHJ coefficients."""
    # Package is ACN-ordered; UHJ needs (W, X, Y, Z) = ACN (0, 3, 1, 2).
    # First convert to SN3D in ACN ordering, then extract W/X/Y/Z and
    # rescale W by 1/√2 to match Furse-Malham.
    if normalization != "sn3d":
        qt = convert_ambi_normalization(
            qt, max_order=1,
            from_=normalization, to="sn3d", axis=0,
        )
    w = qt[0] / np.sqrt(2.0)   # SN3D W → FuMa W.
    x = qt[3]                   # ACN n=1 m=+1.
    y = qt[1]                   # ACN n=1 m=-1.
    z = qt[2]                   # ACN n=1 m=0.
    return np.stack([w, x, y, z], axis=0)


def _from_fuma(
    wxyz: NDArray[np.float64], normalization: NormalizationKind,
) -> NDArray[np.float64]:
    """Convert a Furse-Malham (W, X, Y, Z) layout back to ACN-*normalization*."""
    w, x, y, z = wxyz
    acn = np.stack([w * np.sqrt(2.0), y, z, x], axis=0)  # SN3D ACN.
    if normalization != "sn3d":
        acn = convert_ambi_normalization(
            acn, max_order=1,
            from_="sn3d", to=normalization, axis=0,
        )
    return acn


def uhj_encode(
    foa_signal: ArrayLike,
    *,
    normalization: NormalizationKind = "orthonormal",
    axis: AxisLayout = "channels_first",
    hilbert_method: HilbertMethod = "fft",
    fir_taps: int = 513,
) -> NDArray[np.float64]:
    """Encode a first-order ambisonic signal into 2-channel UHJ stereo.

    Parameters
    ----------
    foa_signal : array_like, shape ``(4, T)`` or ``(T, 4)``
        FOA signal in ACN order ``(W, Y, Z, X)``.  Any supported
        normalisation.
    normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation of *foa_signal*.  Default ``"orthonormal"``
        (package internal).
    axis : {"channels_first", "channels_last"}, optional
        Layout of *foa_signal*.  Default ``"channels_first"``.
    hilbert_method : {"fft", "fir"}, optional
        Quadrature-shift implementation.  ``"fft"`` (default) uses
        :func:`scipy.signal.hilbert` — zero-phase, best accuracy, but
        whole-block semantics (not suitable for streaming).  ``"fir"``
        uses a linear-phase Kaiser-windowed FIR of length *fir_taps*,
        which has a fixed group delay of ``(fir_taps-1)/2`` samples
        and is therefore safe to apply block-by-block as long as the
        delay is tracked and state is maintained across blocks.
    fir_taps : int, optional
        FIR length used when ``hilbert_method="fir"``.  Default
        ``513`` (≈ 10 ms at 48 kHz).  Forced odd internally.

    Returns
    -------
    ndarray, shape ``(2, T)`` if ``axis == "channels_first"`` else ``(T, 2)``.
        UHJ-2 stereo ``(L_T, R_T)``.
    """
    qt, _ = _prepare_foa(foa_signal, axis)
    w, x, y, z = _to_fuma(qt, normalization)

    # Real (non-Hilbert) parts.
    real_sum = 0.9396926 * w + 0.1855740 * x
    real_diff = 0.6554516 * y
    # Hilbert-shifted part: apply H{·} to (aW + bX).
    hilbert_src = -0.3420201 * w + 0.5098604 * x
    hilbert_part = _apply_hilbert(
        hilbert_src, method=hilbert_method, fir_taps=fir_taps,
    )

    l_t = 0.5 * (real_sum + hilbert_part + real_diff)
    r_t = 0.5 * (real_sum - hilbert_part - real_diff)
    stereo = np.stack([l_t, r_t], axis=0)
    return stereo if axis == "channels_first" else stereo.T


def uhj_decode(
    stereo_signal: ArrayLike,
    *,
    normalization: NormalizationKind = "orthonormal",
    axis: AxisLayout = "channels_first",
    hilbert_method: HilbertMethod = "fft",
    fir_taps: int = 513,
) -> NDArray[np.float64]:
    """Decode a 2-channel UHJ stereo signal to an approximate FOA.

    The ``Z`` channel is returned as zeros because UHJ-2 carries no
    vertical information.  Use the 3- or 4-channel UHJ variants (not
    implemented here) for height preservation.

    Parameters
    ----------
    stereo_signal : array_like, shape ``(2, T)`` or ``(T, 2)``
        UHJ-2 stereo pair ``(L_T, R_T)``.
    normalization : {"orthonormal", "n3d", "sn3d"}, optional
        Normalisation to use for the returned FOA.  Default
        ``"orthonormal"``.
    axis : {"channels_first", "channels_last"}, optional
        Layout convention; applies to both input and output.

    Returns
    -------
    ndarray, shape ``(4, T)`` or ``(T, 4)``
        Approximate FOA in ACN order, with ``Z = 0``.
    """
    sig = np.asarray(stereo_signal, dtype=float)
    if sig.ndim != 2:
        raise ValueError(
            f"stereo_signal must be 2-D; got shape {sig.shape}"
        )
    if axis == "channels_first":
        if sig.shape[0] != 2:
            raise ValueError(
                f"channels_first expects shape (2, T); got {sig.shape}"
            )
        l_t, r_t = sig[0], sig[1]
    elif axis == "channels_last":
        if sig.shape[1] != 2:
            raise ValueError(
                f"channels_last expects shape (T, 2); got {sig.shape}"
            )
        l_t, r_t = sig[:, 0], sig[:, 1]
    else:
        raise ValueError(
            "axis must be 'channels_first' or 'channels_last'; got "
            f"{axis!r}"
        )
    s = 0.5 * (l_t + r_t)
    d = 0.5 * (l_t - r_t)
    s_hilbert = _apply_hilbert(s, method=hilbert_method, fir_taps=fir_taps)
    d_hilbert = _apply_hilbert(d, method=hilbert_method, fir_taps=fir_taps)
    w_fuma = 0.982 * s + 0.164 * d_hilbert
    x_fuma = 0.419 * s - 0.828 * d_hilbert
    y_fuma = 0.763 * d + 0.385 * s_hilbert
    z_fuma = np.zeros_like(w_fuma)
    acn = _from_fuma(
        np.stack([w_fuma, x_fuma, y_fuma, z_fuma], axis=0),
        normalization,
    )
    return acn if axis == "channels_first" else acn.T


__all__ = ["uhj_decode", "uhj_encode"]
