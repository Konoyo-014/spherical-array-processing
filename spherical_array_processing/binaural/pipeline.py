"""End-to-end SH ambisonic → binaural time-domain rendering.

Convenience wrapper that ties the rest of the package together:

1. Optional head-tracking rotation via
   :func:`~spherical_array_processing.sh.rotate_ambi_over_time`.
2. MagLS / BiMagLS SH→binaural filter construction from an
   :class:`~spherical_array_processing.hrtf.HRTFDataset`.
3. Inverse FFT of the filter to a real-valued FIR.
4. Per-SH-channel convolution into the two ears with
   :func:`scipy.signal.oaconvolve`.

This is the `ambi-to-headphones` pipeline most users reach for first.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import oaconvolve

from ..hrtf import HRTFDataset
from ..sh import rotate_ambi_over_time
from .magls import magls_binaural_filters


def _magls_time_domain_fir(
    filters_f: NDArray[np.complex128], fft_len: int
) -> NDArray[np.float64]:
    """Convert a one-sided MagLS filter ``(F, Q, 2)`` to a zero-phase,
    circularly-shifted FIR of length *fft_len*.

    The inverse RFFT of a one-sided spectrum gives a causal impulse
    response; ``fftshift`` along the time axis centres it so that the
    filter's main energy lies at ``fft_len // 2``, which makes the
    downstream convolution latency easy to reason about.
    """
    impulse = np.fft.irfft(filters_f, n=fft_len, axis=0)
    return np.fft.fftshift(impulse, axes=0).astype(np.float64)


def _detect_sh_axis(signal: np.ndarray, q_expected: int) -> tuple[np.ndarray, bool]:
    """Return ``(channels_first_signal, was_channels_first)``."""
    if signal.ndim != 2:
        raise ValueError(
            f"sh_signal must be 2-D (Q, T) or (T, Q); got {signal.shape}"
        )
    if signal.shape[0] == q_expected and signal.shape[1] != q_expected:
        return signal, True
    if signal.shape[1] == q_expected and signal.shape[0] != q_expected:
        return signal.T, False
    if signal.shape[0] == q_expected and signal.shape[1] == q_expected:
        return signal, True
    raise ValueError(
        f"sh_signal must have one axis of length (max_order+1)² = "
        f"{q_expected}; got shape {signal.shape}"
    )


def ambi_to_binaural_time_domain(
    sh_signal: ArrayLike,
    hrtf_dataset: HRTFDataset,
    *,
    max_order: int,
    basis: Literal["real", "complex"] = "real",
    f_cut_hz: float = 1500.0,
    head_orientations_zyz: ArrayLike | None = None,
    head_tracker_block_samples: int = 480,
    head_tracker_crossfade_samples: int | None = None,
    fft_len: int | None = None,
    n_iterations: int = 10,
    phase_continuation: bool = True,
    **renderer_kwargs: Any,
) -> NDArray[np.float64]:
    """Render an SH ambisonic signal to binaural stereo, end-to-end.

    Parameters
    ----------
    sh_signal : array_like, shape ``(Q, T)`` or ``(T, Q)``
        Ambisonic signal in ACN ordering.  The channel axis is
        auto-detected (it's the axis whose length matches
        ``(max_order+1)²``).
    hrtf_dataset : HRTFDataset
        Time-domain HRTF container (from
        :func:`spherical_array_processing.hrtf.load_sofa` or built
        manually).  ``hrtf_dataset.fs`` must match the ambisonic
        signal's sampling rate.
    max_order : int
        Ambisonic order ``N``.
    basis : {"real", "complex"}, optional
        SH basis the ambisonic signal lives in.  Default ``"real"``.
    f_cut_hz : float, optional
        Transition frequency between complex LS and MagLS phase-free
        fitting.  Default ``1500`` Hz.
    head_orientations_zyz : array_like, shape ``(K, 3)`` or ``(3,)``, optional
        ZYZ Euler keyframes.  When given, the ambi signal is rotated
        over time before rendering via
        :func:`~spherical_array_processing.sh.rotate_ambi_over_time`.
        Pass a single ``(3,)`` vector for a static rotation.
    head_tracker_block_samples : int, optional
        Forwarded to :func:`rotate_ambi_over_time`.  Default ``480``.
    head_tracker_crossfade_samples : int, optional
        Forwarded to :func:`rotate_ambi_over_time`.
    fft_len : int, optional
        FFT length used to derive the frequency-domain HRTF + filter.
        Defaults to the HRIR length (no zero padding).
    n_iterations, phase_continuation : forwarded to the MagLS solver.
    **renderer_kwargs
        Additional keyword arguments forwarded to
        :func:`magls_binaural_filters` (``rcond``, etc.).

    Notes
    -----
    This wrapper uses MagLS exclusively.  BiMagLS is available as
    :func:`bimagls_binaural_filters` but requires per-direction ITD
    reattachment at render time, which depends on having a source DOA
    estimate — out of scope for a single-call pipeline.

    Returns
    -------
    ndarray, shape ``(2, T_out)``
        Time-domain binaural signal.  ``T_out = T + fft_len − 1`` (full
        convolution length).  The filter is centred via ``fftshift`` so
        the main energy lives at sample ``fft_len // 2``; downstream
        code that needs the ``T`` original samples can slice
        ``binaural[:, fft_len // 2 : fft_len // 2 + T]``.
    """
    q_expected = (int(max_order) + 1) ** 2
    sig = np.asarray(sh_signal)
    sig_cf, _ = _detect_sh_axis(sig, q_expected)

    # ---- 1. Head tracking.
    if head_orientations_zyz is not None:
        sig_cf = rotate_ambi_over_time(
            sig_cf,
            head_orientations_zyz,
            max_order=int(max_order),
            basis=basis,
            block_samples=int(head_tracker_block_samples),
            crossfade_samples=head_tracker_crossfade_samples,
        )

    # ---- 2. Frequency-domain SH → binaural filter.
    fft_len_eff = int(hrtf_dataset.n_taps if fft_len is None else fft_len)
    if fft_len_eff <= 0:
        raise ValueError("fft_len must be positive")
    freqs, hrtfs = hrtf_dataset.to_frequency_domain(fft_len_eff)

    filters_f = magls_binaural_filters(
        hrtfs,
        freqs,
        hrtf_dataset.source_grid,
        int(max_order),
        basis=basis,
        f_cut_hz=float(f_cut_hz),
        n_iterations=int(n_iterations),
        phase_continuation=bool(phase_continuation),
        **renderer_kwargs,
    )

    # ---- 3. IFFT the filter to a real FIR centered at fft_len // 2.
    fir = _magls_time_domain_fir(filters_f, fft_len_eff)
    # fir shape: (fft_len_eff, Q, 2).

    # ---- 4. Per-channel convolution + sum per ear.
    #
    # oaconvolve(sh[q], fir[:, q, e]) has shape (T + fft_len_eff - 1,).
    # Sum over q for each ear.
    q_actual = sig_cf.shape[0]
    if q_actual != q_expected:
        raise ValueError(
            f"signal has {q_actual} SH channels but max_order={max_order} "
            f"requires {q_expected}"
        )
    t = sig_cf.shape[1]
    out_len = t + fft_len_eff - 1
    binaural = np.zeros((2, out_len), dtype=np.float64)
    # Convolve channel-by-channel (Q is small: 4–64 for typical use).
    for ear in (0, 1):
        # Stack convolution as FFT-based batched convolution.
        # oaconvolve with axes=-1 handles the time axis.
        conv = oaconvolve(
            sig_cf,
            fir[:, :, ear].T,  # (Q, fft_len_eff)
            mode="full", axes=-1,
        )
        # conv shape: (Q, out_len).  Sum over Q.
        binaural[ear] = np.real(conv.sum(axis=0))
    return binaural


__all__ = ["ambi_to_binaural_time_domain"]
