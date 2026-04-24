"""Lightweight HRTF container shared by the SOFA reader and the
binaural renderers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..types import SphericalGrid


@dataclass(frozen=True)
class HRTFDataset:
    """Time-domain HRTFs + source grid + (optional) ear positions.

    Attributes
    ----------
    hrirs : ndarray, shape (M, 2, N)
        Head-related impulse responses.  ``M`` is the number of
        measured directions, 2 is left/right, and ``N`` is the IR
        length in samples.
    fs : float
        Sampling rate in Hz.
    source_grid : SphericalGrid
        Measurement directions (``size == M``).  Convention matches
        whatever the source dataset used; the binaural renderers only
        care about the stored convention being self-consistent.
    ear_positions_m : ndarray or None, shape (2, 3), optional
        Cartesian ear positions in metres, if the SOFA file supplies
        ``ReceiverPosition`` or the user pre-populates them.  Required
        by :func:`~spherical_array_processing.binaural.bimagls_binaural_filters`.
    data_delay_samples : ndarray or None, shape (2,) or (M, 2), optional
        Per-ear (and optionally per-direction) pre-roll delays in
        samples, mirroring the SOFA ``Data.Delay`` block.  ``None``
        means "no pre-roll" (the common case).  Length-``(2,)`` values
        broadcast to every direction.
    metadata : dict
        Free-form metadata carried forward from the SOFA attributes
        (``Conventions``, ``DatabaseName``, etc.) — useful for
        provenance but not consumed by any downstream code.
    """

    hrirs: NDArray[np.float64]
    fs: float
    source_grid: SphericalGrid
    ear_positions_m: NDArray[np.float64] | None = None
    data_delay_samples: NDArray[np.float64] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_directions(self) -> int:
        return int(self.hrirs.shape[0])

    @property
    def n_taps(self) -> int:
        return int(self.hrirs.shape[-1])

    def to_frequency_domain(
        self, fft_len: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Return ``(freqs_hz, hrtfs)`` with ``hrtfs`` shaped ``(F, M, 2)``.

        Convenience wrapper to match the signature expected by
        :func:`spherical_array_processing.binaural.magls_binaural_filters`
        / :func:`~spherical_array_processing.binaural.bimagls_binaural_filters`:

        * ``freqs_hz`` is the non-negative frequency axis derived from
          ``fs`` and the (padded) FFT length.
        * ``hrtfs[f, g, ear] = FFT(hrirs[g, ear])[f]``.

        Parameters
        ----------
        fft_len : int or None, optional
            FFT length.  When ``None`` (default) uses the stored IR
            length, i.e. no zero-padding.

        Returns
        -------
        freqs_hz : ndarray, shape (F,)
            Frequency bin centres in Hz; ``F = fft_len // 2 + 1``.
        hrtfs : ndarray, shape (F, M, 2), complex128
            One-sided FFT of the HRIRs.
        """
        n = int(self.n_taps if fft_len is None else fft_len)
        if n <= 0:
            raise ValueError("fft_len must be positive")
        freqs = np.fft.rfftfreq(n, d=1.0 / float(self.fs)).astype(float)
        hrtfs = np.fft.rfft(self.hrirs, n=n, axis=-1).transpose(2, 0, 1)
        return freqs, np.asarray(hrtfs, dtype=np.complex128)


__all__ = ["HRTFDataset"]
