"""Magnitude Least Squares (MagLS) binaural rendering filters.

Implements the Schörkhuber–Zotter (2018) alternating-projection
algorithm for producing SH → binaural rendering filters from a set of
measured HRTFs.  Below the cutoff frequency ``f_cut`` the filter is the
ordinary complex least-squares fit; above ``f_cut`` the phase is left
free to minimise the magnitude error, so that spatial cues encoded in
the magnitude spectrum are preserved even when the ambisonic order is
too low to fit the exact complex HRTF phase.

References
----------
.. [1] C. Schörkhuber and F. Zotter, "Binaural rendering of ambisonic
   signals via magnitude least squares", *DAGA 2018*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SphericalGrid


def magls_binaural_filters(
    hrtfs: ArrayLike,
    freqs_hz: ArrayLike,
    hrtf_grid: SphericalGrid,
    max_order: int,
    *,
    basis: str = "real",
    f_cut_hz: float = 1500.0,
    n_iterations: int = 10,
    phase_continuation: bool = True,
    rcond: float = 1e-8,
) -> NDArray[np.complex128]:
    """Compute MagLS SH-to-binaural rendering filters.

    Parameters
    ----------
    hrtfs : array_like, shape (F, G, 2)
        Complex HRTFs at ``F`` frequency bins, ``G`` measurement
        directions, two ears (0 = left, 1 = right).
    freqs_hz : array_like, shape (F,)
        Frequency axis in Hz.
    hrtf_grid : SphericalGrid
        Direction grid with ``G`` measurement points (matches the
        second axis of *hrtfs*).
    max_order : int
        Ambisonic order ``N``.  The resulting filter has
        ``(N+1)²`` rows.
    basis : {"real", "complex"}, optional
        SH basis the ambisonic signal lives in.  ``"real"`` (ACN/SN3D)
        is the standard audio-pipeline choice.
    f_cut_hz : float, optional
        Transition frequency: complex LS below, MagLS above.  Typical
        values are ``1–2 kHz``.
    n_iterations : int, optional
        Number of MagLS iterations per high-frequency bin.  5–10 is
        usually plenty.
    phase_continuation : bool, optional
        If ``True`` (default), initialise the MagLS phase at each
        high-frequency bin from the *complex* LS solution of the bin
        just below ``f_cut``.  Gives smoother group delay than bin-wise
        reseeding.
    rcond : float, optional
        Singular-value cutoff for :func:`numpy.linalg.pinv`.

    Returns
    -------
    filters : ndarray, shape (F, (N+1)², 2), complex128
        MagLS SH rendering filters.  Binaural output at a single frame
        is obtained as
        ``binaural[:, ear] = ambi_signal @ filters[:, :, ear]``.
    """
    h = np.asarray(hrtfs, dtype=np.complex128)
    if h.ndim != 3 or h.shape[-1] != 2:
        raise ValueError(
            "hrtfs must have shape (F, G, 2); got " f"{h.shape}"
        )
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if freqs.size != h.shape[0]:
        raise ValueError("freqs_hz length must equal the frequency axis of hrtfs")
    if h.shape[1] != hrtf_grid.size:
        raise ValueError(
            f"hrtfs' spatial axis ({h.shape[1]}) must match "
            f"hrtf_grid.size ({hrtf_grid.size})"
        )
    if int(max_order) < 0:
        raise ValueError("max_order must be non-negative")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")

    spec = SHBasisSpec(
        max_order=int(max_order),
        basis=basis,
        angle_convention=hrtf_grid.convention,
    )
    y = np.asarray(sh_matrix(spec, hrtf_grid))  # (G, Q)
    pinv_y = np.linalg.pinv(y, rcond=rcond)  # (Q, G)

    n_bins, _, _ = h.shape
    q = y.shape[1]
    out = np.zeros((n_bins, q, 2), dtype=np.complex128)

    # Complex LS at every bin; will be overwritten above cutoff.
    complex_ls = np.einsum("qg,fge->fqe", pinv_y, h)  # (F, Q, 2)
    out[:] = complex_ls

    f_cut = float(f_cut_hz)
    mask_high = freqs >= f_cut
    if not np.any(mask_high):
        return out

    # Phase continuation: for each high-frequency bin, use the phase
    # produced by the previous bin's final LS fit to seed the target
    # spectrum, i.e. build the first target as
    # ``H_target = |H[f]| · exp(j · prev_phase)`` before doing any LS
    # solves.  Without this, the first projected target inside the
    # alternating loop immediately overwrites the seed, and
    # ``phase_continuation=True`` collapses into bit-wise identical
    # behaviour with the ``False`` branch.
    for ear in (0, 1):
        if phase_continuation:
            low_bins = np.where(~mask_high)[0]
            if low_bins.size > 0:
                init_phase = np.angle(y @ complex_ls[low_bins[-1], :, ear])
            else:
                init_phase = np.angle(h[0, :, ear])
        else:
            init_phase = None

        prev_phase = init_phase
        for f_idx in np.nonzero(mask_high)[0]:
            mag = np.abs(h[f_idx, :, ear])
            seed_phase = (
                prev_phase
                if prev_phase is not None
                else np.angle(h[f_idx, :, ear])
            )
            # First LS fit uses the seeded phase so that phase
            # continuity across bins actually takes effect.
            h_target = mag * np.exp(1j * seed_phase)
            w = pinv_y @ h_target
            for _ in range(int(n_iterations) - 1):
                projected = y @ w
                phase = np.angle(projected)
                h_target = mag * np.exp(1j * phase)
                w = pinv_y @ h_target
            out[f_idx, :, ear] = w
            # Propagate the final projected phase to the next bin.
            prev_phase = np.angle(y @ w)

    return out


__all__ = ["magls_binaural_filters"]
