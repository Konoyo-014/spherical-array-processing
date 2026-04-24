"""Near-field compensation filters for ambisonic rendering.

Standard higher-order ambisonics (HOA) encodes **plane waves** only:
the ``Y_n^m(û)`` expansion assumes an infinitely distant source.  To
add a *near-field distance cue* to a plane-wave-encoded ambi signal,
this module provides the per-order "near-field emphasis" filter

.. math::

    F_n(\\omega; d, R) = \\frac{h_n^{(2)}(k\\,d)}{h_n^{(2)}(k\\,R)}

where ``d`` is the intended source distance and ``R`` is a
**stabilisation reference** distance — typically the loudspeaker
array radius.  Each of the two Hankel functions diverges at
``k → 0`` for ``n ≥ 1``, but their ratio stays bounded as long as
both distances are positive, so ``F_n`` is a well-defined IIR filter
that can be applied per-SH-channel at render time.

Physical interpretation (DC limit ``F_n(0) = (R/d)^{n+1}``):

* If ``d = R`` the filter is identically ``1`` — no compensation
  needed when the intended source distance matches the reproduction
  array radius.
* If ``d < R`` the filter **boosts** near-field low frequencies
  (source closer than speakers ⇒ stronger bass cue).
* If ``d > R`` the filter **attenuates** low frequencies; in the
  ``d → ∞`` plane-wave limit the DC gain collapses to ``0`` for
  every order, with the monopole term scaling as ``R/d`` and the
  higher orders collapsing faster.

Note that the filter's high-frequency behaviour approaches
``(R/d) · exp(-i k (d − R))`` up to order-dependent ``O(1/k)``
corrections from the spherical-wave asymptotic series.  The
high-frequency magnitude therefore tends to ``R/d`` rather than to
unity.  Callers who want a purely magnitude-based distance cue can
take ``|F_n|`` or pre-compose with a linear-phase equaliser.

References
----------
.. [1] J. Daniel, "Spatial sound encoding including near field effect:
   introducing distance coding filters and a viable, new ambisonic
   format", *Proc. AES 23rd Int. Conf.*, 2003.
.. [2] F. Zotter and M. Frank, *Ambisonics — A Practical 3D Audio
   Theory for Recording, Studio Production, Sound Reinforcement, and
   Virtual Reality*, Springer, 2019, Ch. 4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..acoustics.radial import besselhs2


def nfc_hoa_distance_filter(
    max_order: int,
    freqs_hz: ArrayLike,
    source_distance_m: float,
    reference_distance_m: float,
    *,
    c: float = 343.0,
    repeat_per_order: bool = True,
) -> NDArray[np.complex128]:
    """NFC-HOA per-order distance-compensation filter.

    Parameters
    ----------
    max_order : int
        Ambisonic order ``N``.  Filter length along the SH axis is
        ``(N+1)²`` when *repeat_per_order* is true, else ``N+1``.
    freqs_hz : array_like, shape ``(F,)``
        Frequency bins in Hz.  ``0`` Hz is handled as the DC limit
        ``F_n(0) = (R/d)^{n+1}`` so the returned filter is bounded
        everywhere.  For ``d = R`` this collapses to ``1`` for every
        order.
    source_distance_m : float
        Intended source distance ``d_src``.  ``> 0``.
    reference_distance_m : float
        Stabilisation reference distance ``R_ref`` (typically the
        loudspeaker-array radius).  ``> 0``.
    c : float, optional
        Speed of sound.  Default ``343`` m/s.
    repeat_per_order : bool, optional
        If ``True`` (default), expand the per-order filter to ACN
        ordering with ``(N+1)²`` columns; otherwise return the compact
        ``(N+1)``-column form.

    Returns
    -------
    ndarray, shape ``(F, C)``, complex
        Per-frequency complex filter.  ``C = (N+1)²`` when
        *repeat_per_order* is ``True``, else ``C = N+1``.  Apply to an
        SH-domain STFT with
        :func:`spherical_array_processing.encoding.apply_radial_equalizer`.
    """
    if source_distance_m <= 0.0:
        raise ValueError("source_distance_m must be positive")
    if reference_distance_m <= 0.0:
        raise ValueError("reference_distance_m must be positive")
    if c <= 0.0:
        raise ValueError("c must be positive")
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    order = int(max_order)
    if order < 0:
        raise ValueError("max_order must be non-negative")

    k = 2.0 * np.pi * f / float(c)
    kd = k * float(source_distance_m)
    kR = k * float(reference_distance_m)

    # Per-order h_n^(2) evaluated on the frequency axis.
    # Shape: (F, N+1).
    h_d = np.stack([besselhs2(n, kd) for n in range(order + 1)], axis=-1)
    h_R = np.stack([besselhs2(n, kR) for n in range(order + 1)], axis=-1)

    # Low-frequency limit: h_n^(2)(x) ~ i (2n-1)!! / x^(n+1) (leading
    # order at small x).  Ratio h_n^(2)(kd) / h_n^(2)(kR) → (R/d)^(n+1)
    # at k → 0.  Use that closed form at the exact-DC bin to avoid the
    # ∞/∞ from evaluating the Hankel functions at zero argument.
    dc_mask = (f == 0.0)
    if np.any(dc_mask):
        n_vec = np.arange(order + 1)
        dc_ratio = (
            float(reference_distance_m) / float(source_distance_m)
        ) ** (n_vec + 1)
        for n in range(order + 1):
            h_d[dc_mask, n] = dc_ratio[n]
            h_R[dc_mask, n] = 1.0

    filt = h_d / h_R  # (F, N+1)

    if repeat_per_order:
        n_vec = np.arange(order + 1)
        # Repeat filter[:, n] exactly (2n+1) times → (N+1)² columns.
        counts = 2 * n_vec + 1
        expanded = np.repeat(filt, counts, axis=1)
        return np.asarray(expanded, dtype=np.complex128)
    return np.asarray(filt, dtype=np.complex128)


__all__ = ["nfc_hoa_distance_filter"]
