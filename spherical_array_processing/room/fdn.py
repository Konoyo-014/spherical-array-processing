"""Feedback-Delay-Network (FDN) diffuse reverberator (Jot 1992).

An FDN produces a dense diffuse tail via ``N`` recirculating delay
lines with a lossless mixing matrix.  Per-line gains ``α_i`` control
the decay rate (``RT60``); a unitary mixing matrix ``M`` preserves
energy while scattering it across taps so the tail is dense within a
few circulations.

The classical Jot formulation, which this module implements, is:

.. math::

    y_i[n] &= d_i[n - L_i] \\\\
    \\mathbf{y}[n] &= M\\,(\\alpha \\odot \\mathbf{y}[n-L]) + \\mathbf{b}\\,x[n]

where ``α_i = 10^{-3 L_i / (\\mathrm{RT60}\\,f_s)}`` tunes the per-line
gain to meet the target ``RT60`` at the delay-line length ``L_i``.

Typical defaults:
* ``N = 8`` delay lines
* ``L_i`` = mutually-coprime odd samples spread between ~1000 and
  ~4000 samples (≈ 20 – 80 ms at 48 kHz)
* Hadamard mixing matrix, which is orthogonal and fast to apply

Two public entry points:

* :func:`fdn_reverb` — monaural reverb on a monaural dry signal.
* :func:`fdn_sh_tail` — ambisonic diffuse tail built by scattering the
  ``N`` FDN taps over random directions on the sphere and encoding
  each with plane-wave SH coefficients.  Use as a late-reverb
  companion to :func:`shoebox_sh_rir` (which handles early
  reflections).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SphericalGrid


_DEFAULT_DELAYS = (1153, 1327, 1523, 1787, 2003, 2309, 2593, 2851)


def _hadamard(n: int) -> NDArray[np.float64]:
    """Orthonormal Hadamard matrix of size ``n`` — ``n`` must be a
    power of two.  Falls back to a random orthogonal matrix otherwise.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    # Power of two check.
    if (n & (n - 1)) == 0:
        h = np.array([[1.0]], dtype=float)
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        return h / np.sqrt(n)
    # Non-power-of-two: orthonormalise a deterministic random matrix.
    rng = np.random.default_rng(0xA11B15)
    a = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(a)
    return q


def _resolve_mixing_matrix(
    mixing_matrix: ArrayLike | None,
    n_lines: int,
    *,
    check_orthogonality: bool,
    atol: float = 1e-6,
) -> NDArray[np.float64]:
    """Return an ``(N, N)`` FDN mixing matrix, optionally enforcing
    orthogonality.

    Orthogonality is what makes the FDN's feedback loop **lossless**
    before the per-line decay gains are applied, which in turn is what
    lets ``α_i = 10^{-3 L_i/(RT60·f_s)}`` map cleanly to the target
    reverberation time.  Passing a non-orthogonal matrix silently
    breaks that relationship, so the default now rejects it.  Users
    who deliberately want a non-unitary feedback matrix (for
    experiments, or for intentionally coloured decays) can set
    *check_orthogonality* to ``False``.
    """
    if mixing_matrix is None:
        return _hadamard(n_lines)
    mixing = np.asarray(mixing_matrix, dtype=float)
    if mixing.shape != (n_lines, n_lines):
        raise ValueError(
            f"mixing_matrix must be {n_lines}×{n_lines}; "
            f"got {mixing.shape}"
        )
    if check_orthogonality:
        gram = mixing @ mixing.T
        eye = np.eye(n_lines)
        deviation = float(np.max(np.abs(gram - eye)))
        if deviation > atol:
            raise ValueError(
                "mixing_matrix is not orthogonal — ``M·Mᵀ`` deviates "
                f"from I by {deviation:.2e} (tolerance {atol:.1e}).  "
                "Non-orthogonal feedback breaks the FDN's energy-"
                "preservation contract and the RT60 calibration.  "
                "Pass `check_orthogonality=False` if you really want "
                "this."
            )
    return mixing


def _fdn_run(
    dry_signal: NDArray[np.float64],
    *,
    delays: NDArray[np.int64],
    decay_gains: NDArray[np.float64],
    mixing: NDArray[np.float64],
    input_gains: NDArray[np.float64],
    output_len: int,
) -> NDArray[np.float64]:
    """Stateful FDN time-domain loop, returning per-line outputs.

    Returns
    -------
    ndarray, shape ``(N, output_len)``
        Time-domain output of every delay line ``y_i[n]``.  The caller
        decides how to mix them (scalar sum for monaural, direction-
        weighted sum per SH channel for ambisonic).
    """
    n_lines = delays.size
    max_delay = int(np.max(delays))
    # One ring-buffer per line.  A single (N, max_delay) buffer works
    # because we advance all lines together.
    buffers = np.zeros((n_lines, max_delay), dtype=float)
    write_idx = 0
    out = np.zeros((n_lines, output_len), dtype=float)
    n_input = dry_signal.size
    for n in range(output_len):
        # Read delayed samples.
        read_indices = (write_idx - delays) % max_delay
        delayed = buffers[np.arange(n_lines), read_indices]
        # Current FDN output is the delayed samples (before feedback
        # scaling) — users read them as "y_i[n]".
        out[:, n] = delayed
        # Compute new-sample-to-store = mixing · (α · delayed) + b · x[n].
        scaled = decay_gains * delayed
        mixed = mixing @ scaled
        x_n = dry_signal[n] if n < n_input else 0.0
        next_sample = mixed + input_gains * x_n
        buffers[np.arange(n_lines), write_idx] = next_sample
        write_idx = (write_idx + 1) % max_delay
    return out


def _decay_gains_from_rt60(
    delays: NDArray[np.int64], rt60_s: float, fs: float,
) -> NDArray[np.float64]:
    return np.power(
        10.0, -3.0 * delays.astype(float) / (float(rt60_s) * float(fs)),
    )


def fdn_reverb(
    dry_signal: ArrayLike,
    *,
    rt60_s: float,
    fs: float,
    delays: ArrayLike | None = None,
    output_len: int | None = None,
    mixing_matrix: ArrayLike | None = None,
    check_orthogonality: bool = True,
) -> NDArray[np.float64]:
    """Monaural FDN reverb of ``dry_signal`` at target ``rt60_s``.

    Parameters
    ----------
    dry_signal : array_like, shape ``(T,)``
        Monaural input.
    rt60_s : float
        Target reverberation time (seconds).
    fs : float
        Sampling rate in Hz.
    delays : array_like, optional
        Per-line delay lengths in samples.  Defaults to the built-in
        8-line mutually-coprime set.  Must contain at least 2 entries.
    output_len : int, optional
        Output length in samples.  Defaults to
        ``len(dry_signal) + round(2 · rt60_s · fs)`` so the decay tail
        is fully captured.
    mixing_matrix : array_like, optional
        ``(N, N)`` orthogonal matrix.  Defaults to a Hadamard matrix
        for ``N`` powers of two, or a deterministic random orthogonal
        matrix otherwise.  A non-orthogonal matrix is rejected by
        default — see *check_orthogonality*.
    check_orthogonality : bool, optional
        When ``True`` (default), validate that the supplied
        ``mixing_matrix`` satisfies ``M·Mᵀ = I`` within ``1e-6``.
        Non-orthogonal feedback breaks the lossless-loop assumption
        that underlies the ``α_i = 10^{-3 L_i/(RT60·fs)}`` decay
        calibration, so accidentally passing a badly conditioned
        matrix produces an unpredictable ``RT60``.  Set to ``False``
        if you genuinely need a non-unitary feedback matrix.

    Returns
    -------
    ndarray, shape ``(output_len,)``
    """
    if rt60_s <= 0:
        raise ValueError("rt60_s must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")
    dry = np.asarray(dry_signal, dtype=float).reshape(-1)
    if dry.size == 0:
        raise ValueError("dry_signal must be non-empty")
    if delays is None:
        delay_arr = np.asarray(_DEFAULT_DELAYS, dtype=np.int64)
    else:
        delay_arr = np.asarray(delays, dtype=np.int64).reshape(-1)
        if delay_arr.size < 2:
            raise ValueError("delays must have at least 2 entries")
        if np.any(delay_arr <= 0):
            raise ValueError("delays must be positive")
    n_lines = delay_arr.size
    mixing = _resolve_mixing_matrix(
        mixing_matrix, n_lines,
        check_orthogonality=check_orthogonality,
    )
    if output_len is None:
        output_len = dry.size + int(round(2.0 * float(rt60_s) * float(fs)))
    output_len_int = int(output_len)
    if output_len_int <= 0:
        raise ValueError("output_len must be positive")

    decay = _decay_gains_from_rt60(delay_arr, rt60_s, fs)
    input_gains = np.ones(n_lines, dtype=float) / np.sqrt(n_lines)
    out_lines = _fdn_run(
        dry,
        delays=delay_arr,
        decay_gains=decay,
        mixing=mixing,
        input_gains=input_gains,
        output_len=output_len_int,
    )
    # Mix all lines down to monaural.
    return np.asarray(
        np.sum(out_lines, axis=0) / np.sqrt(n_lines), dtype=np.float64,
    )


def fdn_sh_tail(
    dry_signal: ArrayLike,
    *,
    rt60_s: float,
    fs: float,
    max_order: int,
    basis: str = "real",
    delays: ArrayLike | None = None,
    output_len: int | None = None,
    mixing_matrix: ArrayLike | None = None,
    check_orthogonality: bool = True,
    seed: int | None = None,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Ambisonic diffuse tail via an FDN scattered across the sphere.

    Each of the ``N`` FDN delay lines is assigned a fixed random
    direction on the unit sphere (uniform distribution).  Its output
    is encoded as a plane wave at that direction and summed into the
    SH output.  The result is a dense, spatially diffuse late-reverb
    tail with the correct per-channel energy ratio for a uniform
    diffuse field.

    This is the standard "scattered FDN" approach (e.g. Alary et al.
    2017) for feeding a rough but spatially convincing late reverb
    into an ambisonic pipeline.

    Parameters
    ----------
    dry_signal : array_like, shape ``(T,)``
        Monaural dry source.
    rt60_s : float
    fs : float
    max_order : int
        Ambisonic order ``N``.  Output has ``(N+1)²`` SH channels.
    basis : {"real", "complex"}, optional
    delays, mixing_matrix, output_len, check_orthogonality : see
        :func:`fdn_reverb`.
    seed : int, optional
        Seed for the direction distribution.  ``None`` (default)
        draws a fresh set of directions each call, so two consecutive
        renders of the same dry signal give statistically independent
        diffuse tails — the behaviour most users expect.  Pass an
        integer for reproducible output.

    Returns
    -------
    ndarray, shape ``((N+1)², output_len)``
        Real-valued for ``basis="real"`` and complex-valued for
        ``basis="complex"``.
    """
    if rt60_s <= 0:
        raise ValueError("rt60_s must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")
    dry = np.asarray(dry_signal, dtype=float).reshape(-1)
    if dry.size == 0:
        raise ValueError("dry_signal must be non-empty")
    if delays is None:
        delay_arr = np.asarray(_DEFAULT_DELAYS, dtype=np.int64)
    else:
        delay_arr = np.asarray(delays, dtype=np.int64).reshape(-1)
        if delay_arr.size < 2:
            raise ValueError("delays must have at least 2 entries")
        if np.any(delay_arr <= 0):
            raise ValueError("delays must be positive")
    n_lines = delay_arr.size
    mixing = _resolve_mixing_matrix(
        mixing_matrix, n_lines,
        check_orthogonality=check_orthogonality,
    )
    if output_len is None:
        output_len = dry.size + int(round(2.0 * float(rt60_s) * float(fs)))
    output_len_int = int(output_len)
    if output_len_int <= 0:
        raise ValueError("output_len must be positive")

    rng = np.random.default_rng(seed)
    # Uniform-on-sphere sampling.
    u = rng.uniform(-1.0, 1.0, n_lines)
    az = rng.uniform(0.0, 2.0 * np.pi, n_lines)
    colat = np.arccos(np.clip(u, -1.0, 1.0))
    grid = SphericalGrid(
        azimuth=az, angle2=colat, convention="az_colat",
    )
    spec = SHBasisSpec(max_order=int(max_order), basis=basis)
    y = np.asarray(sh_matrix(spec, grid))  # (N, Q)

    decay = _decay_gains_from_rt60(delay_arr, rt60_s, fs)
    input_gains = np.ones(n_lines, dtype=float) / np.sqrt(n_lines)
    out_lines = _fdn_run(
        dry,
        delays=delay_arr,
        decay_gains=decay,
        mixing=mixing,
        input_gains=input_gains,
        output_len=output_len_int,
    )
    # SH output: sh[q, t] = Σ_i Y_q(d_i) · y_i[t]  /  √N.
    sh = (y.T @ out_lines) / np.sqrt(n_lines)
    if np.iscomplexobj(sh):
        return np.asarray(sh, dtype=np.complex128)
    return np.asarray(sh, dtype=np.float64)


__all__ = ["fdn_reverb", "fdn_sh_tail"]
