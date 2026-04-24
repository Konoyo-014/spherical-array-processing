"""Shoebox image-source room-impulse-response simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SphericalGrid


Interpolation = Literal["nearest", "sinc"]


def _write_mask(
    delay_samples: NDArray[np.float64],
    n_samples: int,
    *,
    interpolation: Interpolation,
    fir_taps: int,
) -> NDArray[np.bool_]:
    """Return the image-source mask for the chosen write strategy.

    ``"nearest"`` keeps contributions whose rounded target index falls
    inside the output buffer.  ``"sinc"`` keeps contributions whose
    clipped fractional-delay kernel still overlaps the buffer, even if
    the kernel centre lies slightly beyond either edge.
    """
    if interpolation == "nearest":
        sample_idx = np.round(delay_samples).astype(np.int64)
        return (sample_idx >= 0) & (sample_idx < int(n_samples))
    if interpolation == "sinc":
        L = int(fir_taps)
        if L < 3:
            raise ValueError("fir_taps must be >= 3")
        if L % 2 == 0:
            L += 1
        half = (L - 1) // 2
        center = np.round(delay_samples).astype(np.int64)
        return (center + half >= 0) & (center - half < int(n_samples))
    raise ValueError(
        f"interpolation must be 'nearest' or 'sinc', got {interpolation!r}"
    )


def _scatter_sinc(
    out: NDArray,
    delay_samples: NDArray[np.float64],
    weights: NDArray,
    *,
    fir_taps: int,
    beta: float = 8.6,
) -> None:
    """Add Kaiser-windowed sinc fractional-delay kernels into *out*.

    Each entry of *delay_samples* places a kernel centred on its
    fractional location, weighted by the matching column of *weights*.
    The kernel has length *fir_taps* (forced odd) so its group delay
    is ``(fir_taps - 1) / 2``.  Kernels hanging off either end of the
    buffer are truncated rather than wrapped.

    *out* may be ``(T,)`` for monaural IRs or ``(Q, T)`` for
    multi-channel SH IRs; *weights* must broadcast to the same
    leading shape.
    """
    n_samples = out.shape[-1]
    L = int(fir_taps)
    if L < 3:
        raise ValueError("fir_taps must be >= 3")
    if L % 2 == 0:
        L += 1
    half = (L - 1) // 2
    window = np.kaiser(L, float(beta))
    tap_idx = np.arange(L) - half  # -half .. +half
    weights_arr = np.asarray(weights)
    for k in range(delay_samples.size):
        d = float(delay_samples[k])
        center = int(np.round(d))
        frac = d - center
        kernel = np.sinc(tap_idx - frac) * window
        lo_target = center - half
        hi_target = center + half + 1
        lo_src = max(0, -lo_target)
        hi_src = L - max(0, hi_target - n_samples)
        lo_clip = lo_target + lo_src
        hi_clip = hi_target - (L - hi_src)
        if lo_src >= hi_src or lo_clip >= hi_clip:
            continue
        w_k = weights_arr[..., k] if weights_arr.ndim > 0 else weights_arr
        out[..., lo_clip:hi_clip] += (
            np.asarray(w_k)[..., None] * kernel[lo_src:hi_src]
        )


@dataclass(frozen=True)
class ShoeboxRoom:
    """Rectangular-room geometry plus per-wall reflection coefficients.

    Attributes
    ----------
    dimensions_m : tuple[float, float, float]
        Room extents ``(Lx, Ly, Lz)`` in metres.
    reflection : tuple[float, float, float, float, float, float]
        Wall reflection coefficients in the order
        ``(−x, +x, −y, +y, −z, +z)``.  Values in ``[0, 1]``; ``1`` is
        a rigid wall, ``0`` is anechoic.  A single scalar broadcasts
        to all six walls.
    """

    dimensions_m: tuple[float, float, float]
    reflection: tuple[float, float, float, float, float, float] | float = (0.7,) * 6

    def __post_init__(self) -> None:
        dims = np.asarray(self.dimensions_m, dtype=float)
        if dims.shape != (3,) or np.any(dims <= 0.0):
            raise ValueError("dimensions_m must be 3 positive floats")
        refl = np.asarray(self.reflection, dtype=float)
        if refl.ndim == 0:
            refl = np.full(6, float(refl), dtype=float)
        else:
            refl = refl.reshape(-1)
        if refl.shape != (6,) or np.any(refl < 0.0) or np.any(refl > 1.0):
            raise ValueError("reflection must be a scalar or 6 values in [0, 1]")
        object.__setattr__(self, "dimensions_m", tuple(dims.tolist()))
        object.__setattr__(self, "reflection", tuple(refl.tolist()))

    @classmethod
    def with_uniform_reflection(
        cls, dimensions_m: tuple[float, float, float], coefficient: float
    ) -> "ShoeboxRoom":
        """Convenience constructor for a room with identical walls."""
        r = float(coefficient)
        return cls(dimensions_m=tuple(map(float, dimensions_m)), reflection=(r,) * 6)


def _image_source_grid(
    source_m: NDArray[np.float64],
    room_lengths: NDArray[np.float64],
    max_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return the image-source positions and the wall-bounce count per axis.

    Parameters
    ----------
    source_m : ndarray, shape (3,)
    room_lengths : ndarray, shape (3,)
    max_order : int
        Maximum reflection order along any axis.  Total image-sources
        scale as ``(2·max_order + 1)³``.

    Returns
    -------
    positions : ndarray, shape (K, 3)
    bounces : ndarray, shape (K, 6), int
        Number of reflections on each of the six walls
        ``(−x, +x, −y, +y, −z, +z)`` for each image source.
    """
    order_axis = np.arange(-max_order, max_order + 1)
    nx, ny, nz = np.meshgrid(order_axis, order_axis, order_axis, indexing="ij")
    n = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=-1)  # (K, 3)

    # Position: for axis a, image source is at
    #   x_im = n·L + s_a   when n is even,
    #   x_im = n·L + (L - s_a) when n is odd.
    # Equivalently  x_im = n·L + (-1)**n · s_a + (1 - (-1)**n)/2 · L
    parity = np.abs(n) % 2  # (K, 3) — 0 or 1 per axis
    # When parity=1 (odd bounce), mirror the source about the nearer wall.
    flipped = np.where(parity == 1, room_lengths - source_m, source_m)
    positions = n * room_lengths + flipped  # (K, 3)

    # Bounce counts per wall: if n is even with n=0, no bounce; if n=±2k,
    # it reflected k times off each of the corresponding walls.  We adopt
    # the simpler model that the image at reflection order |n| reflects
    # ``ceil(|n|/2)`` times off the further wall and ``floor(|n|/2)``
    # times off the nearer wall, regardless of sign.
    abs_n = np.abs(n)
    bounces = np.zeros((n.shape[0], 6), dtype=np.int64)
    # Axis 0 — walls (-x=0, +x=1); axis 1 → (-y, +y); axis 2 → (-z, +z).
    for axis in range(3):
        neg_wall = 2 * axis
        pos_wall = 2 * axis + 1
        # Number of reflections off the wall on the positive side of
        # the source: image travels through the positive wall on bounces
        # n[axis] > 0 (odd) and the negative wall on bounces n[axis] < 0.
        pos_bounces = np.where(
            n[:, axis] >= 0,
            (abs_n[:, axis] + 1) // 2,
            abs_n[:, axis] // 2,
        )
        neg_bounces = np.where(
            n[:, axis] >= 0,
            abs_n[:, axis] // 2,
            (abs_n[:, axis] + 1) // 2,
        )
        bounces[:, neg_wall] = neg_bounces
        bounces[:, pos_wall] = pos_bounces
    return positions.astype(float), bounces


def _shoebox_contributions(
    room: ShoeboxRoom,
    source_position_m: ArrayLike,
    listener_position_m: ArrayLike,
    *,
    fs: float,
    ir_length: int,
    max_reflection_order: int,
    c: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return per-image directions, delays (s), fractional delay
    (samples) and amplitudes.  The fractional delay lets callers
    choose either nearest-sample or sinc-interpolated scattering.
    """
    room_lengths = np.asarray(room.dimensions_m, dtype=float)
    src = np.asarray(source_position_m, dtype=float)
    lis = np.asarray(listener_position_m, dtype=float)
    for name, pos in (("source", src), ("listener", lis)):
        if pos.shape != (3,):
            raise ValueError(f"{name}_position_m must have shape (3,)")
        if np.any(pos <= 0.0) or np.any(pos >= room_lengths):
            raise ValueError(
                f"{name}_position_m must lie strictly inside the room"
            )

    fs_f = float(fs)
    if fs_f <= 0.0:
        raise ValueError("fs must be positive")
    n_samples = int(ir_length)
    if n_samples <= 0:
        raise ValueError("ir_length must be positive")
    max_order = int(max_reflection_order)
    if max_order < 0:
        raise ValueError("max_reflection_order must be non-negative")
    c_f = float(c)
    if c_f <= 0.0:
        raise ValueError("c must be positive")

    refl = np.asarray(room.reflection, dtype=float)
    positions, bounces = _image_source_grid(src, room_lengths, max_order)
    displacement = positions - lis[None, :]
    distances = np.linalg.norm(displacement, axis=1)
    safe_dist = np.maximum(distances, 1e-9)
    delays_s = distances / c_f
    amplitudes = np.prod(refl ** bounces, axis=1) / safe_dist
    delay_samples = delays_s * fs_f  # fractional
    mask = amplitudes > 0.0

    keep_disp = displacement[mask]
    keep_dist = safe_dist[mask]
    arrival_dirs = np.zeros_like(keep_disp)
    nonzero = keep_dist > 1e-12
    arrival_dirs[nonzero] = keep_disp[nonzero] / keep_dist[nonzero, None]
    return (
        arrival_dirs,
        delays_s[mask],
        delay_samples[mask],
        amplitudes[mask],
    )


def shoebox_rir(
    room: ShoeboxRoom,
    source_position_m: ArrayLike,
    listener_position_m: ArrayLike,
    *,
    fs: float,
    ir_length: int,
    max_reflection_order: int = 16,
    c: float = 343.0,
    interpolation: Interpolation = "nearest",
    fir_taps: int = 21,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Monaural shoebox room impulse response via image sources.

    Parameters
    ----------
    room : ShoeboxRoom
    source_position_m, listener_position_m : array_like, shape (3,)
        Cartesian positions inside the room, in metres.  Must satisfy
        ``0 < x_i < L_i`` for every axis.
    fs : float
        Sampling rate in Hz.
    ir_length : int
        Output IR length in samples.  For ``interpolation="nearest"``
        contributions whose rounded target index falls outside the
        buffer are dropped.  For ``interpolation="sinc"`` a
        contribution is kept as long as its clipped FIR kernel still
        overlaps the buffer.
    max_reflection_order : int, optional
        Cap on the per-axis bounce count.  Total image-source count is
        ``(2·max_reflection_order + 1)³``.  Default ``16``.
    c : float, optional
        Speed of sound.  Default ``343`` m/s.
    interpolation : {"nearest", "sinc"}, optional
        How each image-source contribution is written into the IR
        buffer.  ``"nearest"`` (default) rounds the arrival time to
        the closest sample — fast and what the classical Allen–Berkley
        implementation does, but introduces up to ``1/(2·fs)`` of
        time-quantisation error per reflection (≈ ±10 µs at 48 kHz).
        ``"sinc"`` scatters a Kaiser-windowed fractional-delay FIR
        kernel so the arrival time is reproduced to the fractional
        sample — noticeably cleaner high-frequency comb structure at
        the cost of ``O(K · fir_taps)`` instead of ``O(K)`` writes.
    fir_taps : int, optional
        Kernel length for ``interpolation="sinc"``.  Default ``21``
        (≈ 0.4 ms at 48 kHz).  Forced odd internally.  Larger values
        reduce high-frequency aliasing further but widen the pre- /
        post-ringing around each reflection.

    Returns
    -------
    rir : ndarray, shape (ir_length,)
        Band-unlimited monaural impulse response at the listener.
    arrival_dirs_xyz : ndarray, shape (K, 3)
        Unit-norm arrival directions for every image source kept
        within the IR window — useful for feeding
        :func:`shoebox_sh_rir` or for debugging.
    arrival_delays_s : ndarray, shape (K,)
        Arrival times of the corresponding images in seconds.
    """
    arrival_dirs, delays_s, delay_samples, amplitudes = _shoebox_contributions(
        room,
        source_position_m,
        listener_position_m,
        fs=fs,
        ir_length=ir_length,
        max_reflection_order=max_reflection_order,
        c=c,
    )
    write_mask = _write_mask(
        delay_samples, int(ir_length),
        interpolation=interpolation, fir_taps=fir_taps,
    )
    arrival_dirs = arrival_dirs[write_mask]
    delays_s = delays_s[write_mask]
    delay_samples = delay_samples[write_mask]
    amplitudes = amplitudes[write_mask]

    rir = np.zeros(int(ir_length), dtype=float)
    if interpolation == "nearest":
        sample_idx = np.round(delay_samples).astype(np.int64)
        np.add.at(rir, sample_idx, amplitudes)
    elif interpolation == "sinc":
        if delay_samples.size > 0:
            _scatter_sinc(
                rir, delay_samples, amplitudes, fir_taps=fir_taps,
            )
    else:
        raise ValueError(
            f"interpolation must be 'nearest' or 'sinc', got "
            f"{interpolation!r}"
        )
    return rir, arrival_dirs, delays_s


def shoebox_sh_rir(
    room: ShoeboxRoom,
    source_position_m: ArrayLike,
    listener_position_m: ArrayLike,
    *,
    fs: float,
    ir_length: int,
    max_order: int,
    basis: str = "real",
    max_reflection_order: int = 16,
    c: float = 343.0,
    interpolation: Interpolation = "nearest",
    fir_taps: int = 21,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Ambisonic shoebox room impulse response.

    Convenience wrapper: for each image-source contribution returned by
    :func:`shoebox_rir`, encode the arrival direction into the
    requested SH basis and scatter its amplitude into the matching
    sample bin and SH channel.  The result is a ``((N+1)^2, T)``
    impulse response that, when convolved with a monaural source
    signal, gives an Ambisonic microphone recording of the room at
    the listener.

    Parameters
    ----------
    interpolation : {"nearest", "sinc"}, optional
        See :func:`shoebox_rir`.  Applied identically along every SH
        channel — all channels share the same fractional kernel per
        image source, so arrival timing stays consistent across the
        spatial basis.
    fir_taps : int, optional
        Kernel length for ``interpolation="sinc"``.  Default ``21``.

    Returns
    -------
    ndarray, shape ((N+1)^2, ir_length)
        Ambisonic room impulse response in ACN order.
    """
    spec = SHBasisSpec(max_order=int(max_order), basis=basis, angle_convention="az_el")
    n_coeffs = spec.n_coeffs
    out_dtype = np.complex128 if spec.basis == "complex" else np.float64
    sh = np.zeros((n_coeffs, int(ir_length)), dtype=out_dtype)
    arrival_dirs, _delays, delay_samples, amplitudes = _shoebox_contributions(
        room,
        source_position_m,
        listener_position_m,
        fs=fs,
        ir_length=ir_length,
        max_reflection_order=max_reflection_order,
        c=c,
    )
    write_mask = _write_mask(
        delay_samples, int(ir_length),
        interpolation=interpolation, fir_taps=fir_taps,
    )
    arrival_dirs = arrival_dirs[write_mask]
    delay_samples = delay_samples[write_mask]
    amplitudes = amplitudes[write_mask]
    if arrival_dirs.size == 0:
        return sh
    # Convert Cartesian directions to (azimuth, elevation) for sh.matrix.
    az = np.arctan2(arrival_dirs[:, 1], arrival_dirs[:, 0]) % (2.0 * np.pi)
    el = np.arcsin(np.clip(arrival_dirs[:, 2], -1.0, 1.0))
    grid = SphericalGrid(azimuth=az, angle2=el, convention="az_el")
    y = np.asarray(sh_matrix(spec, grid))  # (K, Q)
    if interpolation == "nearest":
        sample_idx = np.round(delay_samples).astype(np.int64)
        for k in range(sample_idx.size):
            sh[:, sample_idx[k]] += amplitudes[k] * y[k]
    elif interpolation == "sinc":
        # Weights per image-source: amplitudes · Y — shape (Q, K).
        weights = (amplitudes[:, None] * y).T.astype(sh.dtype)
        _scatter_sinc(sh, delay_samples, weights, fir_taps=fir_taps)
    else:
        raise ValueError(
            f"interpolation must be 'nearest' or 'sinc', got "
            f"{interpolation!r}"
        )
    return sh


__all__ = ["ShoeboxRoom", "shoebox_rir", "shoebox_sh_rir"]
