"""Measured-array transfer-function containers and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..types import ArrayGeometry, SphericalGrid


NormalizationMode = Literal["none", "reference_channel", "mean_sensor", "max_magnitude"]


@dataclass
class ArrayMeasurement:
    """Frequency-domain transfer functions for a measured microphone array.

    The canonical shape is ``(F, M, S)``: frequency bins, sensors, and
    source/measurement directions.  A two-dimensional ``(F, M)`` array is
    accepted and promoted to one source direction.
    """

    frequencies_hz: NDArray[np.float64]
    transfer: NDArray[np.complex128]
    array: ArrayGeometry | None = None
    source_grid: SphericalGrid | None = None
    sample_rate_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequencies_hz = np.asarray(self.frequencies_hz, dtype=float).reshape(-1)
        if self.frequencies_hz.size == 0 or np.any(self.frequencies_hz < 0.0):
            raise ValueError("frequencies_hz must be non-empty and non-negative")
        transfer = np.asarray(self.transfer, dtype=np.complex128)
        if transfer.ndim == 2:
            transfer = transfer[:, :, None]
        if transfer.ndim != 3:
            raise ValueError("transfer must have shape (F, M) or (F, M, S)")
        if transfer.shape[0] != self.frequencies_hz.size:
            raise ValueError("transfer frequency axis must match frequencies_hz")
        if self.array is not None and transfer.shape[1] != self.array.n_sensors:
            raise ValueError("transfer sensor axis must match array.n_sensors")
        if self.source_grid is not None and transfer.shape[2] != self.source_grid.size:
            raise ValueError("transfer source axis must match source_grid.size")
        if self.sample_rate_hz is not None and float(self.sample_rate_hz) <= 0.0:
            raise ValueError("sample_rate_hz must be positive when provided")
        self.transfer = transfer

    @property
    def n_frequencies(self) -> int:
        return int(self.transfer.shape[0])

    @property
    def n_sensors(self) -> int:
        return int(self.transfer.shape[1])

    @property
    def n_sources(self) -> int:
        return int(self.transfer.shape[2])

    def source_slice(self, source_index: int) -> NDArray[np.complex128]:
        """Return the ``(F, M)`` transfer matrix for one source index."""

        idx = int(source_index)
        if idx < 0 or idx >= self.n_sources:
            raise ValueError("source_index out of range")
        return self.transfer[:, :, idx]

    def with_transfer(self, transfer: ArrayLike) -> "ArrayMeasurement":
        """Return a copy with replacement transfer data."""

        return ArrayMeasurement(
            frequencies_hz=self.frequencies_hz.copy(),
            transfer=np.asarray(transfer, dtype=np.complex128),
            array=self.array,
            source_grid=self.source_grid,
            sample_rate_hz=self.sample_rate_hz,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class MeasurementDiagnostics:
    """Summary diagnostics for measured transfer functions."""

    n_frequencies: int
    n_sensors: int
    n_sources: int
    rank_min: int
    rank_max: int
    condition_min: float
    condition_median: float
    condition_max: float
    gain_mismatch_db: NDArray[np.float64]
    phase_mismatch_rad: NDArray[np.float64]


def _transfer3(x: ArrayLike) -> NDArray[np.complex128]:
    arr = np.asarray(x, dtype=np.complex128)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("transfer must have shape (F, M) or (F, M, S)")
    return arr


def transfer_magnitude_db(
    transfer: ArrayLike,
    *,
    reference: float = 1.0,
    floor: float = 1e-15,
) -> NDArray[np.float64]:
    """Magnitude of transfer functions in decibels."""

    ref = float(reference)
    if ref <= 0.0:
        raise ValueError("reference must be positive")
    mag = np.maximum(np.abs(np.asarray(transfer)), float(floor))
    return 20.0 * np.log10(mag / ref)


def transfer_phase_rad(transfer: ArrayLike, *, unwrap: bool = False, axis: int = 0) -> NDArray[np.float64]:
    """Phase of transfer functions in radians."""

    phase = np.angle(np.asarray(transfer))
    return np.unwrap(phase, axis=axis) if unwrap else phase


def transfer_group_delay_s(
    transfer: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    axis: int = 0,
) -> NDArray[np.float64]:
    """Group delay from unwrapped transfer-function phase."""

    f = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if f.size < 2:
        raise ValueError("at least two frequency bins are required")
    phase = transfer_phase_rad(transfer, unwrap=True, axis=axis)
    dphi = np.gradient(phase, axis=axis)
    df = np.gradient(f)
    shape = [1] * phase.ndim
    shape[axis % phase.ndim] = f.size
    return -dphi / (2.0 * np.pi * df.reshape(shape))


def normalize_transfer(
    transfer: ArrayLike,
    *,
    mode: NormalizationMode = "reference_channel",
    reference_channel: int = 0,
    eps: float = 1e-12,
) -> NDArray[np.complex128]:
    """Normalize measured transfer functions by a simple reference rule."""

    h = _transfer3(transfer)
    if mode == "none":
        return h.copy()
    if mode == "reference_channel":
        idx = int(reference_channel)
        if idx < 0 or idx >= h.shape[1]:
            raise ValueError("reference_channel out of range")
        denom = h[:, idx : idx + 1, :]
    elif mode == "mean_sensor":
        denom = np.mean(h, axis=1, keepdims=True)
    elif mode == "max_magnitude":
        denom = np.max(np.abs(h), axis=1, keepdims=True)
    else:
        raise ValueError("mode must be 'none', 'reference_channel', 'mean_sensor', or 'max_magnitude'")
    safe = np.where(np.abs(denom) < float(eps), 1.0 + 0.0j, denom)
    return h / safe


def reference_channel_equalization(
    transfer: ArrayLike,
    *,
    reference_channel: int = 0,
    eps: float = 1e-12,
) -> NDArray[np.complex128]:
    """Alias for reference-channel transfer normalization."""

    return normalize_transfer(
        transfer,
        mode="reference_channel",
        reference_channel=reference_channel,
        eps=eps,
    )


def sensor_mean_magnitude(transfer: ArrayLike) -> NDArray[np.float64]:
    """Mean transfer magnitude per sensor."""

    h = _transfer3(transfer)
    return np.mean(np.abs(h), axis=(0, 2))


def sensor_rms_magnitude(transfer: ArrayLike) -> NDArray[np.float64]:
    """RMS transfer magnitude per sensor."""

    h = _transfer3(transfer)
    return np.sqrt(np.mean(np.abs(h) ** 2, axis=(0, 2)))


def capsule_gain_mismatch_db(transfer: ArrayLike, *, eps: float = 1e-15) -> NDArray[np.float64]:
    """Per-sensor gain mismatch relative to the median sensor RMS."""

    rms = np.maximum(sensor_rms_magnitude(transfer), float(eps))
    ref = float(np.median(rms))
    return 20.0 * np.log10(rms / ref)


def capsule_phase_mismatch_rad(transfer: ArrayLike) -> NDArray[np.float64]:
    """Circular mean phase offset per sensor relative to the sensor median."""

    h = _transfer3(transfer)
    unit = np.exp(1j * np.angle(h))
    mean_phase = np.angle(np.mean(unit, axis=(0, 2)))
    return mean_phase - float(np.median(mean_phase))


def frequency_smooth_magnitude_db(
    magnitude_db: ArrayLike,
    *,
    window_bins: int = 5,
    axis: int = 0,
) -> NDArray[np.float64]:
    """Simple moving-average smoothing for magnitude responses in dB."""

    mag = np.asarray(magnitude_db, dtype=float)
    win = int(window_bins)
    if win <= 1:
        return mag.copy()
    kernel = np.ones(win, dtype=float) / win
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis, mag)


def steering_condition_numbers(transfer: ArrayLike) -> NDArray[np.float64]:
    """Condition number of each frequency-bin steering matrix."""

    h = _transfer3(transfer)
    out = np.empty(h.shape[0], dtype=float)
    for i in range(h.shape[0]):
        out[i] = np.linalg.cond(h[i])
    return out


def steering_ranks(transfer: ArrayLike, *, rtol: float = 1e-10) -> NDArray[np.int64]:
    """Numerical rank of each frequency-bin steering matrix."""

    h = _transfer3(transfer)
    out = np.empty(h.shape[0], dtype=np.int64)
    for i in range(h.shape[0]):
        s = np.linalg.svd(h[i], compute_uv=False)
        out[i] = int(np.count_nonzero(s > float(rtol) * s[0])) if s.size else 0
    return out


def tikhonov_inverse_bank(
    matrix_bank: ArrayLike,
    *,
    regularization: float | ArrayLike = 1e-6,
) -> NDArray[np.complex128]:
    """Per-frequency Tikhonov pseudoinverse for a matrix bank.

    Input shape is ``(F, M, Q)`` and output shape is ``(F, Q, M)``.
    """

    h = _transfer3(matrix_bank)
    reg = np.asarray(regularization, dtype=float)
    if reg.ndim == 0:
        reg = np.full(h.shape[0], float(reg))
    if reg.shape != (h.shape[0],):
        raise ValueError("regularization must be scalar or length F")
    out = np.empty((h.shape[0], h.shape[2], h.shape[1]), dtype=np.complex128)
    for i in range(h.shape[0]):
        lam = float(reg[i])
        if lam < 0.0:
            raise ValueError("regularization must be non-negative")
        hi = h[i]
        gram = hi.conj().T @ hi
        out[i] = np.linalg.solve(
            gram + lam * np.eye(gram.shape[0], dtype=gram.dtype),
            hi.conj().T,
        )
    return out


def apply_inverse_bank(
    inverse_bank: ArrayLike,
    sensor_spectra: ArrayLike,
) -> NDArray[np.complex128]:
    """Apply a ``(F, Q, M)`` inverse bank to ``(F, M, ...)`` spectra."""

    inv = np.asarray(inverse_bank, dtype=np.complex128)
    x = np.asarray(sensor_spectra, dtype=np.complex128)
    if inv.ndim != 3 or x.ndim < 2:
        raise ValueError("inverse_bank must be (F, Q, M) and sensor_spectra at least (F, M)")
    if inv.shape[0] != x.shape[0] or inv.shape[2] != x.shape[1]:
        raise ValueError("frequency and sensor axes must match")
    return np.einsum("fqm,fm...->fq...", inv, x)


def regularization_sweep_error(
    matrix_bank: ArrayLike,
    targets: ArrayLike,
    *,
    regularizations: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruction error for candidate Tikhonov regularizations."""

    h = _transfer3(matrix_bank)
    y = np.asarray(targets, dtype=np.complex128)
    if y.shape[:2] != h.shape[:2]:
        raise ValueError("targets must start with shape (F, M)")
    regs = np.asarray(regularizations, dtype=float).reshape(-1)
    errors = []
    for lam in regs:
        inv = tikhonov_inverse_bank(h, regularization=float(lam))
        estimate = apply_inverse_bank(inv, y)
        recon = np.einsum("fmq,fq...->fm...", h, estimate)
        denom = np.linalg.norm(y)
        errors.append(float(np.linalg.norm(recon - y) / max(denom, 1e-15)))
    return np.asarray(errors, dtype=float)


def best_regularization(
    matrix_bank: ArrayLike,
    targets: ArrayLike,
    *,
    regularizations: ArrayLike,
) -> float:
    """Return the regularization candidate with smallest reconstruction error."""

    regs = np.asarray(regularizations, dtype=float).reshape(-1)
    errors = regularization_sweep_error(
        matrix_bank,
        targets,
        regularizations=regs,
    )
    return float(regs[int(np.argmin(errors))])


def measurement_diagnostics(measurement: ArrayMeasurement | ArrayLike) -> MeasurementDiagnostics:
    """Compute summary diagnostics for measured transfer functions."""

    if isinstance(measurement, ArrayMeasurement):
        h = measurement.transfer
    else:
        h = _transfer3(measurement)
    ranks = steering_ranks(h)
    cond = steering_condition_numbers(h)
    return MeasurementDiagnostics(
        n_frequencies=int(h.shape[0]),
        n_sensors=int(h.shape[1]),
        n_sources=int(h.shape[2]),
        rank_min=int(np.min(ranks)),
        rank_max=int(np.max(ranks)),
        condition_min=float(np.min(cond)),
        condition_median=float(np.median(cond)),
        condition_max=float(np.max(cond)),
        gain_mismatch_db=capsule_gain_mismatch_db(h),
        phase_mismatch_rad=capsule_phase_mismatch_rad(h),
    )


def source_frequency_slice(measurement: ArrayMeasurement, frequency_index: int) -> NDArray[np.complex128]:
    """Return the ``(M, S)`` steering matrix for one frequency index."""

    idx = int(frequency_index)
    if idx < 0 or idx >= measurement.n_frequencies:
        raise ValueError("frequency_index out of range")
    return measurement.transfer[idx]


def interpolate_transfer_linear(
    measurement: ArrayMeasurement,
    new_frequencies_hz: ArrayLike,
) -> ArrayMeasurement:
    """Linearly interpolate complex transfer functions over frequency."""

    new_f = np.asarray(new_frequencies_hz, dtype=float).reshape(-1)
    if np.any(new_f < measurement.frequencies_hz[0]) or np.any(new_f > measurement.frequencies_hz[-1]):
        raise ValueError("new frequencies must lie inside the measured frequency range")
    real = np.empty((new_f.size, measurement.n_sensors, measurement.n_sources), dtype=float)
    imag = np.empty_like(real)
    for m in range(measurement.n_sensors):
        for s in range(measurement.n_sources):
            real[:, m, s] = np.interp(new_f, measurement.frequencies_hz, measurement.transfer[:, m, s].real)
            imag[:, m, s] = np.interp(new_f, measurement.frequencies_hz, measurement.transfer[:, m, s].imag)
    return ArrayMeasurement(
        frequencies_hz=new_f,
        transfer=real + 1j * imag,
        array=measurement.array,
        source_grid=measurement.source_grid,
        sample_rate_hz=measurement.sample_rate_hz,
        metadata=dict(measurement.metadata),
    )


__all__ = [
    "ArrayMeasurement",
    "MeasurementDiagnostics",
    "NormalizationMode",
    "apply_inverse_bank",
    "best_regularization",
    "capsule_gain_mismatch_db",
    "capsule_phase_mismatch_rad",
    "frequency_smooth_magnitude_db",
    "interpolate_transfer_linear",
    "measurement_diagnostics",
    "normalize_transfer",
    "reference_channel_equalization",
    "regularization_sweep_error",
    "sensor_mean_magnitude",
    "sensor_rms_magnitude",
    "source_frequency_slice",
    "steering_condition_numbers",
    "steering_ranks",
    "tikhonov_inverse_bank",
    "transfer_group_delay_s",
    "transfer_magnitude_db",
    "transfer_phase_rad",
]
