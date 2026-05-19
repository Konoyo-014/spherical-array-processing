"""Ambisonic decoder diagnostics and response metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..coords import unit_sph_to_cart
from ..types import SphericalGrid


def infer_decoder_order(decoder: ArrayLike) -> int:
    """Infer Ambisonic order from decoder column count."""

    d = np.asarray(decoder)
    if d.ndim != 2:
        raise ValueError("decoder must be a 2-D matrix")
    root = int(round(np.sqrt(d.shape[1])))
    if root * root != d.shape[1]:
        raise ValueError("decoder column count must equal (max_order + 1) ** 2")
    return root - 1


def validate_decoder_matrix(decoder: ArrayLike, max_order: int | None = None) -> NDArray:
    """Return a decoder matrix after shape/order validation."""

    d = np.asarray(decoder)
    if d.ndim != 2:
        raise ValueError("decoder must be a 2-D matrix")
    order = infer_decoder_order(d) if max_order is None else int(max_order)
    expected = (order + 1) ** 2
    if d.shape[1] != expected:
        raise ValueError(f"decoder has {d.shape[1]} columns but max_order={order} expects {expected}")
    return d


def decoder_singular_values(decoder: ArrayLike) -> NDArray[np.float64]:
    """Singular values of a decoder matrix."""

    return np.linalg.svd(validate_decoder_matrix(decoder), compute_uv=False)


def decoder_rank(decoder: ArrayLike, tol: float | None = None) -> int:
    """Numerical rank of a decoder matrix."""

    return int(np.linalg.matrix_rank(validate_decoder_matrix(decoder), tol=tol))


def decoder_condition_number(decoder: ArrayLike, *, floor: float = 1e-15) -> float:
    """Decoder condition number from its singular values."""

    s = decoder_singular_values(decoder)
    if s.size == 0 or float(s[-1]) <= float(floor):
        return float("inf")
    return float(s[0] / s[-1])


def decoder_frobenius_norm(decoder: ArrayLike) -> float:
    """Frobenius norm of the decoder matrix."""

    return float(np.linalg.norm(validate_decoder_matrix(decoder), ord="fro"))


def decoder_column_norms(decoder: ArrayLike) -> NDArray[np.float64]:
    """Per-Ambisonic-channel decoder column norms."""

    return np.linalg.norm(validate_decoder_matrix(decoder), axis=0).astype(float)


def decoder_row_norms(decoder: ArrayLike) -> NDArray[np.float64]:
    """Per-loudspeaker decoder row norms."""

    return np.linalg.norm(validate_decoder_matrix(decoder), axis=1).astype(float)


def decoder_column_power(decoder: ArrayLike) -> NDArray[np.float64]:
    """Per-Ambisonic-channel decoded power."""

    d = validate_decoder_matrix(decoder)
    return np.sum(np.abs(d) ** 2, axis=0).astype(float)


def decoder_row_power(decoder: ArrayLike) -> NDArray[np.float64]:
    """Per-loudspeaker diffuse-input power."""

    d = validate_decoder_matrix(decoder)
    return np.sum(np.abs(d) ** 2, axis=1).astype(float)


def decoder_diffuse_loudspeaker_levels_db(
    decoder: ArrayLike,
    *,
    reference: str | float = "mean",
    floor: float = 1e-30,
) -> NDArray[np.float64]:
    """Diffuse-field loudspeaker level offsets in dB."""

    power = np.maximum(decoder_row_power(decoder), float(floor))
    if reference == "mean":
        ref = float(np.mean(power))
    elif reference == "max":
        ref = float(np.max(power))
    else:
        ref = float(reference)
    if ref <= 0.0:
        raise ValueError("reference must be positive")
    return 10.0 * np.log10(power / ref)


def decoder_loudspeaker_gain_spread_db(decoder: ArrayLike) -> float:
    """Peak-to-peak diffuse loudspeaker level spread in dB."""

    levels = decoder_diffuse_loudspeaker_levels_db(decoder)
    return float(np.max(levels) - np.min(levels))


def decoder_mode_gain_spread_db(decoder: ArrayLike, *, floor: float = 1e-30) -> float:
    """Peak-to-peak decoded SH-channel power spread in dB."""

    power = np.maximum(decoder_column_power(decoder), float(floor))
    levels = 10.0 * np.log10(power / float(np.mean(power)))
    return float(np.max(levels) - np.min(levels))


def decoder_energy_matrix(decoder: ArrayLike) -> NDArray:
    """Return ``DᴴD`` for decoder energy-preservation inspection."""

    d = validate_decoder_matrix(decoder)
    return d.conj().T @ d


def decoder_energy_preservation_error(decoder: ArrayLike, target: float | None = None) -> float:
    """Relative Frobenius error of ``DᴴD`` against a scaled identity."""

    gram = decoder_energy_matrix(decoder)
    if target is None:
        target = float(np.real(np.trace(gram)) / gram.shape[0])
    ref = float(target) * np.eye(gram.shape[0], dtype=gram.dtype)
    denom = max(float(np.linalg.norm(ref, ord="fro")), 1e-30)
    return float(np.linalg.norm(gram - ref, ord="fro") / denom)


def decoder_mode_leakage_ratio(decoder: ArrayLike) -> float:
    """Off-diagonal-to-diagonal energy ratio of ``DᴴD``."""

    gram = decoder_energy_matrix(decoder)
    off = gram.copy()
    np.fill_diagonal(off, 0.0)
    diag = np.diag(np.diag(gram))
    return float(np.linalg.norm(off, ord="fro") / max(float(np.linalg.norm(diag, ord="fro")), 1e-30))


def decoder_column_correlation(decoder: ArrayLike, *, floor: float = 1e-30) -> NDArray:
    """Normalised decoder-column correlation matrix."""

    gram = decoder_energy_matrix(decoder)
    norms = np.sqrt(np.maximum(np.real(np.diag(gram)), float(floor)))
    return gram / (norms[:, None] * norms[None, :])


def decoder_row_correlation(decoder: ArrayLike, *, floor: float = 1e-30) -> NDArray:
    """Normalised loudspeaker-row correlation matrix."""

    d = validate_decoder_matrix(decoder)
    gram = d @ d.conj().T
    norms = np.sqrt(np.maximum(np.real(np.diag(gram)), float(floor)))
    return gram / (norms[:, None] * norms[None, :])


def decoder_projection_matrix(decoder: ArrayLike, *, rcond: float = 1e-12) -> NDArray:
    """Projection matrix onto the decoder row space."""

    d = validate_decoder_matrix(decoder)
    return d @ np.linalg.pinv(d, rcond=float(rcond))


def mode_matching_matrix(
    decoder: ArrayLike,
    loudspeaker_basis: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> NDArray:
    """Return ``YᴴWD`` mode-response matrix for a loudspeaker basis ``Y``."""

    d = validate_decoder_matrix(decoder)
    y = np.asarray(loudspeaker_basis)
    if y.ndim != 2 or y.shape != d.shape:
        raise ValueError("loudspeaker_basis must have the same shape as decoder")
    if weights is None:
        yw = y
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != (y.shape[0],):
            raise ValueError("weights must match loudspeaker count")
        yw = y * w[:, None]
    return y.conj().T @ d if weights is None else yw.conj().T @ d


def mode_matching_error(
    decoder: ArrayLike,
    loudspeaker_basis: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    target: ArrayLike | None = None,
    relative: bool = True,
) -> float:
    """Frobenius error of the mode-response matrix against a target."""

    response = mode_matching_matrix(decoder, loudspeaker_basis, weights=weights)
    ref = np.eye(response.shape[0], dtype=response.dtype) if target is None else np.asarray(target)
    if ref.shape != response.shape:
        raise ValueError("target must match mode response shape")
    err = float(np.linalg.norm(response - ref, ord="fro"))
    if not relative:
        return err
    return err / max(float(np.linalg.norm(ref, ord="fro")), 1e-30)


def mode_response_diagonal(
    decoder: ArrayLike,
    loudspeaker_basis: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> NDArray:
    """Diagonal gains of the loudspeaker-basis mode-response matrix."""

    return np.diag(mode_matching_matrix(decoder, loudspeaker_basis, weights=weights))


def mode_response_leakage_ratio(
    decoder: ArrayLike,
    loudspeaker_basis: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Off-diagonal-to-diagonal ratio of a mode-response matrix."""

    response = mode_matching_matrix(decoder, loudspeaker_basis, weights=weights)
    off = response.copy()
    np.fill_diagonal(off, 0.0)
    diag = np.diag(np.diag(response))
    return float(np.linalg.norm(off, ord="fro") / max(float(np.linalg.norm(diag, ord="fro")), 1e-30))


def speaker_directions_cartesian(loudspeaker_grid: SphericalGrid) -> NDArray[np.float64]:
    """Cartesian unit vectors for a loudspeaker grid."""

    return unit_sph_to_cart(
        loudspeaker_grid.azimuth,
        loudspeaker_grid.angle2,
        convention=loudspeaker_grid.convention,
    )


def probe_response(decoder: ArrayLike, probe_basis: ArrayLike) -> NDArray:
    """Loudspeaker responses for probe SH coefficient rows."""

    d = validate_decoder_matrix(decoder)
    y = np.asarray(probe_basis)
    if y.ndim != 2 or y.shape[1] != d.shape[1]:
        raise ValueError("probe_basis must have shape (n_probes, n_coeffs)")
    return y @ d.conj().T


def probe_response_energy(decoder: ArrayLike, probe_basis: ArrayLike) -> NDArray[np.float64]:
    """Total loudspeaker energy for each probe direction."""

    response = probe_response(decoder, probe_basis)
    return np.sum(np.abs(response) ** 2, axis=1).astype(float)


def probe_response_peak_speaker(decoder: ArrayLike, probe_basis: ArrayLike) -> NDArray[np.int64]:
    """Index of the strongest loudspeaker for each probe response."""

    response = probe_response(decoder, probe_basis)
    return np.argmax(np.abs(response), axis=1).astype(np.int64)


def probe_energy_vector(
    decoder: ArrayLike,
    probe_basis: ArrayLike,
    loudspeaker_grid: SphericalGrid,
) -> NDArray[np.float64]:
    """Energy-vector direction for each probe response."""

    response = probe_response(decoder, probe_basis)
    xyz = speaker_directions_cartesian(loudspeaker_grid)
    if response.shape[1] != xyz.shape[0]:
        raise ValueError("decoder row count must match loudspeaker grid")
    power = np.abs(response) ** 2
    denom = np.maximum(np.sum(power, axis=1, keepdims=True), 1e-30)
    return np.asarray((power @ xyz) / denom, dtype=float)


def probe_velocity_vector(
    decoder: ArrayLike,
    probe_basis: ArrayLike,
    loudspeaker_grid: SphericalGrid,
) -> NDArray[np.float64]:
    """Amplitude/velocity-vector direction for each probe response."""

    response = np.real(probe_response(decoder, probe_basis))
    xyz = speaker_directions_cartesian(loudspeaker_grid)
    if response.shape[1] != xyz.shape[0]:
        raise ValueError("decoder row count must match loudspeaker grid")
    denom = np.maximum(np.sum(np.abs(response), axis=1, keepdims=True), 1e-30)
    return np.asarray((response @ xyz) / denom, dtype=float)


def vector_magnitudes(vectors: ArrayLike) -> NDArray[np.float64]:
    """Euclidean magnitudes for a stack of 3-D vectors."""

    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vectors must have shape (n, 3)")
    return np.linalg.norm(v, axis=1).astype(float)


def vector_angle_errors_deg(vectors: ArrayLike, target_vectors: ArrayLike) -> NDArray[np.float64]:
    """Angular errors between vector rows in degrees."""

    v = np.asarray(vectors, dtype=float)
    t = np.asarray(target_vectors, dtype=float)
    if v.shape != t.shape or v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vectors and target_vectors must have matching shape (n, 3)")
    vn = np.linalg.norm(v, axis=1)
    tn = np.linalg.norm(t, axis=1)
    dots = np.sum(v * t, axis=1) / np.maximum(vn * tn, 1e-30)
    out = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return np.where((vn > 1e-12) & (tn > 1e-12), out, np.nan).astype(float)


def probe_energy_vector_errors_deg(
    decoder: ArrayLike,
    probe_basis: ArrayLike,
    loudspeaker_grid: SphericalGrid,
    target_grid: SphericalGrid,
) -> NDArray[np.float64]:
    """Energy-vector angular error for probe directions."""

    target = speaker_directions_cartesian(target_grid)
    return vector_angle_errors_deg(probe_energy_vector(decoder, probe_basis, loudspeaker_grid), target)


def probe_velocity_vector_errors_deg(
    decoder: ArrayLike,
    probe_basis: ArrayLike,
    loudspeaker_grid: SphericalGrid,
    target_grid: SphericalGrid,
) -> NDArray[np.float64]:
    """Velocity-vector angular error for probe directions."""

    target = speaker_directions_cartesian(target_grid)
    return vector_angle_errors_deg(probe_velocity_vector(decoder, probe_basis, loudspeaker_grid), target)


def normalize_decoder_column_norms(decoder: ArrayLike, *, target_norm: float = 1.0, floor: float = 1e-30) -> NDArray:
    """Scale decoder columns to a target norm."""

    d = validate_decoder_matrix(decoder)
    norms = np.maximum(decoder_column_norms(d), float(floor))
    return d * (float(target_norm) / norms)[None, :]


def normalize_decoder_row_norms(decoder: ArrayLike, *, target_norm: float = 1.0, floor: float = 1e-30) -> NDArray:
    """Scale decoder rows to a target norm."""

    d = validate_decoder_matrix(decoder)
    norms = np.maximum(decoder_row_norms(d), float(floor))
    return d * (float(target_norm) / norms)[:, None]


def decoder_health_report(
    decoder: ArrayLike,
    *,
    loudspeaker_grid: SphericalGrid | None = None,
    loudspeaker_basis: ArrayLike | None = None,
    probe_basis: ArrayLike | None = None,
    target_grid: SphericalGrid | None = None,
) -> dict[str, object]:
    """Compact machine-readable report for a decoder matrix."""

    d = validate_decoder_matrix(decoder)
    report: dict[str, object] = {
        "n_speakers": int(d.shape[0]),
        "n_coeffs": int(d.shape[1]),
        "max_order": infer_decoder_order(d),
        "rank": decoder_rank(d),
        "condition_number": decoder_condition_number(d),
        "frobenius_norm": decoder_frobenius_norm(d),
        "energy_preservation_error": decoder_energy_preservation_error(d),
        "mode_leakage_ratio": decoder_mode_leakage_ratio(d),
        "loudspeaker_gain_spread_db": decoder_loudspeaker_gain_spread_db(d),
        "mode_gain_spread_db": decoder_mode_gain_spread_db(d),
    }
    if loudspeaker_basis is not None:
        report["mode_matching_error"] = mode_matching_error(d, loudspeaker_basis)
        report["mode_response_leakage_ratio"] = mode_response_leakage_ratio(d, loudspeaker_basis)
    if loudspeaker_grid is not None and probe_basis is not None:
        evec = probe_energy_vector(d, probe_basis, loudspeaker_grid)
        vvec = probe_velocity_vector(d, probe_basis, loudspeaker_grid)
        report["energy_vector_magnitude_mean"] = float(np.mean(vector_magnitudes(evec)))
        report["velocity_vector_magnitude_mean"] = float(np.mean(vector_magnitudes(vvec)))
        if target_grid is not None:
            report["energy_vector_error_deg_mean"] = float(np.nanmean(probe_energy_vector_errors_deg(d, probe_basis, loudspeaker_grid, target_grid)))
            report["velocity_vector_error_deg_mean"] = float(np.nanmean(probe_velocity_vector_errors_deg(d, probe_basis, loudspeaker_grid, target_grid)))
    return report


__all__ = [
    "decoder_column_correlation",
    "decoder_column_norms",
    "decoder_column_power",
    "decoder_condition_number",
    "decoder_diffuse_loudspeaker_levels_db",
    "decoder_energy_matrix",
    "decoder_energy_preservation_error",
    "decoder_frobenius_norm",
    "decoder_health_report",
    "decoder_loudspeaker_gain_spread_db",
    "decoder_mode_gain_spread_db",
    "decoder_mode_leakage_ratio",
    "decoder_projection_matrix",
    "decoder_rank",
    "decoder_row_correlation",
    "decoder_row_norms",
    "decoder_row_power",
    "decoder_singular_values",
    "infer_decoder_order",
    "mode_matching_error",
    "mode_matching_matrix",
    "mode_response_diagonal",
    "mode_response_leakage_ratio",
    "normalize_decoder_column_norms",
    "normalize_decoder_row_norms",
    "probe_energy_vector",
    "probe_energy_vector_errors_deg",
    "probe_response",
    "probe_response_energy",
    "probe_response_peak_speaker",
    "probe_velocity_vector",
    "probe_velocity_vector_errors_deg",
    "speaker_directions_cartesian",
    "validate_decoder_matrix",
    "vector_angle_errors_deg",
    "vector_magnitudes",
]
