"""JSON-friendly interchange helpers for package data containers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..layouts import LoudspeakerLayout
from ..measurement import ArrayMeasurement
from ..types import ArrayGeometry, SHBasisSpec, SphericalGrid


SCHEMA_VERSION = "spherical-array-processing.interop.v1"


def ndarray_to_list(array: ArrayLike) -> list[Any]:
    """Convert a NumPy-compatible array to nested Python lists."""

    return np.asarray(array).tolist()


def complex_array_to_dict(array: ArrayLike) -> dict[str, Any]:
    """Represent a complex array as JSON-safe real and imaginary lists."""

    arr = np.asarray(array, dtype=np.complex128)
    return {
        "dtype": "complex128",
        "shape": list(arr.shape),
        "real": arr.real.tolist(),
        "imag": arr.imag.tolist(),
    }


def complex_array_from_dict(payload: dict[str, Any]) -> NDArray[np.complex128]:
    """Decode :func:`complex_array_to_dict` output."""

    real = np.asarray(payload["real"], dtype=float)
    imag = np.asarray(payload["imag"], dtype=float)
    out = real + 1j * imag
    shape = tuple(int(x) for x in payload.get("shape", out.shape))
    return np.asarray(out, dtype=np.complex128).reshape(shape)


def grid_to_dict(grid: SphericalGrid) -> dict[str, Any]:
    """Serialize a :class:`SphericalGrid` to a JSON-safe dictionary."""

    return {
        "schema": f"{SCHEMA_VERSION}.grid",
        "azimuth": np.asarray(grid.azimuth, dtype=float).tolist(),
        "angle2": np.asarray(grid.angle2, dtype=float).tolist(),
        "weights": None if grid.weights is None else np.asarray(grid.weights, dtype=float).tolist(),
        "convention": grid.convention,
    }


def grid_from_dict(payload: dict[str, Any]) -> SphericalGrid:
    """Deserialize a :class:`SphericalGrid`."""

    return SphericalGrid(
        azimuth=np.asarray(payload["azimuth"], dtype=float),
        angle2=np.asarray(payload["angle2"], dtype=float),
        weights=None if payload.get("weights") is None else np.asarray(payload["weights"], dtype=float),
        convention=payload.get("convention", "az_el"),
    )


def sh_basis_spec_to_dict(spec: SHBasisSpec) -> dict[str, Any]:
    """Serialize a spherical-harmonic basis spec."""

    return {
        "schema": f"{SCHEMA_VERSION}.sh-basis",
        "max_order": int(spec.max_order),
        "basis": spec.basis,
        "normalization": spec.normalization,
        "angle_convention": spec.angle_convention,
        "channel_order": spec.channel_order,
    }


def sh_basis_spec_from_dict(payload: dict[str, Any]) -> SHBasisSpec:
    """Deserialize a spherical-harmonic basis spec."""

    return SHBasisSpec(
        max_order=int(payload["max_order"]),
        basis=payload.get("basis", "complex"),
        normalization=payload.get("normalization", "orthonormal"),
        angle_convention=payload.get("angle_convention", "az_colat"),
        channel_order=payload.get("channel_order", "acn"),
    )


def array_geometry_to_dict(array: ArrayGeometry) -> dict[str, Any]:
    """Serialize :class:`ArrayGeometry`."""

    return {
        "schema": f"{SCHEMA_VERSION}.array-geometry",
        "radius_m": float(array.radius_m),
        "sensor_grid": grid_to_dict(array.sensor_grid),
        "array_type": array.array_type,
        "sensor_kind": array.sensor_kind,
        "metadata": dict(array.metadata),
    }


def array_geometry_from_dict(payload: dict[str, Any]) -> ArrayGeometry:
    """Deserialize :class:`ArrayGeometry`."""

    return ArrayGeometry(
        radius_m=float(payload["radius_m"]),
        sensor_grid=grid_from_dict(payload["sensor_grid"]),
        array_type=payload.get("array_type", "rigid"),
        sensor_kind=payload.get("sensor_kind"),
        metadata=dict(payload.get("metadata", {})),
    )


def loudspeaker_layout_to_dict(layout: LoudspeakerLayout) -> dict[str, Any]:
    """Serialize :class:`LoudspeakerLayout`."""

    return {
        "schema": f"{SCHEMA_VERSION}.loudspeaker-layout",
        "directions_rad": np.asarray(layout.directions_rad, dtype=float).tolist(),
        "labels": None if layout.labels is None else list(layout.labels),
        "convention": layout.convention,
        "weights": None if layout.weights is None else np.asarray(layout.weights, dtype=float).tolist(),
        "metadata": dict(layout.metadata),
    }


def loudspeaker_layout_from_dict(payload: dict[str, Any]) -> LoudspeakerLayout:
    """Deserialize :class:`LoudspeakerLayout`."""

    labels = payload.get("labels")
    return LoudspeakerLayout(
        directions_rad=np.asarray(payload["directions_rad"], dtype=float),
        labels=None if labels is None else tuple(labels),
        convention=payload.get("convention", "az_el"),
        weights=None if payload.get("weights") is None else np.asarray(payload["weights"], dtype=float),
        metadata=dict(payload.get("metadata", {})),
    )


def measurement_to_dict(measurement: ArrayMeasurement, *, include_transfer: bool = True) -> dict[str, Any]:
    """Serialize an :class:`ArrayMeasurement`."""

    payload: dict[str, Any] = {
        "schema": f"{SCHEMA_VERSION}.array-measurement",
        "frequencies_hz": np.asarray(measurement.frequencies_hz, dtype=float).tolist(),
        "array": None if measurement.array is None else array_geometry_to_dict(measurement.array),
        "source_grid": None if measurement.source_grid is None else grid_to_dict(measurement.source_grid),
        "sample_rate_hz": measurement.sample_rate_hz,
        "metadata": dict(measurement.metadata),
    }
    if include_transfer:
        payload["transfer"] = complex_array_to_dict(measurement.transfer)
    else:
        payload["transfer_shape"] = list(measurement.transfer.shape)
    return payload


def measurement_from_dict(payload: dict[str, Any]) -> ArrayMeasurement:
    """Deserialize an :class:`ArrayMeasurement` with transfer data."""

    if "transfer" not in payload:
        raise ValueError("payload does not include transfer data")
    return ArrayMeasurement(
        frequencies_hz=np.asarray(payload["frequencies_hz"], dtype=float),
        transfer=complex_array_from_dict(payload["transfer"]),
        array=None if payload.get("array") is None else array_geometry_from_dict(payload["array"]),
        source_grid=None if payload.get("source_grid") is None else grid_from_dict(payload["source_grid"]),
        sample_rate_hz=payload.get("sample_rate_hz"),
        metadata=dict(payload.get("metadata", {})),
    )


def write_json_payload(payload: dict[str, Any], path: str | Path, *, indent: int = 2) -> None:
    """Write a JSON payload to disk."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
        f.write("\n")


def read_json_payload(path: str | Path) -> dict[str, Any]:
    """Read a JSON payload from disk."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def object_to_dict(obj: Any) -> dict[str, Any]:
    """Serialize known package containers by type."""

    if isinstance(obj, SphericalGrid):
        return grid_to_dict(obj)
    if isinstance(obj, SHBasisSpec):
        return sh_basis_spec_to_dict(obj)
    if isinstance(obj, ArrayGeometry):
        return array_geometry_to_dict(obj)
    if isinstance(obj, LoudspeakerLayout):
        return loudspeaker_layout_to_dict(obj)
    if isinstance(obj, ArrayMeasurement):
        return measurement_to_dict(obj)
    raise TypeError(f"unsupported object type: {type(obj)!r}")


def object_from_dict(payload: dict[str, Any]) -> Any:
    """Deserialize known package containers based on their schema field."""

    schema = str(payload.get("schema", ""))
    if schema.endswith(".grid"):
        return grid_from_dict(payload)
    if schema.endswith(".sh-basis"):
        return sh_basis_spec_from_dict(payload)
    if schema.endswith(".array-geometry"):
        return array_geometry_from_dict(payload)
    if schema.endswith(".loudspeaker-layout"):
        return loudspeaker_layout_from_dict(payload)
    if schema.endswith(".array-measurement"):
        return measurement_from_dict(payload)
    raise ValueError(f"unsupported schema: {schema!r}")


def write_object(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    """Serialize a known package object directly to JSON."""

    write_json_payload(object_to_dict(obj), path, indent=indent)


def read_object(path: str | Path) -> Any:
    """Read a known package object from JSON."""

    return object_from_dict(read_json_payload(path))


def validate_schema(payload: dict[str, Any], suffix: str) -> None:
    """Validate that a payload carries the expected schema suffix."""

    schema = str(payload.get("schema", ""))
    if not schema.startswith(SCHEMA_VERSION) or not schema.endswith(suffix):
        raise ValueError(f"expected schema suffix {suffix!r}, got {schema!r}")


__all__ = [
    "SCHEMA_VERSION",
    "array_geometry_from_dict",
    "array_geometry_to_dict",
    "complex_array_from_dict",
    "complex_array_to_dict",
    "grid_from_dict",
    "grid_to_dict",
    "loudspeaker_layout_from_dict",
    "loudspeaker_layout_to_dict",
    "measurement_from_dict",
    "measurement_to_dict",
    "ndarray_to_list",
    "object_from_dict",
    "object_to_dict",
    "read_json_payload",
    "read_object",
    "sh_basis_spec_from_dict",
    "sh_basis_spec_to_dict",
    "validate_schema",
    "write_json_payload",
    "write_object",
]
