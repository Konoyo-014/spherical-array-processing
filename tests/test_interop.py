"""Tests for JSON-friendly interchange helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spherical_array_processing as sap


def test_grid_and_basis_spec_round_trip():
    grid = sap.array.fibonacci_grid(8)
    payload = sap.interop.grid_to_dict(grid)
    sap.interop.validate_schema(payload, ".grid")
    restored = sap.interop.grid_from_dict(payload)
    assert_allclose(restored.azimuth, grid.azimuth)
    assert_allclose(restored.angle2, grid.angle2)
    assert_allclose(restored.weights, grid.weights)
    assert restored.convention == grid.convention

    spec = sap.SHBasisSpec(max_order=3, basis="real", normalization="sn3d", angle_convention="az_el")
    restored_spec = sap.interop.sh_basis_spec_from_dict(sap.interop.sh_basis_spec_to_dict(spec))
    assert restored_spec.max_order == 3
    assert restored_spec.basis == "real"
    assert restored_spec.normalization == "sn3d"
    assert restored_spec.angle_convention == "az_el"


def test_array_layout_and_measurement_round_trip(tmp_path):
    sensor_grid = sap.SphericalGrid(
        azimuth=np.array([0.0, np.pi]),
        angle2=np.array([0.0, 0.0]),
        convention="az_el",
    )
    array = sap.ArrayGeometry(radius_m=0.042, sensor_grid=sensor_grid, array_type="rigid")
    layout = sap.layouts.stereo_layout()
    freqs = np.array([1000.0, 2000.0])
    transfer = np.array(
        [
            [[1.0 + 0.0j, 0.5 + 0.25j], [0.8 - 0.1j, 0.4 + 0.2j]],
            [[0.9 + 0.1j, 0.7 + 0.0j], [0.6 - 0.2j, 0.3 + 0.1j]],
        ]
    )
    measurement = sap.measurement.ArrayMeasurement(
        frequencies_hz=freqs,
        transfer=transfer,
        array=array,
        source_grid=layout.as_grid(),
        sample_rate_hz=48000.0,
        metadata={"source": "unit-test"},
    )

    array_restored = sap.interop.array_geometry_from_dict(sap.interop.array_geometry_to_dict(array))
    assert_allclose(array_restored.sensor_grid.azimuth, sensor_grid.azimuth)
    assert array_restored.radius_m == array.radius_m

    layout_restored = sap.interop.loudspeaker_layout_from_dict(sap.interop.loudspeaker_layout_to_dict(layout))
    assert_allclose(layout_restored.directions_rad, layout.directions_rad)
    assert layout_restored.labels == layout.labels

    path = tmp_path / "measurement.json"
    sap.interop.write_object(measurement, path)
    restored = sap.interop.read_object(path)
    assert isinstance(restored, sap.measurement.ArrayMeasurement)
    assert_allclose(restored.frequencies_hz, freqs)
    assert_allclose(restored.transfer, transfer)
    assert restored.metadata["source"] == "unit-test"


def test_complex_array_and_payload_helpers(tmp_path):
    data = np.array([[1.0 + 2.0j, 3.0 - 4.0j]])
    payload = sap.interop.complex_array_to_dict(data)
    assert_allclose(sap.interop.complex_array_from_dict(payload), data)
    assert sap.interop.ndarray_to_list(np.array([1, 2, 3])) == [1, 2, 3]

    path = tmp_path / "payload.json"
    sap.interop.write_json_payload({"schema": "custom", "values": [1, 2]}, path)
    assert sap.interop.read_json_payload(path)["values"] == [1, 2]


def test_interop_validation_errors():
    with pytest.raises(TypeError, match="unsupported"):
        sap.interop.object_to_dict(object())
    with pytest.raises(ValueError, match="unsupported schema"):
        sap.interop.object_from_dict({"schema": "unknown"})
    with pytest.raises(ValueError, match="expected schema"):
        sap.interop.validate_schema({"schema": sap.interop.SCHEMA_VERSION + ".grid"}, ".array")
    with pytest.raises(ValueError, match="transfer"):
        sap.interop.measurement_from_dict({"schema": sap.interop.SCHEMA_VERSION + ".array-measurement"})
