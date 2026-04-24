"""Tests for SOFA ``SimpleFreeFieldHRIR`` writer and round-trip."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from spherical_array_processing.array import fibonacci_grid
from spherical_array_processing.hrtf import (
    HRTFDataset,
    load_sofa,
    rigid_sphere_hrtf,
    save_sofa,
)
from spherical_array_processing.types import SphericalGrid


def _small_synthetic_dataset(n_dirs: int = 16) -> HRTFDataset:
    head_r = 0.085
    ears = np.array([[0.0, head_r, 0.0], [0.0, -head_r, 0.0]])
    grid = fibonacci_grid(n_dirs)
    return rigid_sphere_hrtf(head_r, ears, grid, 48000.0, 128, max_order=8)


class TestSofaWriter:
    def test_round_trip_preserves_all_fields(self):
        ds = _small_synthetic_dataset(24)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rt.sofa")
            save_sofa(ds, path, database_name="rigid-sphere")
            loaded = load_sofa(path)
        np.testing.assert_allclose(loaded.hrirs, ds.hrirs, atol=1e-12)
        assert loaded.fs == ds.fs
        np.testing.assert_allclose(
            loaded.ear_positions_m, ds.ear_positions_m, atol=1e-12,
        )
        # Grids match within floating-point noise (degree/radian conversion).
        np.testing.assert_allclose(
            loaded.source_grid.azimuth, ds.source_grid.azimuth, atol=1e-12,
        )
        np.testing.assert_allclose(
            loaded.source_grid.elevation, ds.source_grid.elevation, atol=1e-12,
        )
        assert loaded.metadata.get("DatabaseName") == "rigid-sphere"

    def test_writes_mandatory_groups(self):
        ds = _small_synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mand.sofa")
            save_sofa(ds, path)
            with h5py.File(path, "r") as fh:
                for key in (
                    "Data.IR",
                    "Data.SamplingRate",
                    "Data.Delay",
                    "SourcePosition",
                    "ReceiverPosition",
                    "ListenerPosition",
                    "ListenerView",
                    "ListenerUp",
                    "EmitterPosition",
                ):
                    assert key in fh, f"missing mandatory key {key}"
                assert fh.attrs["SOFAConventions"] == b"SimpleFreeFieldHRIR"
                assert fh.attrs["Version"] == b"2.1"
                assert "AuthorContact" in fh.attrs
                assert "Organization" in fh.attrs
                assert "ListenerShortName" in fh.attrs

    def test_matches_simplefreefieldhrir_listenerup_and_delay_convention(self):
        ds = _small_synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "conv.sofa")
            save_sofa(ds, path)
            with h5py.File(path, "r") as fh:
                assert "Type" not in fh["ListenerUp"].attrs
                assert "Units" not in fh["ListenerUp"].attrs
                assert fh["Data.Delay"].dtype == np.float64

    def test_stores_ear_positions_as_receiver_2_3_1(self):
        ds = _small_synthetic_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rp.sofa")
            save_sofa(ds, path)
            with h5py.File(path, "r") as fh:
                rp = fh["ReceiverPosition"][...]
        assert rp.shape == (2, 3, 1)

    def test_dataset_without_ears_writes_zeros(self):
        # Build a dataset with ear_positions_m=None.
        rng = np.random.default_rng(0)
        grid = fibonacci_grid(8)
        hrirs = rng.standard_normal((8, 2, 64)) * 0.01
        ds = HRTFDataset(
            hrirs=hrirs, fs=16000.0, source_grid=grid,
            ear_positions_m=None,
            metadata={"DatabaseName": "no-ears"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no_ears.sofa")
            save_sofa(ds, path)
            with h5py.File(path, "r") as fh:
                rp = fh["ReceiverPosition"][...]
        np.testing.assert_allclose(rp, 0.0, atol=0)

    def test_extra_metadata_round_trips(self):
        ds = _small_synthetic_dataset()
        ds_with = HRTFDataset(
            hrirs=ds.hrirs,
            fs=ds.fs,
            source_grid=ds.source_grid,
            ear_positions_m=ds.ear_positions_m,
            metadata={
                "DatabaseName": "test",
                "Origin": "synthetic",
                "Custom": "value-123",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meta.sofa")
            save_sofa(ds_with, path)
            loaded = load_sofa(path)
        assert loaded.metadata["Origin"] == "synthetic"
        assert loaded.metadata["Custom"] == "value-123"

    def test_data_delay_per_ear_round_trip(self):
        ds = _small_synthetic_dataset(8)
        ds_with_delay = HRTFDataset(
            hrirs=ds.hrirs, fs=ds.fs, source_grid=ds.source_grid,
            ear_positions_m=ds.ear_positions_m,
            data_delay_samples=np.array([3.0, 5.0]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "delay.sofa")
            save_sofa(ds_with_delay, path)
            loaded = load_sofa(path)
        assert loaded.data_delay_samples is not None
        np.testing.assert_allclose(
            loaded.data_delay_samples, np.array([3.0, 5.0]), atol=0,
        )

    def test_data_delay_per_direction_round_trip(self):
        ds = _small_synthetic_dataset(12)
        n_dirs = ds.hrirs.shape[0]
        per_dir_delay = np.stack(
            [np.arange(n_dirs, dtype=float),
             np.arange(n_dirs, dtype=float) + 10.0],
            axis=-1,
        )
        ds_per = HRTFDataset(
            hrirs=ds.hrirs, fs=ds.fs, source_grid=ds.source_grid,
            ear_positions_m=ds.ear_positions_m,
            data_delay_samples=per_dir_delay,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "delay_dir.sofa")
            save_sofa(ds_per, path)
            loaded = load_sofa(path)
        assert loaded.data_delay_samples is not None
        assert loaded.data_delay_samples.shape == (n_dirs, 2)
        np.testing.assert_allclose(
            loaded.data_delay_samples, per_dir_delay, atol=0,
        )

    def test_no_delay_load_returns_none(self):
        ds = _small_synthetic_dataset(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nodelay.sofa")
            save_sofa(ds, path)
            loaded = load_sofa(path)
        assert loaded.data_delay_samples is None

    def test_preserve_zero_delay_keeps_explicit_zero_block(self):
        """``preserve_zero_delay=True`` must keep the explicit zero
        block so the dataset can round-trip the file byte-faithfully."""
        ds = _small_synthetic_dataset(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "zero.sofa")
            save_sofa(ds, path)
            loaded = load_sofa(path, preserve_zero_delay=True)
        assert loaded.data_delay_samples is not None
        assert loaded.data_delay_samples.shape == (2,)
        np.testing.assert_array_equal(loaded.data_delay_samples, 0.0)

    def test_explicit_zero_delay_round_trip(self):
        """Write an explicit zero-delay vector, read it back with
        ``preserve_zero_delay=True``, confirm byte-identical result."""
        ds = _small_synthetic_dataset(6)
        ds_zero = HRTFDataset(
            hrirs=ds.hrirs, fs=ds.fs, source_grid=ds.source_grid,
            ear_positions_m=ds.ear_positions_m,
            data_delay_samples=np.zeros(2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "zero_rt.sofa")
            round_trip_path = os.path.join(tmp, "zero_rt_roundtrip.sofa")
            save_sofa(ds_zero, path)
            loaded = load_sofa(path, preserve_zero_delay=True)
            save_sofa(loaded, round_trip_path)
            with h5py.File(path, "r") as fh:
                assert fh["Data.Delay"].shape == (1, 2)
            with h5py.File(round_trip_path, "r") as fh:
                assert fh["Data.Delay"].shape == (1, 2)
            with open(path, "rb") as fh:
                original_bytes = fh.read()
            with open(round_trip_path, "rb") as fh:
                round_trip_bytes = fh.read()
        assert loaded.data_delay_samples is not None
        assert loaded.data_delay_samples.shape == (2,)
        np.testing.assert_array_equal(
            loaded.data_delay_samples, np.zeros(2),
        )
        assert round_trip_bytes == original_bytes

    def test_default_load_still_folds_zero_to_none(self):
        """Backward compat: default loader behaviour is unchanged."""
        ds = _small_synthetic_dataset(4)
        ds_zero = HRTFDataset(
            hrirs=ds.hrirs, fs=ds.fs, source_grid=ds.source_grid,
            ear_positions_m=ds.ear_positions_m,
            data_delay_samples=np.zeros(2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "compat.sofa")
            save_sofa(ds_zero, path)
            loaded = load_sofa(path)
        assert loaded.data_delay_samples is None

    def test_default_load_still_folds_per_direction_zero_to_none(self):
        """Backward compat also covers explicit all-zero ``(M, 2)`` blocks."""
        ds = _small_synthetic_dataset(5)
        ds_zero = HRTFDataset(
            hrirs=ds.hrirs, fs=ds.fs, source_grid=ds.source_grid,
            ear_positions_m=ds.ear_positions_m,
            data_delay_samples=np.zeros((ds.hrirs.shape[0], 2)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "compat_m.sofa")
            save_sofa(ds_zero, path)
            loaded = load_sofa(path)
        assert loaded.data_delay_samples is None

    def test_default_load_rejects_malformed_zero_delay_block(self):
        """A bad all-zero block must still fail validation, not fold to ``None``."""
        ds = _small_synthetic_dataset(6)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_zero_delay.sofa")
            save_sofa(ds, path)
            with h5py.File(path, "a") as fh:
                del fh["Data.Delay"]
                fh.create_dataset("Data.Delay", data=np.zeros((3, 2)))
            with pytest.raises(ValueError, match=r"Data\.Delay"):
                load_sofa(path)

    def test_rejects_wrong_delay_shape(self):
        ds = _small_synthetic_dataset(4)
        bad = HRTFDataset(
            hrirs=ds.hrirs, fs=ds.fs, source_grid=ds.source_grid,
            data_delay_samples=np.zeros((5,)),  # neither (2,) nor (M,2)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_delay.sofa")
            with pytest.raises(ValueError, match="data_delay"):
                save_sofa(bad, path)

    def test_rejects_wrong_hrir_shape(self):
        grid = fibonacci_grid(4)
        ds = HRTFDataset(
            hrirs=np.zeros((4, 3, 64)),  # 3 ears — invalid.
            fs=16000.0, source_grid=grid,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.sofa")
            with pytest.raises(ValueError, match="shape"):
                save_sofa(ds, path)

    def test_rejects_wrong_ear_shape(self):
        ds = _small_synthetic_dataset()
        bad = HRTFDataset(
            hrirs=ds.hrirs,
            fs=ds.fs,
            source_grid=ds.source_grid,
            ear_positions_m=np.zeros((3, 3)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_ears.sofa")
            with pytest.raises(ValueError, match=r"\(2, 3\)"):
                save_sofa(bad, path)
