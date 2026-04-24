"""Tests for the `spherical_array_processing.hrtf` SOFA reader."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.hrtf import HRTFDataset, load_sofa


h5py = pytest.importorskip("h5py")


def _write_synthetic_sofa(
    path, *, n_dirs=26, n_taps=64, fs=48000.0, include_ears=True
):
    """Write a minimal ``SimpleFreeFieldHRIR`` SOFA file for testing."""
    rng = np.random.default_rng(0)
    az = np.linspace(0, 360, n_dirs, endpoint=False)
    el = np.zeros(n_dirs)
    radius = np.ones(n_dirs)
    positions = np.stack([az, el, radius], axis=1)  # (M, 3)
    hrirs = rng.normal(size=(n_dirs, 2, n_taps)) * 0.1  # (M, R=2, N)
    with h5py.File(str(path), "w") as fh:
        fh.attrs["Conventions"] = "SOFA"
        fh.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
        fh.attrs["DataType"] = "FIR"
        fh.attrs["GLOBAL:Title"] = "synthetic test HRTF"
        ir_ds = fh.create_dataset("Data.IR", data=hrirs)
        ir_ds.attrs["ChannelOrdering"] = "left-right"
        fh.create_dataset("Data.SamplingRate", data=np.array([fs]))
        sp = fh.create_dataset("SourcePosition", data=positions)
        sp.attrs["Type"] = "spherical"
        sp.attrs["Units"] = "degree, degree, metre"
        if include_ears:
            # (R=2, C=3, I=1) per SOFA convention.
            ear = np.array(
                [[[-0.09], [0.0], [0.0]], [[0.09], [0.0], [0.0]]]
            )  # shape (2, 3, 1)
            fh.create_dataset("ReceiverPosition", data=ear)
    return hrirs, fs, positions


class TestSofaReader:
    def test_roundtrip_synthetic_file(self, tmp_path):
        path = tmp_path / "synthetic.sofa"
        hrirs, fs, positions = _write_synthetic_sofa(path)
        ds = load_sofa(path)
        assert isinstance(ds, HRTFDataset)
        assert_allclose(ds.hrirs, hrirs, atol=1e-12)
        assert_allclose(ds.fs, fs)
        assert ds.source_grid.size == positions.shape[0]
        assert ds.source_grid.convention == "az_el"
        # SOFA azimuth in degrees → radians in the grid.
        assert_allclose(
            ds.source_grid.azimuth,
            np.radians(positions[:, 0]) % (2 * np.pi),
            atol=1e-12,
        )
        assert ds.ear_positions_m is not None
        assert_allclose(
            ds.ear_positions_m,
            np.array([[-0.09, 0.0, 0.0], [0.09, 0.0, 0.0]]),
            atol=1e-12,
        )
        assert "Conventions" in ds.metadata

    def test_to_frequency_domain_shape(self, tmp_path):
        path = tmp_path / "synthetic.sofa"
        _write_synthetic_sofa(path, n_dirs=10, n_taps=32)
        ds = load_sofa(path)
        freqs, hrtfs = ds.to_frequency_domain()
        assert hrtfs.shape == (ds.n_taps // 2 + 1, ds.n_directions, 2)
        assert freqs.shape == (ds.n_taps // 2 + 1,)

    def test_to_frequency_domain_matches_numpy_rfft(self, tmp_path):
        path = tmp_path / "synthetic.sofa"
        hrirs, _, _ = _write_synthetic_sofa(path, n_dirs=5, n_taps=16)
        ds = load_sofa(path)
        _, hrtfs = ds.to_frequency_domain()
        expected = np.fft.rfft(hrirs, axis=-1).transpose(2, 0, 1)
        assert_allclose(hrtfs, expected, atol=1e-12)

    def test_rejects_unsupported_convention(self, tmp_path):
        path = tmp_path / "bad.sofa"
        with h5py.File(str(path), "w") as fh:
            fh.attrs["SOFAConventions"] = "SingleRoomDRIR"
            fh.create_dataset("Data.IR", data=np.zeros((1, 2, 4)))
            fh.create_dataset("Data.SamplingRate", data=np.array([44100.0]))
            fh.create_dataset("SourcePosition", data=np.zeros((1, 3)))
        with pytest.raises(ValueError, match="Unsupported SOFA convention"):
            load_sofa(path)

    def test_cartesian_source_positions(self, tmp_path):
        """Accept ``Type=cartesian`` source position blocks."""
        path = tmp_path / "cartesian.sofa"
        rng = np.random.default_rng(0)
        cart = np.array(
            [
                [1.0, 0.0, 0.0],  # front
                [0.0, 1.0, 0.0],  # left
                [0.0, 0.0, 1.0],  # up
            ]
        )
        hrirs = rng.normal(size=(3, 2, 8))
        with h5py.File(str(path), "w") as fh:
            fh.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
            fh.create_dataset("Data.IR", data=hrirs)
            fh.create_dataset("Data.SamplingRate", data=np.array([48000.0]))
            sp = fh.create_dataset("SourcePosition", data=cart)
            sp.attrs["Type"] = "cartesian"
            sp.attrs["Units"] = "metre, metre, metre"
        ds = load_sofa(path)
        # Front: az=0, el=0; left: az=π/2, el=0; up: az=0, el=π/2.
        assert_allclose(ds.source_grid.azimuth, [0.0, np.pi / 2, 0.0], atol=1e-12)
        assert_allclose(
            ds.source_grid.angle2, [0.0, 0.0, np.pi / 2], atol=1e-12
        )

    def test_handles_missing_receiver_position(self, tmp_path):
        path = tmp_path / "no_ears.sofa"
        _write_synthetic_sofa(path, include_ears=False)
        ds = load_sofa(path)
        assert ds.ear_positions_m is None

    def test_rejects_data_delay_with_wrong_direction_count(self, tmp_path):
        path = tmp_path / "bad_delay_shape.sofa"
        _write_synthetic_sofa(path, n_dirs=5, include_ears=False)
        with h5py.File(str(path), "a") as fh:
            fh.create_dataset("Data.Delay", data=np.ones((3, 2)))
        with pytest.raises(ValueError, match="Data.Delay"):
            load_sofa(path)

    def test_unknown_receiver_position_shape_drops_to_none(self, tmp_path):
        path = tmp_path / "weird_ears.sofa"
        _write_synthetic_sofa(path, include_ears=False)
        with h5py.File(str(path), "a") as fh:
            fh.create_dataset("ReceiverPosition", data=np.zeros((1, 2, 3)))
        ds = load_sofa(path)
        assert ds.ear_positions_m is None

    def test_frequency_domain_feeds_bimagls_without_reshape(self, tmp_path):
        """Regression: ``HRTFDataset.to_frequency_domain`` output can be
        passed straight to :func:`bimagls_binaural_filters` without any
        user-side transpose.  Nails down the (F, M, 2) contract that
        BiMagLS expects.
        """
        from spherical_array_processing.binaural import bimagls_binaural_filters

        path = tmp_path / "synthetic.sofa"
        _write_synthetic_sofa(
            path, n_dirs=48, n_taps=128, fs=16000.0, include_ears=True
        )
        ds = load_sofa(path)
        freqs, hrtfs = ds.to_frequency_domain()
        filters, delay_sh = bimagls_binaural_filters(
            hrtfs, freqs, ds.source_grid,
            max_order=2,
            ear_positions_m=ds.ear_positions_m,
        )
        assert filters.shape == (freqs.size, 3 * 3, 2)
        assert delay_sh.shape == (3 * 3, 2)

    def test_rejects_unknown_source_position_type(self, tmp_path):
        path = tmp_path / "bad_type.sofa"
        hrirs = np.zeros((1, 2, 8))
        with h5py.File(str(path), "w") as fh:
            fh.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
            fh.create_dataset("Data.IR", data=hrirs)
            fh.create_dataset("Data.SamplingRate", data=np.array([48000.0]))
            sp = fh.create_dataset("SourcePosition", data=np.array([[0.0, 0.0, 1.0]]))
            sp.attrs["Type"] = "cylindrical"
            sp.attrs["Units"] = "degree, degree, metre"

        with pytest.raises(ValueError, match="SourcePosition.Type"):
            load_sofa(path)
