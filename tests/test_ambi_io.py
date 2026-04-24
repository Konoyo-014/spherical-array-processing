"""Tests for `spherical_array_processing.ambi.io` — AmbiX WAV I/O."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

from spherical_array_processing.ambi import (
    convert_ambi_normalization,
    read_ambix_wav,
    write_ambix_wav,
)


class TestAmbixRoundTrip:
    @pytest.mark.parametrize("N", [0, 1, 2, 3, 4])
    def test_shape_and_fs_round_trip(self, N):
        rng = np.random.default_rng(N)
        Q = (N + 1) ** 2
        T = 4096
        sig = (rng.standard_normal((Q, T)) * 0.05).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"order_{N}.wav")
            write_ambix_wav(path, sig, fs=16000.0)
            loaded, fs = read_ambix_wav(path)
        assert loaded.shape == (Q, T)
        assert fs == 16000.0
        np.testing.assert_allclose(loaded, sig, atol=1e-6)

    def test_channels_last_layout_round_trip(self):
        rng = np.random.default_rng(42)
        N = 2
        Q = (N + 1) ** 2
        sig_tq = (rng.standard_normal((8000, Q)) * 0.1).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cl.wav")
            write_ambix_wav(path, sig_tq, fs=48000.0, axis="channels_last")
            loaded, fs = read_ambix_wav(path, axis="channels_last")
        assert loaded.shape == sig_tq.shape
        np.testing.assert_allclose(loaded, sig_tq, atol=1e-6)

    def test_normalization_conversion_on_write_and_read(self):
        """Write from ortho, tag file as SN3D, read back as ortho — full round trip."""
        rng = np.random.default_rng(7)
        N = 3
        Q = (N + 1) ** 2
        sig_ortho = (rng.standard_normal((Q, 5000)) * 0.05).astype(np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ortho_to_sn3d.wav")
            write_ambix_wav(
                path, sig_ortho, fs=16000.0,
                source_normalization="orthonormal",
                file_normalization="sn3d",
            )
            # Read the raw file to confirm SN3D scaling is present.
            raw_sn3d_tq, _ = soundfile.read(path, always_2d=True)
            expected_sn3d = convert_ambi_normalization(
                sig_ortho.T, max_order=N,
                from_="orthonormal", to="sn3d", axis=1,
            )
            np.testing.assert_allclose(raw_sn3d_tq, expected_sn3d, atol=1e-5)

            loaded, _ = read_ambix_wav(
                path, normalization="sn3d", target_normalization="orthonormal",
            )
        np.testing.assert_allclose(loaded, sig_ortho, atol=1e-6)

    def test_normalization_unchanged_when_source_matches_file(self):
        """When source_normalization == file_normalization, the stored
        samples should equal the input exactly (up to WAV quantisation)."""
        rng = np.random.default_rng(5)
        Q = 9
        sig = (rng.standard_normal((Q, 2000)) * 0.05).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "same_norm.wav")
            write_ambix_wav(
                path, sig, fs=16000.0,
                source_normalization="sn3d", file_normalization="sn3d",
            )
            raw, _ = soundfile.read(path, always_2d=True)
            np.testing.assert_allclose(raw, sig.T, atol=1e-6)

    def test_double_subtype_preserves_float64_samples(self):
        rng = np.random.default_rng(9)
        sig = (rng.standard_normal((4, 256)) * 1e-6).astype(np.float64)
        sig += np.linspace(0.0, 1e-12, sig.size, dtype=np.float64).reshape(sig.shape)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "double.wav")
            write_ambix_wav(
                path, sig, fs=48000.0, subtype="DOUBLE",
                source_normalization="orthonormal",
                file_normalization="orthonormal",
            )
            assert soundfile.info(path).subtype == "DOUBLE"
            raw, _ = soundfile.read(path, always_2d=True, dtype="float64")
        np.testing.assert_allclose(raw, sig.T, atol=0.0, rtol=0.0)


class TestAmbixValidation:
    def test_max_order_mismatch_raises(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal((9, 1000)).astype(np.float32) * 0.05
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_order.wav")
            write_ambix_wav(path, sig, fs=16000.0)
            with pytest.raises(ValueError, match="max_order"):
                read_ambix_wav(path, max_order=3)  # file is order 2

    def test_bad_channel_count_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weird.wav")
            # 5 channels is not an (N+1)² count.
            soundfile.write(
                path, np.zeros((1000, 5), dtype=np.float32), 16000,
                subtype="FLOAT",
            )
            with pytest.raises(ValueError, match="1, 4, 9"):
                read_ambix_wav(path)

    def test_non_2d_write_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nd.wav")
            with pytest.raises(ValueError, match="2-D"):
                write_ambix_wav(path, np.zeros((3, 4, 5)), fs=16000.0)

    def test_bad_axis_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.wav")
            with pytest.raises(ValueError, match="axis"):
                write_ambix_wav(
                    path, np.zeros((4, 100)), fs=16000.0, axis="bad",
                )

    def test_read_bad_axis_literal(self):
        rng = np.random.default_rng(8)
        sig = (rng.standard_normal((4, 1000)) * 0.05).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_read_axis.wav")
            write_ambix_wav(path, sig, fs=16000.0)
            with pytest.raises(ValueError, match="axis"):
                read_ambix_wav(path, axis="bad")
