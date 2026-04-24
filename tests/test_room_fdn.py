"""Tests for the FDN-based diffuse reverberator."""

from __future__ import annotations

import numpy as np
import pytest

from spherical_array_processing.room import (
    fdn_reverb,
    fdn_sh_tail,
    reverberation_time,
)


class TestFdnReverb:
    def test_output_shape_default(self):
        fs = 16000.0
        dry = np.zeros(100); dry[0] = 1.0
        out = fdn_reverb(dry, rt60_s=0.5, fs=fs)
        assert out.shape == (100 + int(round(2.0 * 0.5 * fs)),)

    def test_impulse_rt60_matches_target(self):
        fs = 16000.0
        impulse = np.zeros(int(0.2 * fs))
        impulse[0] = 1.0
        for target_rt60 in [0.5, 1.0, 1.5]:
            rir = fdn_reverb(impulse, rt60_s=target_rt60, fs=fs)
            measured = reverberation_time(rir, fs, method="T30")
            assert abs(measured - target_rt60) / target_rt60 < 0.1

    def test_output_len_override(self):
        fs = 16000.0
        dry = np.zeros(50); dry[0] = 1.0
        out = fdn_reverb(dry, rt60_s=0.3, fs=fs, output_len=2000)
        assert out.shape == (2000,)

    def test_custom_delays(self):
        fs = 16000.0
        dry = np.zeros(50); dry[0] = 1.0
        out = fdn_reverb(
            dry, rt60_s=0.5, fs=fs,
            delays=np.array([521, 751, 997, 1217], dtype=np.int64),
        )
        assert out.size > 0
        # Should still decay.
        early = float(np.sum(out[:1000] ** 2))
        late = float(np.sum(out[-1000:] ** 2))
        assert late < early

    def test_rejects_empty_dry(self):
        with pytest.raises(ValueError, match="non-empty"):
            fdn_reverb(np.array([]), rt60_s=0.5, fs=16000.0)

    def test_rejects_bad_rt60(self):
        with pytest.raises(ValueError, match="rt60"):
            fdn_reverb(np.zeros(10), rt60_s=-1.0, fs=16000.0)

    def test_rejects_bad_delays(self):
        with pytest.raises(ValueError, match="delays"):
            fdn_reverb(
                np.zeros(10), rt60_s=0.5, fs=16000.0,
                delays=np.array([100]),  # only one delay line
            )

    def test_rejects_wrong_mixing_shape(self):
        with pytest.raises(ValueError, match="mixing"):
            fdn_reverb(
                np.zeros(10), rt60_s=0.5, fs=16000.0,
                mixing_matrix=np.eye(4),  # wrong size (default is 8)
            )

    def test_rejects_non_orthogonal_mixing_matrix(self):
        """A matrix that isn't orthogonal breaks the RT60 calibration,
        so the default validation must reject it with a clear error."""
        # 8x8 with one off-diagonal entry — obviously not orthogonal.
        bad_mixing = np.eye(8)
        bad_mixing[0, 1] = 0.5
        with pytest.raises(ValueError, match="orthogonal"):
            fdn_reverb(
                np.zeros(50), rt60_s=0.5, fs=16000.0,
                mixing_matrix=bad_mixing,
            )

    def test_check_orthogonality_false_bypasses_the_gate(self):
        """Explicit opt-out lets experimenters use a non-unitary
        feedback matrix."""
        bad_mixing = np.eye(8)
        bad_mixing[0, 1] = 0.5
        out = fdn_reverb(
            np.zeros(50), rt60_s=0.5, fs=16000.0,
            mixing_matrix=bad_mixing, check_orthogonality=False,
        )
        assert out.shape == (50 + int(round(2.0 * 0.5 * 16000.0)),)

    def test_accepts_valid_orthogonal_matrix(self):
        """An explicit orthogonal matrix (random Q from QR) works."""
        rng = np.random.default_rng(123)
        q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        out = fdn_reverb(
            np.zeros(50), rt60_s=0.5, fs=16000.0, mixing_matrix=q,
        )
        assert out.size > 0


class TestFdnShTail:
    def test_shape(self):
        fs = 16000.0
        dry = np.zeros(100); dry[0] = 1.0
        sh = fdn_sh_tail(dry, rt60_s=0.5, fs=fs, max_order=3)
        assert sh.shape == (16, 100 + int(round(2.0 * 0.5 * fs)))

    def test_w_channel_rt60_matches_target(self):
        fs = 16000.0
        impulse = np.zeros(int(0.2 * fs)); impulse[0] = 1.0
        for target in [0.4, 1.0]:
            sh = fdn_sh_tail(impulse, rt60_s=target, fs=fs, max_order=2)
            rt_w = reverberation_time(sh[0], fs, method="T30")
            assert abs(rt_w - target) / target < 0.1

    def test_output_has_directional_energy(self):
        """The SH output must have non-zero energy in directional (n≥1)
        channels — a purely diffuse reverb has some statistical
        directional structure per block even when ⟨I⟩ is zero."""
        fs = 16000.0
        impulse = np.zeros(3000); impulse[0] = 1.0
        sh = fdn_sh_tail(impulse, rt60_s=0.3, fs=fs, max_order=1)
        energy_w = float(np.sum(sh[0] ** 2))
        energy_dir = float(np.sum(sh[1:] ** 2))
        assert energy_w > 0.0
        assert energy_dir > 0.0

    def test_seed_reproducibility(self):
        fs = 16000.0
        dry = np.random.default_rng(0).standard_normal(1024) * 0.1
        a = fdn_sh_tail(dry, rt60_s=0.3, fs=fs, max_order=1, seed=7)
        b = fdn_sh_tail(dry, rt60_s=0.3, fs=fs, max_order=1, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        fs = 16000.0
        dry = np.random.default_rng(0).standard_normal(1024) * 0.1
        a = fdn_sh_tail(dry, rt60_s=0.3, fs=fs, max_order=1, seed=1)
        b = fdn_sh_tail(dry, rt60_s=0.3, fs=fs, max_order=1, seed=2)
        assert not np.array_equal(a, b)

    def test_complex_basis_returns_complex_output(self):
        fs = 16000.0
        dry = np.zeros(128); dry[0] = 1.0
        sh = fdn_sh_tail(
            dry, rt60_s=0.3, fs=fs, max_order=1, basis="complex",
        )
        assert np.iscomplexobj(sh)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="non-empty"):
            fdn_sh_tail(np.array([]), rt60_s=0.3, fs=16000.0, max_order=1)
        with pytest.raises(ValueError, match="rt60"):
            fdn_sh_tail(np.zeros(10), rt60_s=0.0, fs=16000.0, max_order=1)
