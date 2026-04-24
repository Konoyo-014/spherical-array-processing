"""Tests for the `spherical_array_processing.stft` helper."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.stft import istft, stft


class TestStftHelper:
    def test_mono_shape(self):
        fs = 16000.0
        x = np.random.default_rng(0).normal(size=int(fs))
        freqs, times, Z = stft(x, fs, nperseg=512)
        assert Z.ndim == 2
        assert Z.shape[0] == 257  # 512/2 + 1
        assert freqs.shape == (257,)

    def test_multichannel_layout_is_f_m_t(self):
        """The wrapper must produce ``(F, M, T)`` for compatibility with
        :func:`doa.srp_map` regardless of scipy's internal axis order.
        """
        fs = 16000.0
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, int(fs / 2)))  # 5 channels, 0.5 s
        freqs, times, Z = stft(X, fs, nperseg=512)
        assert Z.shape == (freqs.size, 5, times.size)

    def test_mono_roundtrip(self):
        fs = 8000.0
        rng = np.random.default_rng(1)
        x = rng.normal(size=int(fs))
        _, _, Z = stft(x, fs, nperseg=256)
        _, x_back = istft(Z, fs, nperseg=256)
        # COLA is inherent to the Hann window + 50 % overlap — recon
        # should be exact on the overlapping interior.
        assert_allclose(x_back[: len(x)], x, atol=1e-10)

    def test_multichannel_roundtrip(self):
        fs = 8000.0
        rng = np.random.default_rng(2)
        X = rng.normal(size=(3, int(fs)))
        _, _, Z = stft(X, fs, nperseg=256)
        _, X_back = istft(Z, fs, nperseg=256)
        assert_allclose(X_back[:, : X.shape[1]], X, atol=1e-10)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            stft(np.zeros((2, 3, 4)), 16000.0)
        with pytest.raises(ValueError, match="2-D.*3-D"):
            istft(np.zeros((5,)), 16000.0)
