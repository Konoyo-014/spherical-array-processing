"""Tests for `spherical_array_processing.room.metrics` (ISO 3382 family)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.room import (
    RIRMetrics,
    ShoeboxRoom,
    banded_rir_metrics,
    center_time,
    clarity,
    definition,
    direct_to_reverberant_ratio,
    early_decay_time,
    early_late_energy,
    energy_decay_curve,
    interaural_cross_correlation,
    lateral_energy_fraction,
    reverberation_time,
    reverberation_times,
    rir_metrics,
    shoebox_rir,
    strength,
    total_energy,
)


def _make_exp_decay_rir(fs: float, tau_s: float, duration_s: float, seed: int = 0):
    """Gaussian noise × exp(-t/τ) has energy decay exp(-2t/τ).

    Analytic RT60 = 3·τ·ln(10) ≈ 6.908·τ.
    """
    n = int(duration_s * fs)
    rng = np.random.default_rng(seed)
    t_samples = np.arange(n) / fs
    return rng.standard_normal(n) * np.exp(-t_samples / tau_s)


class TestEDC:
    def test_monotonically_decreasing(self):
        fs = 16000.0
        h = _make_exp_decay_rir(fs, 0.4, 2.0)
        edc = energy_decay_curve(h)
        assert edc.shape == h.shape
        # Monotone decrease (up to numerical precision).
        diffs = np.diff(edc)
        assert np.all(diffs <= 1e-10)
        # Starts at 0 dB.  The tail end is the log of the last sample's
        # energy divided by the total, which for a decaying noise burst
        # is typically tens of dB down but not necessarily -∞.
        assert_allclose(edc[0], 0.0, atol=1e-12)
        assert edc[-1] < -50

    def test_raises_on_zero_rir(self):
        with pytest.raises(ValueError, match="zero energy"):
            energy_decay_curve(np.zeros(100))


class TestReverberationTime:
    @pytest.mark.parametrize("tau_s,tol", [(0.3, 0.05), (0.5, 0.05), (1.0, 0.05)])
    def test_t30_matches_analytic(self, tau_s, tol):
        fs = 16000.0
        # Make the RIR long enough to capture 65 dB of decay with room
        # to spare, which for exp(-2t/τ) with RT60=3τ·ln(10) means
        # ~3.5 × expected_rt60.
        expected = 3 * tau_s * np.log(10)
        h = _make_exp_decay_rir(fs, tau_s, duration_s=3.5 * expected)
        rt = reverberation_time(h, fs, method="T30")
        assert abs(rt - expected) / expected < tol

    def test_t20_close_to_t30(self):
        fs = 16000.0
        h = _make_exp_decay_rir(fs, 0.4, 5.0)
        t20 = reverberation_time(h, fs, method="T20")
        t30 = reverberation_time(h, fs, method="T30")
        # Perfect exp decay: both should match within 5%.
        assert abs(t20 - t30) / t30 < 0.05

    def test_raises_when_edc_stays_flat(self):
        """An IR with all energy at the last sample produces a constant
        0 dB EDC that never reaches the T30 regression anchor."""
        fs = 16000.0
        h = np.zeros(1000)
        h[-1] = 1.0
        with pytest.raises(ValueError, match="does not reach"):
            reverberation_time(h, fs, method="T30")

    def test_invalid_method(self):
        h = _make_exp_decay_rir(16000.0, 0.3, 5.0)
        with pytest.raises(ValueError, match="method"):
            reverberation_time(h, 16000.0, method="T45")

    def test_invalid_fs(self):
        h = _make_exp_decay_rir(16000.0, 0.3, 1.0)
        with pytest.raises(ValueError, match="fs"):
            reverberation_time(h, -1.0)


class TestEarlyDecayTime:
    def test_matches_rt60_for_pure_exp_decay(self):
        """For a pure exp decay the slope in the first 10 dB equals
        the slope in any later window, so EDT ≈ RT60."""
        fs = 16000.0
        h = _make_exp_decay_rir(fs, 0.4, 4.0)
        rt = reverberation_time(h, fs)
        edt = early_decay_time(h, fs)
        assert abs(edt - rt) / rt < 0.1


class TestClarityAndDefinition:
    def test_clarity_of_pure_direct_path(self):
        """A single impulse at t=0 has infinite early/late ratio."""
        h = np.zeros(1000)
        h[0] = 1.0
        c50 = clarity(h, 16000.0, time_ms=50.0)
        assert c50 == float("inf")

    def test_clarity_negative_for_dense_tail(self):
        """A pure-tail RIR (no direct path) has negative clarity."""
        fs = 16000.0
        h = _make_exp_decay_rir(fs, 1.0, 2.0)
        # Remove the first 100 ms entirely → only tail left.
        h[: int(0.1 * fs)] = 0.0
        c50 = clarity(h, fs, time_ms=50.0)
        assert c50 == -float("inf")

    def test_c50_c80_consistency(self):
        """C80 ≥ C50 because the 80 ms window captures more early energy."""
        fs = 16000.0
        room = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=0.85)
        h, _, _ = shoebox_rir(
            room, (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            fs=fs, ir_length=4 * fs, max_reflection_order=12,
        )
        c50 = clarity(h, fs, time_ms=50.0)
        c80 = clarity(h, fs, time_ms=80.0)
        assert c80 >= c50

    def test_definition_in_zero_one(self):
        fs = 16000.0
        room = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=0.7)
        h, _, _ = shoebox_rir(
            room, (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            fs=fs, ir_length=2 * fs, max_reflection_order=10,
        )
        d50 = definition(h, fs, time_ms=50.0)
        assert 0.0 <= d50 <= 1.0

    def test_split_out_of_range(self):
        h = _make_exp_decay_rir(16000.0, 0.1, 0.1)  # 0.1 s → 1600 samples
        with pytest.raises(ValueError, match="falls outside"):
            clarity(h, 16000.0, time_ms=200.0)  # 200 ms = 3200 samples


class TestRIRMetricsBundle:
    def test_bundle_matches_individual(self):
        fs = 16000.0
        room = ShoeboxRoom(dimensions_m=(6.0, 5.0, 3.0), reflection=0.85)
        h, _, _ = shoebox_rir(
            room, (1.5, 2.0, 1.5), (4.0, 3.0, 1.5),
            fs=fs, ir_length=4 * fs, max_reflection_order=12,
        )
        m = rir_metrics(h, fs)
        assert isinstance(m, RIRMetrics)
        assert_allclose(m.rt60_s, reverberation_time(h, fs), atol=1e-12)
        assert_allclose(m.edt_s, early_decay_time(h, fs), atol=1e-12)
        assert_allclose(m.c50_db, clarity(h, fs, time_ms=50.0), atol=1e-12)
        assert_allclose(m.c80_db, clarity(h, fs, time_ms=80.0), atol=1e-12)
        assert_allclose(m.d50, definition(h, fs, time_ms=50.0), atol=1e-12)
        assert_allclose(m.edc_db, energy_decay_curve(h), atol=1e-12)
        assert_allclose(m.ts_s, center_time(h, fs), atol=1e-12)
        assert_allclose(m.g_db, strength(h), atol=1e-12)
        assert_allclose(m.drr_db, direct_to_reverberant_ratio(h, fs), atol=1e-12)


class TestExtendedISOStyleMetrics:
    def test_energy_helpers_and_center_time(self):
        fs = 1000.0
        h = np.zeros(1000)
        h[100] = 2.0
        h[300] = 1.0
        assert total_energy(h) == 5.0
        assert_allclose(center_time(h, fs), (0.1 * 4.0 + 0.3 * 1.0) / 5.0)
        early, late = early_late_energy(h, fs, time_ms=200.0)
        assert_allclose([early, late], [4.0, 1.0])
        assert_allclose(strength(h, reference_energy=5.0), 0.0)

    def test_direct_to_reverberant_ratio(self):
        fs = 1000.0
        h = np.zeros(1000)
        h[10] = 2.0
        h[200] = 1.0
        assert_allclose(
            direct_to_reverberant_ratio(h, fs, direct_window_ms=3.0),
            10.0 * np.log10(4.0),
        )

    def test_reverberation_times_and_banded_metrics(self):
        fs = 16000.0
        h = _make_exp_decay_rir(fs, 0.4, 5.0)
        rts = reverberation_times(h, fs, methods=("T20", "T30"))
        assert set(rts) == {"T20", "T30"}
        metrics = banded_rir_metrics(np.stack([h, h]), fs)
        assert len(metrics) == 2
        assert all(isinstance(m, RIRMetrics) for m in metrics)

    def test_interaural_cross_correlation(self):
        fs = 48000.0
        h = np.zeros(512)
        h[20] = 1.0
        assert_allclose(interaural_cross_correlation(h, h, fs), 1.0)
        delayed = np.zeros_like(h)
        delayed[24] = 1.0
        assert_allclose(
            interaural_cross_correlation(h, delayed, fs, max_lag_ms=1.0),
            1.0,
        )

    def test_lateral_energy_fraction(self):
        fs = 1000.0
        omni = np.zeros(200)
        lateral = np.zeros(200)
        omni[0] = 1.0
        omni[20] = 2.0
        lateral[20] = 1.0
        assert_allclose(
            lateral_energy_fraction(lateral, omni, fs, early_start_ms=5.0, early_end_ms=80.0),
            1.0 / 5.0,
        )
