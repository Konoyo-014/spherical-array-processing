"""Tests for classical acoustics helper formulas."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing import acoustics


def test_pressure_and_power_db_round_trips():
    values = np.array([0.25, 1.0, 4.0])
    assert_allclose(
        acoustics.db_to_amplitude(acoustics.amplitude_to_db(values)),
        values,
        rtol=1e-14,
    )
    assert_allclose(
        acoustics.db_to_power(acoustics.power_to_db(values)),
        values,
        rtol=1e-14,
    )
    assert_allclose(acoustics.pressure_to_spl(20e-6), 0.0, atol=1e-12)
    assert_allclose(acoustics.spl_to_pressure(94.0), 1.0023744672545445, rtol=1e-12)
    assert_allclose(acoustics.amplitude_ratio_from_db(acoustics.db_from_amplitude_ratio(2.0)), 2.0)
    assert_allclose(acoustics.power_ratio_from_db(acoustics.db_from_power_ratio(2.0)), 2.0)
    assert_allclose(acoustics.level_difference_db(60.0, 66.0), 6.0)
    assert_allclose(acoustics.pressure_ratio_from_spl_difference(6.02059991328), 2.0)
    assert_allclose(acoustics.intensity_ratio_from_level_difference(3.01029995664), 2.0)
    assert_allclose(acoustics.level_change_for_distance_ratio(2.0), -6.02059991328)
    assert_allclose(acoustics.distance_ratio_for_level_change(-6.02059991328), 2.0)


def test_leq_and_exposure_for_constant_pressure():
    p = np.full(48000, 0.02)
    assert_allclose(acoustics.pressure_rms(p), 0.02)
    assert_allclose(acoustics.leq(p), 60.0, atol=1e-12)
    assert_allclose(acoustics.sound_exposure(p, 48000.0), 0.0004, atol=1e-15)
    assert_allclose(acoustics.sound_exposure_level(p, 48000.0), 60.0, atol=1e-12)


def test_energetic_level_helpers():
    assert_allclose(acoustics.energetic_sum_db([60.0, 60.0]), 63.01029995664)
    assert_allclose(acoustics.energetic_mean_db([60.0, 60.0]), 60.0)
    assert_allclose(
        acoustics.equivalent_continuous_level([60.0, 70.0], [1.0, 3.0]),
        68.8930173973,
    )


def test_fractional_octave_bands_have_expected_1000_hz_band():
    centers = acoustics.third_octave_band_centers(f_min_hz=900.0, f_max_hz=1120.0)
    assert 1000.0 in np.round(centers, 12)
    band = acoustics.nearest_fractional_octave_band(995.0, 3)
    assert_allclose(band.center_hz, 1000.0, atol=1e-12)
    assert band.lower_hz < 1000.0 < band.upper_hz
    assert band.nominal_hz == 1000.0
    octave = acoustics.fractional_octave_bands(1, f_min_hz=900.0, f_max_hz=1120.0)
    assert len(octave) == 1
    assert_allclose(octave[0].center_hz, 1000.0, atol=1e-12)


def test_frequency_weighting_reference_values():
    freqs = np.array([1000.0, 100.0])
    a = acoustics.a_weighting_db(freqs)
    c = acoustics.c_weighting_db(freqs)
    assert_allclose(a[0], 0.0, atol=0.02)
    assert_allclose(c[0], 0.0, atol=0.02)
    assert a[1] < c[1] < 0.0
    assert_allclose(acoustics.z_weighting_db(freqs), np.zeros(2))
    assert_allclose(acoustics.apply_frequency_weighting([60.0], [1000.0], "A"), [60.0], atol=0.02)


def test_medium_directivity_and_reflection_helpers():
    assert 343.0 < acoustics.speed_of_sound(20.0) < 344.0
    assert 1.1 < acoustics.air_density(20.0) < 1.3
    assert 400.0 < acoustics.characteristic_impedance(20.0) < 430.0
    assert_allclose(acoustics.directivity_factor_to_index(2.0), 3.01029995664)
    assert_allclose(acoustics.directivity_index_to_factor(3.01029995664), 2.0)
    assert_allclose(acoustics.absorption_to_reflection(0.75), 0.5)
    assert_allclose(acoustics.reflection_to_absorption(0.5), 0.75)
    assert_allclose(acoustics.spherical_spreading_loss_db(2.0), 6.02059991328)
    assert_allclose(acoustics.distance_attenuated_spl(80.0, 2.0), 73.97940008672)
    assert_allclose(acoustics.intensity_to_sound_intensity_level(1e-12), 0.0)
    assert_allclose(acoustics.sound_intensity_level_to_intensity(0.0), 1e-12)
    assert_allclose(acoustics.sound_power_to_sound_power_level(1e-12), 0.0)
    assert_allclose(acoustics.sound_power_level_to_power(0.0), 1e-12)
    assert_allclose(acoustics.plane_wave_pressure_from_intensity(acoustics.plane_wave_intensity_from_pressure(2.0)), 2.0)
    assert_allclose(acoustics.pressure_from_particle_velocity(acoustics.particle_velocity_from_pressure(2.0)), 2.0)
    assert_allclose(acoustics.absorption_loss_db(0.75), 6.02059991328)
    assert_allclose(acoustics.reflection_loss_db(0.5), 6.02059991328)


def test_wave_delay_and_sinusoid_helpers():
    assert_allclose(acoustics.wavelength(343.0), 1.0)
    assert_allclose(acoustics.frequency_from_wavelength(1.0), 343.0)
    assert_allclose(acoustics.period(1000.0), 0.001)
    assert_allclose(acoustics.frequency_from_period(0.001), 1000.0)
    assert_allclose(acoustics.angular_frequency(2.0), 4.0 * np.pi)
    assert_allclose(acoustics.frequency_from_angular_frequency(4.0 * np.pi), 2.0)
    phase = acoustics.phase_shift_for_delay(1000.0, 0.00025)
    assert_allclose(phase, -0.5 * np.pi)
    assert_allclose(acoustics.phase_degrees_to_radians(180.0), np.pi)
    assert_allclose(acoustics.phase_radians_to_degrees(np.pi), 180.0)
    assert_allclose(acoustics.wrap_phase_rad(3.0 * np.pi), -np.pi)
    assert_allclose(acoustics.wrap_phase_deg(540.0), -180.0)
    assert_allclose(acoustics.time_delay_for_phase_shift(1000.0, phase), 0.00025)
    assert_allclose(acoustics.distance_delay(343.0), 1.0)
    assert_allclose(acoustics.sample_delay_from_distance(343.0, 48000.0), 48000.0)
    assert_allclose(acoustics.distance_from_sample_delay(48000.0, 48000.0), 343.0)
    assert_allclose(acoustics.peak_to_rms_sine(acoustics.rms_to_peak_sine(2.0)), 2.0)
    assert_allclose(acoustics.peak_to_peak_to_rms_sine(acoustics.rms_to_peak_to_peak_sine(2.0)), 2.0)
    assert_allclose(acoustics.crest_factor([1.0, -1.0, 1.0, -1.0]), 1.0)
    assert_allclose(acoustics.crest_factor_db([1.0, -1.0, 1.0, -1.0]), 0.0)
    c = acoustics.speed_of_sound(20.0)
    assert_allclose(acoustics.temperature_from_speed_of_sound(c), 20.0)
    assert_allclose(acoustics.speed_from_mach(acoustics.mach_number(34.3)), 34.3)


def test_psychoacoustic_frequency_scales_are_monotonic():
    freqs = np.array([0.0, 1000.0, 4000.0])
    assert np.all(np.diff(acoustics.bark_rate(freqs)) > 0.0)
    assert np.all(np.diff(acoustics.erb_rate(freqs)) > 0.0)
    assert np.all(np.diff(acoustics.mel_rate(freqs)) > 0.0)
    assert_allclose(acoustics.inverse_mel_rate(acoustics.mel_rate(freqs)), freqs, atol=1e-10)
    assert_allclose(acoustics.inverse_erb_rate(acoustics.erb_rate(freqs)), freqs, atol=1e-10)
    assert np.all(acoustics.erb_bandwidth(freqs) > 0.0)
    assert np.all(acoustics.critical_bandwidth(freqs) > 0.0)
    assert_allclose(acoustics.semitones_between(440.0, 880.0), 12.0)
    assert_allclose(acoustics.frequency_ratio_from_semitones(12.0), 2.0)
    assert_allclose(acoustics.cents_between(440.0, 880.0), 1200.0)
    assert_allclose(acoustics.frequency_ratio_from_cents(1200.0), 2.0)
    assert_allclose(acoustics.sone_to_phon(acoustics.phon_to_sone(60.0)), 60.0)


def test_band_interval_and_motion_helpers():
    assert_allclose(acoustics.octave_ratio(1.0), 2.0)
    ratio = acoustics.fractional_octave_ratio(3)
    assert ratio > 1.0
    lo, hi = acoustics.band_edges_from_center_q(1000.0, 4.0)
    assert_allclose(acoustics.center_frequency_from_edges(lo, hi), 1000.0)
    assert_allclose(acoustics.q_factor_from_band_edges(lo, hi), 4.0)
    assert_allclose(acoustics.bandwidth_from_edges(lo, hi), hi - lo)
    assert acoustics.doppler_shift_moving_source(1000.0, 10.0) > 1000.0
    assert acoustics.doppler_shift_moving_observer(1000.0, 10.0) > 1000.0
    assert_allclose(acoustics.hz_to_rpm(acoustics.rpm_to_hz(120.0)), 120.0)


def test_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="reference"):
        acoustics.amplitude_to_db(1.0, reference=0.0)
    with pytest.raises(ValueError, match="power"):
        acoustics.power_to_db(-1.0)
    with pytest.raises(ValueError, match="fraction"):
        acoustics.fractional_octave_center_frequencies(0)
    with pytest.raises(ValueError, match="frequencies"):
        acoustics.a_weighting_db([0.0])
