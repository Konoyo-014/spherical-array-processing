import numpy as np

from spherical_array_processing.experimental import estimate_incomplete_foa_from_stereo


def test_stereo_to_incomplete_foa_interface():
    fs = 16000
    t = np.arange(fs) / fs
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 440 * t + 0.1)
    x = np.stack([left, right], axis=1)
    est = estimate_incomplete_foa_from_stereo(x, fs)
    assert est.foa_stft.shape[0] == 4
    assert est.observability_mask.tolist() == [True, True, False, False]
    assert est.confidence.ndim == 2
    assert est.uncertainty.shape == est.confidence.shape
    assert np.all(est.uncertainty >= 0.0)
    assert np.all(est.uncertainty <= 1.0)


def test_stereo_to_incomplete_foa_low_frequency_stability():
    fs = 16000
    t = np.arange(fs) / fs
    # Dominantly low-frequency content should produce conservative confidence.
    left = np.sin(2 * np.pi * 60 * t)
    right = np.sin(2 * np.pi * 60 * t + 0.2)
    x = np.stack([left, right], axis=1)
    est = estimate_incomplete_foa_from_stereo(x, fs)
    assert np.isfinite(est.confidence).all()
    assert np.isfinite(est.uncertainty).all()
    assert float(np.mean(est.confidence)) < 0.6
