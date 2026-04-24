"""Contract tests for the 0.4.0b15 collapse of the dual-track FOA
intensity API.

``spherical_array_processing.diffuseness.intensity_vectors_from_foa``
is now a thin wrapper over
``spherical_array_processing.ambi.intensity_vector``.  The legacy
entry point keeps its historical shape / ``channel_order`` contract
(including the exact error messages) and additionally exposes the
``normalization`` and ``physical_units`` knobs from the canonical
implementation — pushing the two APIs onto a single implementation
and, crucially, a single definition of "FOA intensity".

These tests lock that merge in place so a later refactor cannot
silently split the two entry points again.
"""

from __future__ import annotations

import numpy as np
import pytest

from spherical_array_processing.ambi.intensity import intensity_vector
from spherical_array_processing.diffuseness.estimators import (
    intensity_vectors_from_foa,
)


def _foa_last_axis(seed: int = 0, T: int = 32) -> np.ndarray:
    """Random complex FOA frames with ACN channels on the last axis."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, 4)) + 1j * rng.normal(size=(T, 4))


def test_wrapper_matches_canonical_intensity_vector_bitwise_default():
    """Default arguments must be byte-level identical to the
    canonical :func:`ambi.intensity_vector` with ``coeff_axis=-1``."""
    foa = _foa_last_axis()
    iv_wrap = intensity_vectors_from_foa(foa)
    iv_canon = intensity_vector(foa, coeff_axis=-1)
    assert np.max(np.abs(iv_wrap - iv_canon)) == 0.0


def test_wrapper_forwards_physical_units():
    """Setting ``physical_units=True`` must propagate the 1/√3
    velocity scaling into the wrapper's output."""
    foa = _foa_last_axis(seed=1)
    iv_raw = intensity_vectors_from_foa(foa)
    iv_phys = intensity_vectors_from_foa(foa, physical_units=True)
    assert np.max(np.abs(iv_phys - iv_raw / np.sqrt(3.0))) < 1e-14
    # And it must still match the canonical implementation.
    iv_canon = intensity_vector(foa, coeff_axis=-1, physical_units=True)
    assert np.max(np.abs(iv_phys - iv_canon)) == 0.0


def test_wrapper_forwards_normalization_sn3d_and_n3d():
    """``normalization`` must be accepted and forwarded so the wrapper
    sees the same SN3D / N3D paths as the canonical entry point."""
    foa = _foa_last_axis(seed=2)
    for norm in ("orthonormal", "n3d", "sn3d"):
        iv_wrap = intensity_vectors_from_foa(foa, normalization=norm)
        iv_canon = intensity_vector(foa, coeff_axis=-1, normalization=norm)
        assert np.max(np.abs(iv_wrap - iv_canon)) == 0.0, norm


def test_wrapper_fuma_matches_acn_same_field():
    """FuMa-ordered FOA of the *same* physical field must produce the
    same output as ACN-ordered FOA (the wrapper reorders internally)."""
    foa_acn = _foa_last_axis(seed=3)
    foa_fuma = foa_acn[:, [0, 3, 1, 2]]  # ACN→FuMa reorder.
    iv_acn = intensity_vectors_from_foa(foa_acn, channel_order="acn")
    iv_fuma = intensity_vectors_from_foa(foa_fuma, channel_order="fuma")
    assert np.max(np.abs(iv_fuma - iv_acn)) == 0.0


def test_wrapper_preserves_historical_error_messages():
    """The pre-wrapper contract used the exact strings ``"4 channels"``
    and ``"channel_order"`` — keep them so users who pattern-match on
    these continue to work after the collapse."""
    with pytest.raises(ValueError, match="4 channels"):
        intensity_vectors_from_foa(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="channel_order"):
        intensity_vectors_from_foa(np.zeros((3, 4)), channel_order="bogus")


def test_wrapper_accepts_real_valued_input():
    """Real-valued FOA must still be accepted without an explicit
    complex cast — historical behaviour from pre-0.4.0b15 users."""
    foa_real = np.asarray(_foa_last_axis(seed=4).real, dtype=np.float64)
    iv = intensity_vectors_from_foa(foa_real)
    # Output must be real and the canonical value.
    assert iv.dtype == np.float64
    iv_canon = intensity_vector(foa_real, coeff_axis=-1)
    assert np.max(np.abs(iv - iv_canon)) == 0.0


def test_wrapper_handles_high_rank_input():
    """Arbitrary leading-axis shapes keep working — the wrapper must
    not flatten or reshape anything unexpected."""
    rng = np.random.default_rng(5)
    foa = rng.normal(size=(2, 3, 4, 4)) + 1j * rng.normal(size=(2, 3, 4, 4))
    iv = intensity_vectors_from_foa(foa)
    assert iv.shape == (2, 3, 4, 3)
    iv_canon = intensity_vector(foa, coeff_axis=-1)
    assert np.max(np.abs(iv - iv_canon)) == 0.0
