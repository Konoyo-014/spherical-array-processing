"""Tests for the `spherical_array_processing.covariance` regularisers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spherical_array_processing.covariance import (
    diagonal_loading,
    forward_backward_average,
    ledoit_wolf_shrinkage,
    oas_shrinkage,
)


class TestLedoitWolf:
    def test_matches_sklearn_on_real_data(self):
        """Cross-check against sklearn.covariance.ledoit_wolf, which is
        the de-facto reference for the Ledoit-Wolf shrinkage on real
        Gaussian data.
        """
        sklearn_covariance = pytest.importorskip("sklearn.covariance")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 20))
        sk_cov, sk_rho = sklearn_covariance.ledoit_wolf(X, assume_centered=True)
        my_cov, my_rho = ledoit_wolf_shrinkage(X, return_shrinkage=True)
        assert_allclose(my_rho, sk_rho, atol=1e-10)
        assert_allclose(my_cov, sk_cov, atol=1e-12)

    def test_preserves_hermitian(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 12)) + 1j * rng.normal(size=(100, 12))
        C = ledoit_wolf_shrinkage(X)
        assert_allclose(C, C.conj().T, atol=1e-12)

    def test_rejects_single_snapshot(self):
        with pytest.raises(ValueError, match="at least 2 snapshots"):
            ledoit_wolf_shrinkage(np.zeros((1, 5)))

    def test_shrinkage_in_unit_interval(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(40, 20)) + 1j * rng.normal(size=(40, 20))
        _, rho = ledoit_wolf_shrinkage(X, return_shrinkage=True)
        assert 0.0 <= rho <= 1.0


class TestOAS:
    def test_shrinkage_in_unit_interval(self):
        rng = np.random.default_rng(3)
        M = 12
        X = rng.normal(size=(30, M)) + 1j * rng.normal(size=(30, M))
        cov = X.conj().T @ X / 30
        _, rho = oas_shrinkage(cov, n_snapshots=30, return_shrinkage=True)
        assert 0.0 <= rho <= 1.0

    def test_high_snapshot_count_shrinks_less(self):
        """With many snapshots the sample covariance is reliable, so
        OAS should use a smaller shrinkage weight than with few
        snapshots."""
        rng = np.random.default_rng(4)
        M = 8
        true_cov = np.diag(np.arange(1, M + 1).astype(complex))
        chol = np.linalg.cholesky(true_cov)

        def _sample_rho(n_snap: int) -> float:
            x = rng.normal(size=(n_snap, M)) + 1j * rng.normal(size=(n_snap, M))
            y = x @ chol.T
            cov = y.conj().T @ y / n_snap
            _, rho = oas_shrinkage(cov, n_snapshots=n_snap, return_shrinkage=True)
            return rho

        rho_few = _sample_rho(20)
        rho_many = _sample_rho(2000)
        assert rho_many < rho_few

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            oas_shrinkage(np.zeros((4, 5)), n_snapshots=10)


class TestForwardBackward:
    def test_is_hermitian_and_idempotent(self):
        rng = np.random.default_rng(5)
        M = 7
        X = rng.normal(size=(30, M)) + 1j * rng.normal(size=(30, M))
        cov = X.conj().T @ X / 30
        fb = forward_backward_average(cov)
        assert_allclose(fb, fb.conj().T, atol=1e-12)
        # Applying FB twice reproduces FB because J² = I.
        fb2 = forward_backward_average(fb)
        assert_allclose(fb2, fb, atol=1e-12)

    def test_custom_exchange_matrix(self):
        rng = np.random.default_rng(6)
        M = 4
        cov = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
        cov = cov + cov.conj().T
        I = np.eye(M, dtype=complex)
        out = forward_backward_average(cov, exchange_matrix=I)
        # J=I gives R + R* averaged.
        assert_allclose(out, 0.5 * (cov + cov.conj()), atol=1e-12)


class TestDiagonalLoading:
    def test_default_loading_is_trace_relative(self):
        rng = np.random.default_rng(7)
        cov = rng.normal(size=(6, 6))
        cov = cov @ cov.T  # PSD
        loaded = diagonal_loading(cov, fraction_of_trace=1e-2)
        diff = np.diag(loaded - cov)
        expected = 1e-2 * np.trace(cov) / 6
        assert_allclose(diff, np.full(6, expected), atol=1e-12)

    def test_output_is_hermitian_even_for_nearly_hermitian_input(self):
        cov = np.array(
            [[1.0 + 0.0j, 2.0 + 1e-9j], [2.0 - 3e-9j, 3.0 + 0.0j]],
            dtype=complex,
        )
        loaded = diagonal_loading(cov)
        assert_allclose(loaded, loaded.conj().T, atol=1e-12)

    def test_rejects_negative_loading(self):
        with pytest.raises(ValueError, match="non-negative"):
            diagonal_loading(np.eye(3), loading=-0.1)
