"""Smoke and property tests for the pure functions in the four analysis scripts.

These cover the previously-untested top-level functions imported from the
exploratory synthesis scripts (P1-11 in review-findings.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from colorectal_cancer_synthesis import pool_quantum, sinkhorn_knopp
from epistemic_horizon_synthesis import (
    anabelian_iutt_deformation,
    compute_path_signature,
    perfectoid_tilting,
)
from omniscience_tier_synthesis import (
    _first_n_primes,
    chow_motive_of_evidence,
    stein_variational_gradient_descent,
)
from the_absolute_limit import (
    chaitins_omega_approximation,
    turing_degree_of_medicine,
)


# ---------------------------------------------------------------------------
# colorectal_cancer_synthesis
# ---------------------------------------------------------------------------

class TestPoolQuantum:
    def test_empty_returns_zero(self):
        assert pool_quantum(np.array([]), np.array([])) == 0.0

    def test_single_study(self):
        out = pool_quantum(np.array([0.4]), np.array([0.05]))
        assert np.isfinite(out)

    def test_symmetry_around_zero(self):
        y = np.array([-0.3, 0.0, 0.3])
        v = np.array([0.04, 0.04, 0.04])
        assert abs(pool_quantum(y, v)) < 1e-6

    def test_sign_follows_effects(self):
        v = np.full(5, 0.04)
        pos = pool_quantum(np.full(5, 0.5), v)
        neg = pool_quantum(np.full(5, -0.5), v)
        assert pos > 0 and neg < 0
        assert np.isclose(pos, -neg, atol=1e-8)


class TestSinkhorn:
    def test_doubly_stochastic_marginals(self):
        rng = np.random.RandomState(0)
        C = rng.uniform(0, 1, (6, 8))
        P = sinkhorn_knopp(C, reg=0.1, num_iters=200)
        np.testing.assert_allclose(P.sum(axis=1), np.full(6, 1 / 6), atol=1e-3)
        np.testing.assert_allclose(P.sum(axis=0), np.full(8, 1 / 8), atol=1e-3)

    def test_nonnegative(self):
        C = np.array([[0.0, 1.0], [1.0, 0.0]])
        P = sinkhorn_knopp(C, reg=0.5)
        assert np.all(P >= 0)


# ---------------------------------------------------------------------------
# epistemic_horizon_synthesis
# ---------------------------------------------------------------------------

class TestPathSignature:
    def test_short_input(self):
        sig = compute_path_signature(np.array([0.1]))
        np.testing.assert_array_equal(sig, np.array([1.0, 0.0, 0.0]))

    def test_level1_is_total_drift(self):
        y = np.array([0.0, 0.2, 0.5, 0.4])
        sig = compute_path_signature(y)
        # S1 = sum(diff(y)) = y[-1] - y[0]
        assert np.isclose(sig[1], y[-1] - y[0])

    def test_constant_series_zero_signature(self):
        sig = compute_path_signature(np.full(10, 0.7))
        assert np.isclose(sig[1], 0.0)
        assert np.isclose(sig[2], 0.0)


class TestPerfectoidTilting:
    def test_empty_inputs_nan(self):
        assert np.isnan(perfectoid_tilting(np.array([]), np.array([0.1])))
        assert np.isnan(perfectoid_tilting(np.array([0.1]), np.array([])))

    def test_identical_inputs_small_gap(self):
        y = np.array([-0.2, 0.1, 0.3, -0.1])
        gap = perfectoid_tilting(y, y)
        # Frobenius lossy compression introduces some error even for identical inputs
        assert abs(gap) < 0.5


class TestJensenGap:
    def test_constant_series_zero_gap(self):
        # arithmetic mean and log-mean-exp coincide when all values equal
        assert anabelian_iutt_deformation(np.full(8, 0.3)) == pytest.approx(0.0)

    def test_nonnegative(self):
        rng = np.random.RandomState(1)
        y = rng.normal(0, 0.5, 50)
        assert anabelian_iutt_deformation(y) >= 0.0


# ---------------------------------------------------------------------------
# omniscience_tier_synthesis
# ---------------------------------------------------------------------------

class TestFirstNPrimes:
    def test_first_ten(self):
        np.testing.assert_array_equal(
            _first_n_primes(10),
            np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0]),
        )

    def test_length(self):
        assert len(_first_n_primes(25)) == 25


class TestChowMotive:
    def test_nonnegative_violations(self):
        rng = np.random.RandomState(2)
        y = rng.normal(0, 0.3, 80)
        assert chow_motive_of_evidence(y) >= 0


class TestSVGD:
    def test_shape_and_finiteness(self):
        rng = np.random.RandomState(3)
        y = rng.normal(0.2, 0.1, 30)
        v = np.full(30, 0.04)
        particles = stein_variational_gradient_descent(y, v, n_particles=20, iters=30)
        assert particles.shape == (20, 1)
        assert np.all(np.isfinite(particles))

    def test_posterior_near_data_mean(self):
        rng = np.random.RandomState(4)
        y = rng.normal(0.5, 0.05, 40)
        v = np.full(40, 0.01)
        particles = stein_variational_gradient_descent(y, v, n_particles=30, iters=200)
        # With strong likelihood the posterior should pull near data mean
        assert abs(np.mean(particles) - np.mean(y)) < 0.3


# ---------------------------------------------------------------------------
# the_absolute_limit
# ---------------------------------------------------------------------------

class TestChaitinsOmega:
    def test_all_positive_yields_zero_bits(self):
        bits, omega = chaitins_omega_approximation(np.full(16, 0.5), num_bits=16)
        assert np.all(bits == 0)
        assert omega == pytest.approx(0.0)

    def test_all_negative_yields_one_bits(self):
        bits, omega = chaitins_omega_approximation(np.full(16, -0.5), num_bits=16)
        assert np.all(bits == 1)
        # sum of 2^-(i+1) for i in 0..15 -> < 1
        assert 0.99 < omega < 1.0

    def test_bit_length_capped(self):
        bits, _ = chaitins_omega_approximation(np.linspace(-1, 1, 100), num_bits=32)
        assert len(bits) == 32


class TestLempelZiv:
    def test_empty_returns_zero(self):
        assert turing_degree_of_medicine(np.array([], dtype=int)) == 0.0

    def test_constant_sequence_finite(self):
        # All-zeros is degenerate but the function still returns a finite value.
        out = turing_degree_of_medicine(np.zeros(64, dtype=int))
        assert np.isfinite(out) and out > 0

    def test_random_more_complex_than_constant(self):
        rng = np.random.RandomState(0)
        n = 256
        const_out = turing_degree_of_medicine(np.zeros(n, dtype=int))
        rand_out = turing_degree_of_medicine(rng.randint(0, 2, size=n))
        # Raw LZ word count for random is much higher than for constant input.
        # Normalized to n/log2(n), random should still exceed constant.
        assert rand_out > const_out
