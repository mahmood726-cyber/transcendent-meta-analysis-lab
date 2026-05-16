import numpy as np
import pytest
from core.types import MAData, MAResult
from methods.bayesian_model_avg import bayesian_model_average


class TestBMA:
    def test_returns_maresult(self, small_5):
        r = bayesian_model_average(small_5)
        assert isinstance(r, MAResult)
        assert r.method == "BMA"

    def test_ci_contains_estimate(self, small_5):
        r = bayesian_model_average(small_5)
        assert r.ci_lower <= r.estimate <= r.ci_upper

    def test_weights_sum_to_one(self, small_5):
        r = bayesian_model_average(small_5)
        weights = [m["weight"] for m in r.details["models"]]
        assert np.isclose(sum(weights), 1.0, atol=1e-10)

    def test_four_models_present(self, small_5):
        r = bayesian_model_average(small_5)
        names = {m["name"] for m in r.details["models"]}
        assert names == {"FE", "DL", "REML", "PM"}

    def test_homogeneous_fe_dominates(self, homogeneous_10):
        r = bayesian_model_average(homogeneous_10)
        fe_weight = [m for m in r.details["models"] if m["name"] == "FE"][0]["weight"]
        assert fe_weight > 0.3

    def test_single_study(self, single_study):
        r = bayesian_model_average(single_study)
        assert np.isclose(r.estimate, 0.5, atol=1e-6)

    def test_pair_studies(self, pair_studies):
        r = bayesian_model_average(pair_studies)
        assert np.isfinite(r.estimate)

    def test_estimate_finite(self, bimodal_20):
        r = bayesian_model_average(bimodal_20)
        assert np.isfinite(r.estimate) and np.isfinite(r.ci_lower) and np.isfinite(r.ci_upper)

    def test_bma_variance_positive(self, small_5):
        r = bayesian_model_average(small_5)
        assert r.details["bma_variance"] > 0

    def test_iv_weighted_reference(self, small_5):
        y, w = small_5.effect, 1.0 / small_5.se**2
        expected_fe = np.sum(w * y) / np.sum(w)
        r = bayesian_model_average(small_5)
        fe_est = [m for m in r.details["models"] if m["name"] == "FE"][0]["estimate"]
        assert np.isclose(fe_est, expected_fe, atol=1e-8)

    def test_outlier_data(self, with_outlier):
        r = bayesian_model_average(with_outlier)
        assert np.isfinite(r.estimate)

    def test_deterministic(self, small_5):
        r1 = bayesian_model_average(small_5)
        r2 = bayesian_model_average(small_5)
        assert np.isclose(r1.estimate, r2.estimate)
