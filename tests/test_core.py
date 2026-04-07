import pytest
import numpy as np

from core.types import MAData, MAResult
from core.validation import (
    require_min_studies,
    safe_exp,
    safe_log,
    logsumexp_mean,
    ensure_positive_definite,
    check_convergence,
)


# ---------------------------------------------------------------------------
# TestMAData (6 tests)
# ---------------------------------------------------------------------------

class TestMAData:
    def test_basic_creation(self):
        """2 studies — verify length and group defaults to None."""
        effect = np.array([0.5, 0.8])
        se = np.array([0.1, 0.2])
        data = MAData(effect=effect, se=se)
        assert len(data.effect) == 2
        assert len(data.se) == 2
        assert data.group is None
        assert data.year is None
        assert data.design is None

    def test_mismatched_lengths_raises(self):
        """Mismatched effect/se lengths must raise AssertionError with 'same length'."""
        with pytest.raises(AssertionError, match="same length"):
            MAData(effect=np.array([0.5, 0.8]), se=np.array([0.1]))

    def test_empty_raises(self):
        """Empty arrays must raise AssertionError with 'at least 1'."""
        with pytest.raises(AssertionError, match="at least 1"):
            MAData(effect=np.array([]), se=np.array([]))

    def test_zero_se_raises(self):
        """A zero standard error must raise AssertionError with 'positive'."""
        with pytest.raises(AssertionError, match="positive"):
            MAData(effect=np.array([0.5, 0.8]), se=np.array([0.0, 0.2]))

    def test_negative_se_raises(self):
        """A negative standard error must raise AssertionError with 'positive'."""
        with pytest.raises(AssertionError, match="positive"):
            MAData(effect=np.array([0.5, 0.8]), se=np.array([-0.1, 0.2]))

    def test_with_optional_fields(self):
        """3 studies with group, year, and design arrays populate correctly."""
        effect = np.array([0.1, 0.2, 0.3])
        se = np.array([0.05, 0.06, 0.07])
        group = np.array([0, 1, 0])
        year = np.array([2010, 2015, 2020])
        design = np.array(["RCT", "cohort", "RCT"])
        data = MAData(effect=effect, se=se, group=group, year=year, design=design)
        assert len(data.effect) == 3
        np.testing.assert_array_equal(data.group, group)
        np.testing.assert_array_equal(data.year, year)
        np.testing.assert_array_equal(data.design, design)


# ---------------------------------------------------------------------------
# TestMAResult (2 tests)
# ---------------------------------------------------------------------------

class TestMAResult:
    def test_basic_creation(self):
        """Verify all fields and that details defaults to empty dict."""
        result = MAResult(estimate=0.5, ci_lower=0.2, ci_upper=0.8, method="DL")
        assert result.estimate == 0.5
        assert result.ci_lower == 0.2
        assert result.ci_upper == 0.8
        assert result.method == "DL"
        assert result.details == {}

    def test_with_details(self):
        """Verify details dict is accessible and correct."""
        details = {"tau2": 0.04, "I2": 55.3, "Q": 12.1}
        result = MAResult(
            estimate=0.42, ci_lower=0.15, ci_upper=0.69,
            method="REML", details=details
        )
        assert result.details["tau2"] == pytest.approx(0.04)
        assert result.details["I2"] == pytest.approx(55.3)
        assert result.details["Q"] == pytest.approx(12.1)


# ---------------------------------------------------------------------------
# TestValidation (14 tests)
# ---------------------------------------------------------------------------

class TestValidation:
    # --- require_min_studies ---

    def test_require_min_studies_passes(self):
        """No error when study count meets the minimum."""
        data = MAData(effect=np.array([0.1, 0.2, 0.3]), se=np.array([0.05, 0.06, 0.07]))
        require_min_studies(data, 3)   # should not raise

    def test_require_min_studies_raises(self):
        """ValueError raised when study count is below minimum."""
        data = MAData(effect=np.array([0.1, 0.2]), se=np.array([0.05, 0.06]))
        with pytest.raises(ValueError, match="at least 5"):
            require_min_studies(data, 5)

    # --- safe_exp ---

    def test_safe_exp_normal(self):
        """safe_exp of 0 is 1.0."""
        assert safe_exp(0.0) == pytest.approx(1.0)

    def test_safe_exp_overflow(self):
        """safe_exp clips very large input so result is finite."""
        result = safe_exp(1e10)
        assert np.isfinite(result)

    def test_safe_exp_array(self):
        """safe_exp works element-wise on arrays, including overflow values."""
        x = np.array([-1000.0, 0.0, 1000.0])
        result = safe_exp(x)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        assert result[1] == pytest.approx(1.0)

    # --- safe_log ---

    def test_safe_log_normal(self):
        """safe_log of 1 is 0."""
        assert safe_log(1.0) == pytest.approx(0.0)

    def test_safe_log_zero(self):
        """safe_log of 0 is finite (clipped to 1e-300 before log)."""
        result = safe_log(0.0)
        assert np.isfinite(result)

    def test_safe_log_negative(self):
        """safe_log of a negative number is finite (clipped)."""
        result = safe_log(-5.0)
        assert np.isfinite(result)

    # --- logsumexp_mean ---

    def test_logsumexp_mean_basic(self):
        """logsumexp_mean of all-zeros array equals 0."""
        x = np.zeros(5)
        result = logsumexp_mean(x)
        assert result == pytest.approx(0.0)

    def test_logsumexp_mean_large(self):
        """logsumexp_mean of values around 800 returns a finite result."""
        x = np.array([800.0, 801.0, 802.0])
        result = logsumexp_mean(x)
        assert np.isfinite(result)

    # --- ensure_positive_definite ---

    def test_ensure_positive_definite_already_pd(self):
        """Already-PD matrix is unchanged (eigenvalues remain positive)."""
        M = np.array([[4.0, 1.0], [1.0, 3.0]])
        result = ensure_positive_definite(M)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0)

    def test_ensure_positive_definite_not_pd(self):
        """Matrix with a negative eigenvalue is fixed to have all positive eigenvalues."""
        # Construct a matrix with a negative eigenvalue
        M = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3 and -1
        result = ensure_positive_definite(M)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals >= 1e-8)

    # --- check_convergence ---

    def test_check_convergence_converged(self):
        """Returns True when the last n values are within tolerance."""
        values = [10.0, 5.0, 1.0001, 1.0002, 1.0001]
        assert check_convergence(values, tol=1e-3, n=3) is True

    def test_check_convergence_not_converged(self):
        """Returns False when the last n values span more than tolerance."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert check_convergence(values, tol=1e-3, n=3) is False

    def test_check_convergence_short(self):
        """Returns False when fewer values are provided than required."""
        values = [1.0]
        assert check_convergence(values, tol=1e-3, n=3) is False


# ---------------------------------------------------------------------------
# TestDataLoader (4 tests — skipped when CSVs are absent)
# ---------------------------------------------------------------------------

import os as _os
_DATA_DIR = _os.environ.get(
    "TMAL_DATA_DIR",
    _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data"),
)
_HAS_INTEGRATED = _os.path.exists(
    _os.path.join(_DATA_DIR, "all_real_data_integrated.csv")
)

from core.data_loader import load_integrated


@pytest.mark.skipif(not _HAS_INTEGRATED, reason="CSV not present in data/")
class TestDataLoader:
    def test_load_integrated_returns_madata(self):
        d = load_integrated()
        assert hasattr(d, "effect") and len(d.effect) > 0

    def test_load_integrated_filter_domain(self):
        d = load_integrated(domain="Colorectal Cancer", source="BMC_2022")
        assert 0 < len(d.effect) < 2900

    def test_load_integrated_bad_domain_raises(self):
        with pytest.raises(ValueError, match="0 studies"):
            load_integrated(domain="NONEXISTENT_XYZ")

    def test_load_integrated_has_optional_fields(self):
        d = load_integrated()
        assert d.year is not None and d.design is not None and d.group is not None
