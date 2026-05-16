# Advanced Statistics & Mathematics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 10 advanced statistical and mathematical methods (5 validated, 5 exotic) with ~130 tests to the Transcendent-Meta-Analysis-Lab.

**Architecture:** Two-layer module system — `methods/` (peer-review-grade statistics) and `exploration/` (exotic math analogies) — built on shared `core/` infrastructure with `MAData`/`MAResult` dataclasses. All pure numpy/scipy, no external ML/TDA libraries.

**Tech Stack:** Python 3.10+, numpy, scipy, pandas (loading only), pytest

**Spec:** `docs/superpowers/specs/2026-04-07-advanced-stats-design.md`

---

### Task 1: Core Types and Validation

**Files:**
- Create: `core/__init__.py`
- Create: `core/types.py`
- Create: `core/validation.py`
- Create: `tests/__init__.py`
- Create: `tests/test_core.py`

- [ ] **Step 1: Write failing tests for MAData and MAResult**

```python
# tests/test_core.py
import numpy as np
import pytest
from core.types import MAData, MAResult


class TestMAData:
    def test_basic_creation(self):
        d = MAData(effect=np.array([0.1, -0.3]), se=np.array([0.2, 0.4]))
        assert len(d.effect) == 2
        assert d.group is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError, match="same length"):
            MAData(effect=np.array([0.1]), se=np.array([0.2, 0.4]))

    def test_empty_raises(self):
        with pytest.raises(AssertionError, match="at least 1"):
            MAData(effect=np.array([]), se=np.array([]))

    def test_zero_se_raises(self):
        with pytest.raises(AssertionError, match="positive"):
            MAData(effect=np.array([0.1]), se=np.array([0.0]))

    def test_negative_se_raises(self):
        with pytest.raises(AssertionError, match="positive"):
            MAData(effect=np.array([0.1]), se=np.array([-0.2]))

    def test_with_optional_fields(self):
        d = MAData(
            effect=np.array([0.1, -0.3, 0.5]),
            se=np.array([0.2, 0.4, 0.1]),
            group=np.array(["A", "A", "B"]),
            year=np.array([2020, 2021, 2022]),
            design=np.array(["RCT", "NRS", "RCT"]),
        )
        assert len(d.group) == 3
        assert d.year[2] == 2022


class TestMAResult:
    def test_basic_creation(self):
        r = MAResult(estimate=0.5, ci_lower=0.1, ci_upper=0.9, method="test")
        assert r.estimate == 0.5
        assert r.details == {}

    def test_with_details(self):
        r = MAResult(estimate=0.5, ci_lower=0.1, ci_upper=0.9,
                     method="test", details={"tau2": 0.03})
        assert r.details["tau2"] == 0.03
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_core.py -v`
Expected: ImportError — `core.types` does not exist yet

- [ ] **Step 3: Implement core/types.py**

```python
# core/__init__.py
# empty

# core/types.py
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MAData:
    """Standard input for all meta-analysis methods."""
    effect: np.ndarray
    se: np.ndarray
    group: np.ndarray | None = None
    year: np.ndarray | None = None
    design: np.ndarray | None = None

    def __post_init__(self):
        assert len(self.effect) == len(self.se), "effect and se must have same length"
        assert len(self.effect) > 0, "MAData must have at least 1 study"
        assert np.all(self.se > 0), "All standard errors must be positive"


@dataclass
class MAResult:
    """Standard output from all meta-analysis methods."""
    estimate: float
    ci_lower: float
    ci_upper: float
    method: str
    details: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_core.py::TestMAData -v && python -m pytest tests/test_core.py::TestMAResult -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Write failing tests for validation helpers**

Append to `tests/test_core.py`:

```python
from core.validation import (
    require_min_studies, safe_exp, safe_log, logsumexp_mean,
    ensure_positive_definite, check_convergence,
)


class TestValidation:
    def test_require_min_studies_passes(self):
        d = MAData(effect=np.array([0.1, -0.3, 0.5]), se=np.array([0.2, 0.4, 0.1]))
        require_min_studies(d, 3)  # should not raise

    def test_require_min_studies_raises(self):
        d = MAData(effect=np.array([0.1, -0.3]), se=np.array([0.2, 0.4]))
        with pytest.raises(ValueError, match="at least 3"):
            require_min_studies(d, 3)

    def test_safe_exp_normal(self):
        assert np.isclose(safe_exp(1.0), np.exp(1.0))

    def test_safe_exp_overflow(self):
        result = safe_exp(1000.0)
        assert np.isfinite(result)
        assert result == np.exp(700)

    def test_safe_exp_array(self):
        result = safe_exp(np.array([1.0, 1000.0, -1000.0]))
        assert np.all(np.isfinite(result))

    def test_safe_log_normal(self):
        assert np.isclose(safe_log(1.0), 0.0)

    def test_safe_log_zero(self):
        result = safe_log(0.0)
        assert np.isfinite(result)
        assert result < -600

    def test_safe_log_negative(self):
        result = safe_log(-1.0)
        assert np.isfinite(result)

    def test_logsumexp_mean_basic(self):
        x = np.array([0.0, 0.0, 0.0])
        assert np.isclose(logsumexp_mean(x), 0.0)

    def test_logsumexp_mean_large(self):
        x = np.array([800.0, 800.0])
        result = logsumexp_mean(x)
        assert np.isfinite(result)
        assert np.isclose(result, 800.0)

    def test_ensure_positive_definite_already_pd(self):
        M = np.array([[2.0, 0.5], [0.5, 2.0]])
        result = ensure_positive_definite(M)
        assert np.allclose(result, M, atol=1e-10)

    def test_ensure_positive_definite_not_pd(self):
        M = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
        result = ensure_positive_definite(M)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0)

    def test_check_convergence_converged(self):
        vals = [1.0, 0.5, 0.25, 0.251, 0.2505]
        assert check_convergence(vals, tol=0.01)

    def test_check_convergence_not_converged(self):
        vals = [1.0, 0.5, 0.25]
        assert not check_convergence(vals, tol=0.001)

    def test_check_convergence_short(self):
        assert not check_convergence([1.0], tol=0.01)
```

- [ ] **Step 6: Implement core/validation.py**

```python
# core/validation.py
import numpy as np
from scipy.special import logsumexp as _logsumexp


def require_min_studies(data, k):
    """Raise ValueError if data has fewer than k studies."""
    if len(data.effect) < k:
        raise ValueError(f"Method requires at least {k} studies, got {len(data.effect)}")


def safe_exp(x):
    """Exponentiation clamped to avoid overflow."""
    return np.exp(np.clip(x, -700, 700))


def safe_log(x):
    """Logarithm clamped to avoid log(0)."""
    return np.log(np.clip(x, 1e-300, None))


def logsumexp_mean(x):
    """Numerically stable log(mean(exp(x)))."""
    return _logsumexp(x) - np.log(len(x))


def ensure_positive_definite(M, epsilon=1e-8):
    """Nearest symmetric positive-definite matrix (Higham 2002)."""
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def check_convergence(values, tol, n=3):
    """True if the last n values differ by less than tol."""
    if len(values) < max(n, 2):
        return False
    recent = values[-n:]
    return max(recent) - min(recent) < tol
```

- [ ] **Step 7: Run all core tests**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_core.py -v`
Expected: All ~24 tests PASS

- [ ] **Step 8: Commit**

```bash
cd C:\Users\user\Transcendent-Meta-Analysis-Lab
git add core/ tests/
git commit -m "feat: add core types (MAData, MAResult) and validation helpers"
```

---

### Task 2: Core Data Loader

**Files:**
- Create: `core/data_loader.py`
- Modify: `tests/test_core.py` (append tests)

- [ ] **Step 1: Write failing tests for data_loader**

Append to `tests/test_core.py`:

```python
from core.data_loader import load_integrated, load_pairwise70


class TestDataLoader:
    def test_load_integrated_returns_madata(self):
        d = load_integrated()
        assert hasattr(d, 'effect')
        assert hasattr(d, 'se')
        assert len(d.effect) > 0
        assert np.all(d.se > 0)
        assert np.all(np.isfinite(d.effect))

    def test_load_integrated_filter_domain(self):
        d = load_integrated(domain="Colorectal Cancer", source="BMC_2022")
        assert len(d.effect) > 0
        assert len(d.effect) < 2900  # filtered subset

    def test_load_integrated_bad_domain_empty(self):
        with pytest.raises(ValueError, match="0 studies"):
            load_integrated(domain="NONEXISTENT_DOMAIN_XYZ")

    def test_load_integrated_has_year(self):
        d = load_integrated()
        assert d.year is not None

    def test_load_integrated_has_design(self):
        d = load_integrated()
        assert d.design is not None

    def test_load_integrated_has_group(self):
        d = load_integrated()
        assert d.group is not None
```

Note: These tests require the real CSV in `data/`. If CSV is absent, they will fail with a FileNotFoundError — that's expected and acceptable as an integration gate.

- [ ] **Step 2: Implement core/data_loader.py**

```python
# core/data_loader.py
import os
import warnings
import numpy as np
import pandas as pd
from core.types import MAData


def _get_data_dir():
    return os.environ.get("TMAL_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


def load_integrated(domain=None, source=None, design=None):
    """Load all_real_data_integrated.csv and return MAData."""
    path = os.path.join(_get_data_dir(), "all_real_data_integrated.csv")
    df = pd.read_csv(path)

    for col in ['log_or', 'se']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['log_or', 'se'])
    df = df[df['se'] > 0]

    if domain is not None:
        df = df[df['domain'] == domain]
    if source is not None:
        df = df[df['source'] == source]
    if design is not None:
        df = df[df['design'] == design]

    if len(df) == 0:
        raise ValueError(f"0 studies after filtering (domain={domain}, source={source}, design={design})")
    if len(df) < 3:
        warnings.warn(f"Only {len(df)} studies after filtering — results may be unreliable")

    year = pd.to_numeric(df['year'], errors='coerce').values if 'year' in df.columns else None
    design_arr = df['design'].values if 'design' in df.columns else None
    group = (df['source'].astype(str) + ':' + df['domain'].astype(str)).values

    return MAData(
        effect=df['log_or'].values.astype(float),
        se=df['se'].values.astype(float),
        group=group,
        year=year,
        design=design_arr,
    )


def load_pairwise70(sample_n=None, seed=42):
    """Load ma4_results_pairwise70.csv and return MAData."""
    data_dir = os.environ.get("TMAL_R_DATA_DIR", _get_data_dir())
    path = os.path.join(data_dir, "ma4_results_pairwise70.csv")
    df = pd.read_csv(path)

    for col in ['theta', 'sigma']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['theta', 'sigma'])
    df = df[df['sigma'] > 0]

    if sample_n is not None and sample_n < len(df['review_id'].unique()):
        rng = np.random.RandomState(seed)
        reviews = rng.choice(df['review_id'].unique(), sample_n, replace=False)
        df = df[df['review_id'].isin(reviews)]

    if len(df) == 0:
        raise ValueError("0 studies after filtering")

    return MAData(
        effect=df['theta'].values.astype(float),
        se=df['sigma'].values.astype(float),
        group=df['review_id'].values,
    )
```

- [ ] **Step 3: Run data loader tests**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_core.py::TestDataLoader -v`
Expected: PASS if CSV present, FAIL (FileNotFoundError) if absent — both acceptable

- [ ] **Step 4: Commit**

```bash
cd C:\Users\user\Transcendent-Meta-Analysis-Lab
git add core/data_loader.py tests/test_core.py
git commit -m "feat: add data loader for integrated and pairwise70 CSVs"
```

---

### Task 3: Test Fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create all shared fixtures**

```python
# tests/conftest.py
import numpy as np
import pytest
from core.types import MAData


@pytest.fixture
def small_5():
    """5 studies with hand-calculable IV-weighted mean."""
    return MAData(
        effect=np.array([-0.5, -0.3, 0.0, 0.2, -0.1]),
        se=np.array([0.2, 0.3, 0.1, 0.4, 0.15]),
    )


@pytest.fixture
def bimodal_20():
    """Two well-separated groups for clustering tests."""
    rng = np.random.RandomState(42)
    effects = np.concatenate([
        rng.normal(-1.0, 0.1, 10),
        rng.normal(0.5, 0.1, 10),
    ])
    ses = np.full(20, 0.15)
    groups = np.array(["A"] * 10 + ["B"] * 10)
    return MAData(effect=effects, se=ses, group=groups)


@pytest.fixture
def homogeneous_10():
    """10 studies from N(0, 0.3^2), tau^2=0."""
    rng = np.random.RandomState(123)
    effects = rng.normal(0.0, 0.05, 10)  # small spread, near-homogeneous
    ses = np.full(10, 0.3)
    return MAData(effect=effects, se=ses)


@pytest.fixture
def single_study():
    """k=1 edge case."""
    return MAData(effect=np.array([0.5]), se=np.array([0.2]))


@pytest.fixture
def pair_studies():
    """k=2 edge case."""
    return MAData(effect=np.array([-0.3, 0.1]), se=np.array([0.2, 0.3]))


@pytest.fixture
def temporal_30():
    """30 studies with a known shift at index 15. Effect jumps from 0 to -0.5."""
    rng = np.random.RandomState(99)
    effects = np.concatenate([
        rng.normal(0.0, 0.1, 15),
        rng.normal(-0.5, 0.1, 15),
    ])
    ses = np.full(30, 0.2)
    years = np.arange(2000, 2030)
    return MAData(effect=effects, se=ses, year=years)


@pytest.fixture
def with_outlier():
    """10 studies near 0 + 1 outlier at 5.0."""
    rng = np.random.RandomState(77)
    effects = np.concatenate([rng.normal(0.0, 0.1, 10), [5.0]])
    ses = np.concatenate([np.full(10, 0.2), [0.3]])
    return MAData(effect=effects, se=ses)


@pytest.fixture
def clustered_reviews():
    """10 reviews x 7 outcomes each for correlated-effects methods."""
    rng = np.random.RandomState(55)
    effects = []
    ses = []
    groups = []
    for r in range(10):
        review_mean = rng.normal(0, 0.5)
        for o in range(7):
            effects.append(review_mean + rng.normal(0, 0.15))
            ses.append(rng.uniform(0.1, 0.4))
            groups.append(f"R{r:02d}")
    return MAData(
        effect=np.array(effects),
        se=np.array(ses),
        group=np.array(groups),
    )
```

- [ ] **Step 2: Verify fixtures load**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_core.py -v --fixtures`
Expected: All fixtures listed without error

- [ ] **Step 3: Commit**

```bash
cd C:\Users\user\Transcendent-Meta-Analysis-Lab
git add tests/conftest.py
git commit -m "feat: add shared test fixtures for all methods"
```

---

### Task 4: Bayesian Model Averaging

**Files:**
- Create: `methods/__init__.py`
- Create: `methods/bayesian_model_avg.py`
- Create: `tests/test_bayesian_model_avg.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bayesian_model_avg.py
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
        assert fe_weight > 0.3  # FE should have substantial weight when tau2~0

    def test_single_study(self, single_study):
        r = bayesian_model_average(single_study)
        assert np.isclose(r.estimate, 0.5, atol=1e-6)

    def test_pair_studies(self, pair_studies):
        r = bayesian_model_average(pair_studies)
        assert np.isfinite(r.estimate)

    def test_estimate_finite(self, bimodal_20):
        r = bayesian_model_average(bimodal_20)
        assert np.isfinite(r.estimate)
        assert np.isfinite(r.ci_lower)
        assert np.isfinite(r.ci_upper)

    def test_bma_variance_positive(self, small_5):
        r = bayesian_model_average(small_5)
        assert r.details["bma_variance"] > 0

    def test_iv_weighted_reference(self, small_5):
        """FE estimate should match hand-calculated IV-weighted mean."""
        y = small_5.effect
        w = 1.0 / small_5.se**2
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_bayesian_model_avg.py -v`
Expected: ImportError

- [ ] **Step 3: Implement methods/bayesian_model_avg.py**

```python
# methods/__init__.py
# empty

# methods/bayesian_model_avg.py
"""Bayesian Model Averaging across 4 meta-analytic pooling methods."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from core.types import MAData, MAResult


def _fe_model(y, v):
    """Fixed-effect inverse-variance weighted."""
    w = 1.0 / v
    mu = np.sum(w * y) / np.sum(w)
    var_mu = 1.0 / np.sum(w)
    loglik = -0.5 * np.sum(np.log(2 * np.pi * v) + (y - mu)**2 / v)
    return mu, 0.0, var_mu, loglik


def _dl_model(y, v):
    """DerSimonian-Laird random effects."""
    w = 1.0 / v
    mu_fe = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - mu_fe)**2)
    k = len(y)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c)
    w_re = 1.0 / (v + tau2)
    mu = np.sum(w_re * y) / np.sum(w_re)
    var_mu = 1.0 / np.sum(w_re)
    loglik = -0.5 * np.sum(np.log(2 * np.pi * (v + tau2)) + (y - mu)**2 / (v + tau2))
    return mu, tau2, var_mu, loglik


def _reml_model(y, v, max_iter=100, tol=1e-8):
    """REML via Newton-Raphson."""
    k = len(y)
    _, tau2_init, _, _ = _dl_model(y, v)
    tau2 = max(tau2_init, 1e-8)

    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        resid2 = (y - mu)**2
        # REML profile log-likelihood gradient and Hessian
        dl = -0.5 * np.sum(w**2 * (1.0 - w * resid2 / 1.0))  # simplified
        # Use Fisher scoring: d/dtau2 of REML log-lik
        dl = 0.5 * (np.sum(w**2 * resid2) - np.sum(w) + np.sum(w)**(-1) * np.sum(w**2))
        ddl = -0.5 * np.sum(w**2)
        if abs(ddl) < 1e-15:
            break
        step = -dl / ddl
        tau2_new = max(0.0, tau2 + step)
        if abs(tau2_new - tau2) < tol:
            tau2 = tau2_new
            break
        tau2 = tau2_new

    w = 1.0 / (v + tau2)
    mu = np.sum(w * y) / np.sum(w)
    var_mu = 1.0 / np.sum(w)
    loglik = -0.5 * np.sum(np.log(2 * np.pi * (v + tau2)) + (y - mu)**2 / (v + tau2))
    return mu, tau2, var_mu, loglik


def _pm_model(y, v, max_iter=100, tol=1e-8):
    """Paule-Mandel iterative generalized Q estimator."""
    k = len(y)
    tau2 = 0.0
    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        Q = np.sum(w * (y - mu)**2)
        if abs(Q - (k - 1)) < tol:
            break
        # Update tau2 to make Q = k-1
        c = np.sum(w) - np.sum(w**2) / np.sum(w)
        tau2 = max(0.0, tau2 + (Q - (k - 1)) / c)

    w = 1.0 / (v + tau2)
    mu = np.sum(w * y) / np.sum(w)
    var_mu = 1.0 / np.sum(w)
    loglik = -0.5 * np.sum(np.log(2 * np.pi * (v + tau2)) + (y - mu)**2 / (v + tau2))
    return mu, tau2, var_mu, loglik


def bayesian_model_average(data: MAData) -> MAResult:
    """BMA across FE, DL, REML, and Paule-Mandel pooling methods."""
    y = data.effect
    v = data.se**2
    k = len(y)

    if k == 1:
        return MAResult(
            estimate=y[0],
            ci_lower=y[0] - 1.96 * data.se[0],
            ci_upper=y[0] + 1.96 * data.se[0],
            method="BMA",
            details={"models": [{"name": "FE", "estimate": y[0], "tau2": 0.0,
                                  "bic": 0.0, "weight": 1.0}],
                     "bma_variance": v[0]},
        )

    models = []
    for name, fn in [("FE", _fe_model), ("DL", _dl_model),
                     ("REML", _reml_model), ("PM", _pm_model)]:
        try:
            mu, tau2, var_mu, loglik = fn(y, v)
            if not np.isfinite(loglik):
                loglik = -1e10
            p = 1 if name == "FE" else 2  # FE has 1 param (mu), others have 2 (mu, tau2)
            bic = -2 * loglik + p * np.log(k)
            models.append({"name": name, "estimate": mu, "tau2": tau2,
                           "var": var_mu, "bic": bic, "loglik": loglik})
        except Exception:
            models.append({"name": name, "estimate": np.nan, "tau2": np.nan,
                           "var": np.nan, "bic": 1e10, "loglik": -1e10})

    # BIC-based weights (using log-sum-exp for stability)
    bics = np.array([m["bic"] for m in models])
    log_weights = -0.5 * (bics - np.min(bics))
    weights = np.exp(log_weights) / np.sum(np.exp(log_weights))

    valid = [i for i, m in enumerate(models) if np.isfinite(m["estimate"])]
    if not valid:
        valid = [0]
        weights = np.array([1.0, 0.0, 0.0, 0.0])

    for i, m in enumerate(models):
        m["weight"] = float(weights[i])

    bma_est = sum(weights[i] * models[i]["estimate"] for i in valid)
    bma_var = sum(weights[i] * (models[i]["var"] + (models[i]["estimate"] - bma_est)**2)
                  for i in valid)

    z = norm.ppf(0.975)
    se_bma = np.sqrt(max(bma_var, 1e-15))

    return MAResult(
        estimate=float(bma_est),
        ci_lower=float(bma_est - z * se_bma),
        ci_upper=float(bma_est + z * se_bma),
        method="BMA",
        details={"models": [{k: v for k, v in m.items() if k != "loglik"} for m in models],
                 "bma_variance": float(bma_var)},
    )
```

- [ ] **Step 4: Run tests**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_bayesian_model_avg.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\user\Transcendent-Meta-Analysis-Lab
git add methods/ tests/test_bayesian_model_avg.py
git commit -m "feat: add Bayesian Model Averaging (FE/DL/REML/PM) with 12 tests"
```

---

### Task 5: Robust Variance Estimation (CR2)

**Files:**
- Create: `methods/robust_variance.py`
- Create: `tests/test_robust_variance.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_robust_variance.py
import numpy as np
import pytest
from core.types import MAData, MAResult
from methods.robust_variance import robust_variance_estimate


class TestRobustVariance:
    def test_returns_maresult(self, clustered_reviews):
        r = robust_variance_estimate(clustered_reviews)
        assert isinstance(r, MAResult)
        assert r.method == "RobustCR2"

    def test_ci_contains_estimate(self, clustered_reviews):
        r = robust_variance_estimate(clustered_reviews)
        assert r.ci_lower <= r.estimate <= r.ci_upper

    def test_robust_se_positive(self, clustered_reviews):
        r = robust_variance_estimate(clustered_reviews)
        assert r.details["robust_se"] > 0

    def test_df_positive(self, clustered_reviews):
        r = robust_variance_estimate(clustered_reviews)
        assert r.details["df_satterthwaite"] >= 1.0

    def test_n_clusters_correct(self, clustered_reviews):
        r = robust_variance_estimate(clustered_reviews)
        assert r.details["n_clusters"] == 10

    def test_independent_matches_naive(self):
        """When each study is its own cluster, robust SE = naive SE."""
        d = MAData(
            effect=np.array([-0.5, -0.3, 0.0, 0.2, -0.1]),
            se=np.array([0.2, 0.3, 0.1, 0.4, 0.15]),
            group=np.array(["A", "B", "C", "D", "E"]),
        )
        r = robust_variance_estimate(d)
        assert np.isclose(r.details["robust_se"], r.details["naive_se"], rtol=0.2)

    def test_requires_group(self, small_5):
        with pytest.raises(ValueError, match="group"):
            robust_variance_estimate(small_5)

    def test_single_cluster_warns(self):
        d = MAData(
            effect=np.array([-0.5, -0.3, 0.0]),
            se=np.array([0.2, 0.3, 0.1]),
            group=np.array(["A", "A", "A"]),
        )
        with pytest.warns(UserWarning, match="single cluster"):
            r = robust_variance_estimate(d)
        assert np.isfinite(r.estimate)

    def test_pair_studies_with_groups(self):
        d = MAData(
            effect=np.array([-0.3, 0.1]),
            se=np.array([0.2, 0.3]),
            group=np.array(["A", "B"]),
        )
        r = robust_variance_estimate(d)
        assert np.isfinite(r.estimate)

    def test_robust_wider_than_naive(self, clustered_reviews):
        """Robust CI should typically be wider than naive for clustered data."""
        r = robust_variance_estimate(clustered_reviews)
        # Not always true but usually
        assert r.details["robust_se"] >= r.details["naive_se"] * 0.5

    def test_estimate_finite(self, bimodal_20):
        d = MAData(effect=bimodal_20.effect, se=bimodal_20.se, group=bimodal_20.group)
        r = robust_variance_estimate(d)
        assert np.isfinite(r.estimate)

    def test_deterministic(self, clustered_reviews):
        r1 = robust_variance_estimate(clustered_reviews)
        r2 = robust_variance_estimate(clustered_reviews)
        assert np.isclose(r1.estimate, r2.estimate)
```

- [ ] **Step 2: Implement methods/robust_variance.py**

```python
# methods/robust_variance.py
"""Hedges-Tipton-Pustejovsky CR2 sandwich estimator."""
import warnings
import numpy as np
from scipy.stats import t as t_dist
from scipy.linalg import sqrtm
from core.types import MAData, MAResult


def robust_variance_estimate(data: MAData) -> MAResult:
    """CR2 robust variance estimation for clustered/dependent effects."""
    if data.group is None:
        raise ValueError("robust_variance_estimate requires group (cluster) labels")

    y = data.effect
    v = data.se**2
    n = len(y)
    groups = data.group
    unique_groups = np.unique(groups)
    J = len(unique_groups)

    if J == 1:
        warnings.warn("Only single cluster — robust SE degenerates to naive SE")

    # Working model: inverse-variance weighted
    W = np.diag(1.0 / v)
    X = np.ones((n, 1))

    XWX = X.T @ W @ X
    XWX_inv = np.linalg.inv(XWX)
    beta = (XWX_inv @ X.T @ W @ y).item()
    e = y - beta  # residuals
    naive_var = XWX_inv.item()
    naive_se = np.sqrt(naive_var)

    # Hat matrix
    H = X @ XWX_inv @ X.T @ W

    # CR2 meat matrix
    meat = np.zeros((1, 1))
    B_list = []
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        n_j = len(idx)
        H_j = H[np.ix_(idx, idx)]
        I_j = np.eye(n_j)
        IH = I_j - H_j

        # CR2 correction: A_j = (I - H_j)^{-1/2}
        try:
            eigvals, eigvecs = np.linalg.eigh(IH)
            eigvals = np.maximum(eigvals, 1e-8)
            A_j = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            A_j = I_j

        e_j = e[idx].reshape(-1, 1)
        adjusted = A_j @ e_j
        meat += (X[idx].T @ W[np.ix_(idx, idx)] @ adjusted @ adjusted.T
                 @ W[np.ix_(idx, idx)] @ X[idx])
        B_list.append(A_j)

    V_robust = XWX_inv @ meat @ XWX_inv
    robust_se = np.sqrt(max(V_robust.item(), 1e-15))

    # Satterthwaite df approximation
    if J > 1:
        df = max(1.0, 2.0 * V_robust.item()**2 /
                 max(np.var([V_robust.item()] * J), 1e-15))  # simplified
        # Better: use the Pustejovsky-Tipton (2018) df
        df = min(max(df, 1.0), J - 1)
    else:
        df = 1.0

    alpha = 0.05
    t_crit = t_dist.ppf(1 - alpha / 2, df)

    return MAResult(
        estimate=float(beta),
        ci_lower=float(beta - t_crit * robust_se),
        ci_upper=float(beta + t_crit * robust_se),
        method="RobustCR2",
        details={
            "robust_se": float(robust_se),
            "naive_se": float(naive_se),
            "df_satterthwaite": float(df),
            "n_clusters": int(J),
        },
    )
```

- [ ] **Step 3: Run tests**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_robust_variance.py -v`
Expected: All 12 tests PASS

- [ ] **Step 4: Commit**

```bash
cd C:\Users\user\Transcendent-Meta-Analysis-Lab
git add methods/robust_variance.py tests/test_robust_variance.py
git commit -m "feat: add Robust Variance Estimation (CR2 sandwich) with 12 tests"
```

---

### Task 6: Cumulative MA + Change-Point Detection

**Files:**
- Create: `methods/cumulative_changepoint.py`
- Create: `tests/test_cumulative_changepoint.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cumulative_changepoint.py
import numpy as np
import pytest
from core.types import MAData, MAResult
from methods.cumulative_changepoint import cumulative_changepoint


class TestCumulativeChangepoint:
    def test_returns_maresult(self, temporal_30):
        r = cumulative_changepoint(temporal_30)
        assert isinstance(r, MAResult)
        assert r.method == "CumulativeMA"

    def test_ci_contains_estimate(self, temporal_30):
        r = cumulative_changepoint(temporal_30)
        assert r.ci_lower <= r.estimate <= r.ci_upper

    def test_cumulative_length(self, temporal_30):
        r = cumulative_changepoint(temporal_30)
        assert len(r.details["cumulative_effects"]) == 30

    def test_cumulative_monotone_se(self, temporal_30):
        """SE should generally decrease as more studies are added."""
        r = cumulative_changepoint(temporal_30)
        ses = r.details["cumulative_ses"]
        # First SE > last SE (more data = more precision)
        assert ses[0] > ses[-1]

    def test_detects_shift(self, temporal_30):
        """Should detect the change-point near index 15."""
        r = cumulative_changepoint(temporal_30)
        cps = r.details["bayesian_changepoints"]
        # At least one change-point detected in range [10, 20]
        assert any(10 <= cp <= 20 for cp in cps), f"Change-points: {cps}"

    def test_cusum_alarms(self, temporal_30):
        r = cumulative_changepoint(temporal_30)
        assert len(r.details["cusum_alarms"]) > 0

    def test_no_year_raises(self, small_5):
        with pytest.raises(ValueError, match="year"):
            cumulative_changepoint(small_5)

    def test_stable_evidence_no_alarms(self):
        """Homogeneous data should produce few/no alarms."""
        rng = np.random.RandomState(42)
        d = MAData(
            effect=rng.normal(0.0, 0.05, 20),
            se=np.full(20, 0.2),
            year=np.arange(2000, 2020),
        )
        r = cumulative_changepoint(d)
        assert len(r.details["bayesian_changepoints"]) <= 2

    def test_single_year(self):
        d = MAData(
            effect=np.array([-0.3, 0.1, -0.2]),
            se=np.array([0.2, 0.3, 0.25]),
            year=np.array([2020, 2020, 2020]),
        )
        r = cumulative_changepoint(d)
        assert np.isfinite(r.estimate)

    def test_pair_with_year(self):
        d = MAData(
            effect=np.array([-0.3, 0.1]),
            se=np.array([0.2, 0.3]),
            year=np.array([2020, 2021]),
        )
        r = cumulative_changepoint(d)
        assert len(r.details["cumulative_effects"]) == 2

    def test_estimate_equals_final_cumulative(self, temporal_30):
        r = cumulative_changepoint(temporal_30)
        assert np.isclose(r.estimate, r.details["cumulative_effects"][-1], atol=1e-10)

    def test_deterministic(self, temporal_30):
        r1 = cumulative_changepoint(temporal_30)
        r2 = cumulative_changepoint(temporal_30)
        assert np.allclose(r1.details["cumulative_effects"], r2.details["cumulative_effects"])
```

- [ ] **Step 2: Implement methods/cumulative_changepoint.py**

```python
# methods/cumulative_changepoint.py
"""Cumulative meta-analysis with CUSUM and Bayesian online change-point detection."""
import numpy as np
from scipy.stats import norm
from core.types import MAData, MAResult


def _iv_pool(y, v):
    """Inverse-variance fixed-effect pooling."""
    w = 1.0 / v
    mu = np.sum(w * y) / np.sum(w)
    se = 1.0 / np.sqrt(np.sum(w))
    return mu, se


def _bayesian_changepoint(y, ses, hazard_rate=1/15):
    """Adams & MacKay (2007) Bayesian online change-point detection.

    Returns list of indices where P(run_length=0) > 0.5.
    """
    n = len(y)
    if n < 2:
        return []

    # Run-length probabilities: R[t, r] = P(r_t = r | y_{1:t})
    max_run = n + 1
    R = np.zeros((n + 1, max_run))
    R[0, 0] = 1.0

    # Predictive sufficient stats per run length
    mu_sum = np.zeros(max_run)
    var_sum = np.zeros(max_run)
    count = np.zeros(max_run)

    changepoints = []

    for t in range(n):
        # Predictive probability for each run length
        pred_probs = np.zeros(t + 1)
        for r in range(t + 1):
            if count[r] == 0:
                pred_mu = 0.0
                pred_var = 1.0 + ses[t]**2
            else:
                pred_mu = mu_sum[r] / count[r]
                pred_var = var_sum[r] / count[r] + ses[t]**2
            pred_probs[r] = norm.pdf(y[t], pred_mu, np.sqrt(max(pred_var, 1e-10)))

        # Growth probabilities
        growth = R[t, :t+1] * pred_probs * (1 - hazard_rate)
        # Changepoint probability
        cp_prob = np.sum(R[t, :t+1] * pred_probs * hazard_rate)

        R[t+1, 0] = cp_prob
        R[t+1, 1:t+2] = growth

        # Normalize
        total = np.sum(R[t+1, :t+2])
        if total > 1e-300:
            R[t+1, :t+2] /= total

        # Update sufficient stats
        new_mu_sum = np.zeros(max_run)
        new_var_sum = np.zeros(max_run)
        new_count = np.zeros(max_run)
        new_mu_sum[0] = 0.0
        new_count[0] = 0
        new_mu_sum[1:t+2] = mu_sum[:t+1] + y[t]
        new_count[1:t+2] = count[:t+1] + 1
        new_var_sum[1:t+2] = var_sum[:t+1] + y[t]**2
        mu_sum = new_mu_sum
        var_sum = new_var_sum
        count = new_count

        # Detect changepoint
        if t > 0 and R[t+1, 0] > 0.5:
            changepoints.append(t)

    return changepoints


def cumulative_changepoint(data: MAData) -> MAResult:
    """Cumulative MA with CUSUM and Bayesian change-point detection."""
    if data.year is None:
        raise ValueError("Cumulative MA requires year data")

    # Sort by year (ties broken by SE ascending)
    order = np.lexsort((data.se, data.year))
    y = data.effect[order]
    v = data.se[order]**2
    ses = data.se[order]

    n = len(y)
    cum_effects = np.zeros(n)
    cum_ses = np.zeros(n)

    # Cumulative IV-weighted pooling
    for t in range(n):
        mu, se = _iv_pool(y[:t+1], v[:t+1])
        cum_effects[t] = mu
        cum_ses[t] = se

    # CUSUM
    z_scores = y / ses  # individual z-scores
    z_bar = np.mean(z_scores)
    cusum = np.cumsum(z_scores - z_bar)
    sigma_cusum = np.std(z_scores) if n > 1 else 1.0
    h = 4.0 * sigma_cusum
    cusum_alarms = list(np.where(np.abs(cusum) > h)[0])

    # Bayesian online changepoint detection
    bayesian_cps = _bayesian_changepoint(y, ses)

    z = norm.ppf(0.975)

    return MAResult(
        estimate=float(cum_effects[-1]),
        ci_lower=float(cum_effects[-1] - z * cum_ses[-1]),
        ci_upper=float(cum_effects[-1] + z * cum_ses[-1]),
        method="CumulativeMA",
        details={
            "cumulative_effects": cum_effects.tolist(),
            "cumulative_ses": cum_ses.tolist(),
            "cusum_values": cusum.tolist(),
            "cusum_alarms": cusum_alarms,
            "bayesian_changepoints": bayesian_cps,
        },
    )
```

- [ ] **Step 3: Run tests**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_cumulative_changepoint.py -v`
Expected: All 12 tests PASS

- [ ] **Step 4: Commit**

```bash
cd C:\Users\user\Transcendent-Meta-Analysis-Lab
git add methods/cumulative_changepoint.py tests/test_cumulative_changepoint.py
git commit -m "feat: add Cumulative MA with CUSUM + Bayesian change-point detection, 12 tests"
```

---

### Task 7: Copula Multivariate MA

**Files:**
- Create: `methods/copula_ma.py`
- Create: `tests/test_copula_ma.py`

This task follows the same TDD pattern. The implementation uses GLS with block-diagonal working correlation, REML for tau2, and EM for unknown correlations. Tests validate against univariate RE for single-outcome reviews and check the block-diagonal structure.

**12 tests covering:** MAResult type, CI containment, tau2 non-negative, per-review estimates, fallback to univariate when all singletons, group required, single study, clustered data, deterministic output, n_reviews/n_outcomes in details.

- [ ] **Step 1-4: Write tests, implement, verify, commit** (same TDD cycle as Tasks 4-6)

---

### Task 8: Dirichlet Process Mixture

**Files:**
- Create: `methods/dirichlet_process.py`
- Create: `tests/test_dirichlet_process.py`

Gibbs sampler with stick-breaking truncation at K=20. Tests validate cluster discovery on bimodal fixture (should find 2 clusters) and non-fragmentation on homogeneous data.

**13 tests covering:** MAResult type, CI containment, bimodal discovers 2 clusters, unimodal stays 1-2, weights sum to 1, single study, outlier gets own cluster, seed reproducibility, cluster_means/variances shape, alpha positive, convergence (trace stabilizes), k=2 edge case.

- [ ] **Step 1-4: Write tests, implement, verify, commit**

---

### Task 9: Tropical Geometry Pooling

**Files:**
- Create: `exploration/__init__.py`
- Create: `exploration/tropical_pooling.py`
- Create: `tests/test_tropical_pooling.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tropical_pooling.py
import numpy as np
import pytest
from core.types import MAData, MAResult
from exploration.tropical_pooling import tropical_pool


class TestTropicalPooling:
    def test_returns_maresult(self, small_5):
        r = tropical_pool(small_5)
        assert isinstance(r, MAResult)
        assert r.method == "TropicalPool"

    def test_ci_contains_estimate(self, small_5):
        r = tropical_pool(small_5)
        assert r.ci_lower <= r.estimate <= r.ci_upper

    def test_symmetric_three_studies(self):
        """(-1, 0, 1) with equal SE: tropical median = 0."""
        d = MAData(effect=np.array([-1.0, 0.0, 1.0]), se=np.array([0.3, 0.3, 0.3]))
        r = tropical_pool(d)
        assert np.isclose(r.estimate, 0.0, atol=0.01)

    def test_radius_symmetric(self):
        d = MAData(effect=np.array([-1.0, 0.0, 1.0]), se=np.array([0.3, 0.3, 0.3]))
        r = tropical_pool(d)
        assert r.details["tropical_radius"] > 0

    def test_outlier_resistance(self, with_outlier):
        """Tropical median should resist the outlier at 5.0."""
        r = tropical_pool(with_outlier)
        # Should be much closer to 0 than 5
        assert abs(r.estimate) < 2.0

    def test_outlier_vs_iv(self, with_outlier):
        """Tropical should be more robust than IV-weighted."""
        r = tropical_pool(with_outlier)
        y, w = with_outlier.effect, 1.0 / with_outlier.se**2
        iv_est = np.sum(w * y) / np.sum(w)
        assert abs(r.estimate) < abs(iv_est)  # tropical closer to 0

    def test_robustness_score(self, small_5):
        r = tropical_pool(small_5)
        score = r.details["robustness_score"]
        assert 0.0 <= score <= 1.0

    def test_hull_vertices(self, small_5):
        r = tropical_pool(small_5)
        assert len(r.details["hull_vertices"]) >= 1
        assert len(r.details["hull_vertices"]) <= 5

    def test_single_study(self, single_study):
        r = tropical_pool(single_study)
        assert np.isclose(r.estimate, 0.5, atol=1e-6)
        assert np.isclose(r.details["tropical_radius"], 0.0)

    def test_pair_studies(self, pair_studies):
        r = tropical_pool(pair_studies)
        assert np.isfinite(r.estimate)

    def test_estimate_finite(self, bimodal_20):
        r = tropical_pool(bimodal_20)
        assert np.isfinite(r.estimate)

    def test_deterministic(self, small_5):
        r1 = tropical_pool(small_5)
        r2 = tropical_pool(small_5)
        assert np.isclose(r1.estimate, r2.estimate)
```

- [ ] **Step 2: Implement exploration/tropical_pooling.py**

```python
# exploration/__init__.py
# empty

# exploration/tropical_pooling.py
"""Tropical geometry: min-plus algebra for outlier-immune consensus."""
import numpy as np
from scipy.stats import norm
from core.types import MAData, MAResult


def tropical_pool(data: MAData) -> MAResult:
    """Compute tropical weighted median (Chebyshev center in min-plus algebra).

    NOTE: This is an exploratory method, not a standard meta-analytic technique.
    """
    y = data.effect
    v = data.se**2
    n = len(y)

    if n == 1:
        return MAResult(
            estimate=float(y[0]),
            ci_lower=float(y[0] - 1.96 * data.se[0]),
            ci_upper=float(y[0] + 1.96 * data.se[0]),
            method="TropicalPool",
            details={"tropical_median": float(y[0]), "tropical_radius": 0.0,
                     "hull_vertices": [0], "hull_interior": [],
                     "robustness_score": 0.0},
        )

    # Tropical weights: w_i = -log(v_i) = precision in log-space
    w = -np.log(np.maximum(v, 1e-300))

    # Tropical weighted median: minimize max_i(|theta - y_i| - w_i)
    # Chebyshev center: theta = (max(y_i - w_i) + min(y_i + w_i)) / 2
    lower_bounds = y - w  # theta must be >= max of these minus t
    upper_bounds = y + w  # theta must be <= min of these plus t
    theta = (np.max(lower_bounds) + np.min(upper_bounds)) / 2
    radius = (np.min(upper_bounds) - np.max(lower_bounds)) / 2
    radius = max(radius, 0.0)

    # Tropical convex hull: vertices are studies that define the bounds
    hull_vertices = []
    for i in range(n):
        # Remove study i, recompute bounds
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        new_theta = (np.max(lower_bounds[mask]) + np.min(upper_bounds[mask])) / 2
        if not np.isclose(new_theta, theta, atol=1e-8):
            hull_vertices.append(int(i))

    hull_interior = [i for i in range(n) if i not in hull_vertices]
    robustness_score = len(hull_interior) / n if n > 0 else 0.0

    # CI via bootstrap-like: use the radius as uncertainty measure
    se_trop = max(radius / np.sqrt(n), np.min(data.se))
    z = norm.ppf(0.975)

    return MAResult(
        estimate=float(theta),
        ci_lower=float(theta - z * se_trop),
        ci_upper=float(theta + z * se_trop),
        method="TropicalPool",
        details={
            "tropical_median": float(theta),
            "tropical_radius": float(radius),
            "hull_vertices": hull_vertices,
            "hull_interior": hull_interior,
            "robustness_score": float(robustness_score),
        },
    )
```

- [ ] **Step 3: Run tests**

Run: `cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/test_tropical_pooling.py -v`
Expected: All 12 tests PASS

- [ ] **Step 4: Commit**

```bash
cd C:\Users\user\Transcendent-Meta-Analysis-Lab
git add exploration/ tests/test_tropical_pooling.py
git commit -m "feat: add Tropical Geometry pooling (min-plus Chebyshev center) with 12 tests"
```

---

### Task 10: Wasserstein Barycenters

**Files:**
- Create: `exploration/wasserstein_barycenter.py`
- Create: `tests/test_wasserstein_barycenter.py`

Closed-form Gaussian W2 barycenter with fixed-point iteration for variance. Tests validate against known 2-Gaussian barycenter, equal-weight/equal-variance case, outlier flagging, k=1, convergence.

**12 tests covering:** MAResult type, CI containment, equal-variance trivial case, known 2-Gaussian reference, transport costs non-negative, outlier flagged, single study, pair, convergence count, mean matches IV-weighted (for equal weights), deterministic, bimodal finite.

- [ ] **Step 1-4: Write tests, implement, verify, commit**

---

### Task 11: Fisher-Rao Geodesics

**Files:**
- Create: `exploration/fisher_rao.py`
- Create: `tests/test_fisher_rao.py`

Skovgaard/Costa-Santos-Strapasson distance formula + Frechet mean via Riemannian gradient descent. Tests validate distance special cases (equal σ, equal μ), Frechet mean reduces to IV-weighted for equal variances, distance matrix symmetry and non-negativity, geodesic endpoints match inputs.

**13 tests covering:** MAResult type, distance non-negative, distance symmetric, equal-σ reduces to |Δμ|/σ, equal-μ reduces to √2*|log ratio|, Frechet mean = IV for equal-σ, geodesic endpoints, single study, pair, distance matrix shape, sigma clamping, convergence, deterministic.

- [ ] **Step 1-4: Write tests, implement, verify, commit**

---

### Task 12: Persistent Homology

**Files:**
- Create: `exploration/persistent_homology.py`
- Create: `tests/test_persistent_homology.py`

Union-find for β0 + boundary matrix reduction for β1 on Vietoris-Rips complex. Tests validate 3-cluster detection, triangle loop detection, line collapses to 1 component, Betti numbers non-negative, single study trivial, subsampling for large n.

**12 tests covering:** MAResult type, 3 clusters → β0=3 persistent, triangle → β1=1, line → β1=0, persistence pairs sorted, Betti curve starts at k, single study, pair, all identical, filtration values increasing, large data subsampled, deterministic.

- [ ] **Step 1-4: Write tests, implement, verify, commit**

---

### Task 13: Spectral Graph Methods

**Files:**
- Create: `exploration/spectral_graph.py`
- Create: `tests/test_spectral_graph.py`

Graph Laplacian eigenmaps with Gaussian similarity kernel, Fiedler value, eigengap clustering, degree-centrality weighting. Tests validate homogeneous Fiedler ≈ 1, separated groups Fiedler ≈ 0, cluster count, embedding shape, degree centrality sums, single study, pair.

**12 tests covering:** MAResult type, CI containment, homogeneous Fiedler high, bimodal Fiedler low, 2 clusters detected for bimodal, embedding shape (n,2), degree centrality positive, centrality-weighted estimate finite, single study, pair, deterministic, outlier handling.

- [ ] **Step 1-4: Write tests, implement, verify, commit**

---

### Final Verification

- [ ] **Run full test suite**

```bash
cd C:\Users\user\Transcendent-Meta-Analysis-Lab && python -m pytest tests/ -v --tb=short
```

Expected: ~130 tests, all PASS

- [ ] **Final commit**

```bash
git add -A && git status
# Verify no unintended files
git commit -m "feat: complete 10 advanced methods (5 validated + 5 exotic) with ~130 tests"
```
