"""Bayesian Model Averaging over four meta-analysis models.

Models:
  FE   – Fixed-Effect (IV-weighted mean, tau2=0, p=1)
  DL   – DerSimonian-Laird random effects (method of moments, p=2)
  REML – Restricted Maximum Likelihood random effects (Newton-Raphson, p=2)
  PM   – Paule-Mandel random effects (iterative Q-based, p=2)

BIC weights are used to combine the four model-level estimates.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.types import MAData, MAResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loglik_normal(y: np.ndarray, mu: float, sigma2: np.ndarray) -> float:
    """Log-likelihood of normal model: sum log N(y_i; mu, sigma2_i)."""
    return float(np.sum(stats.norm.logpdf(y, loc=mu, scale=np.sqrt(sigma2))))


def _iv_estimate(y: np.ndarray, w: np.ndarray):
    """Return (estimate, variance) from inverse-variance weights."""
    W = np.sum(w)
    mu = np.sum(w * y) / W
    var = 1.0 / W
    return mu, var


# ---------------------------------------------------------------------------
# Model 1: Fixed-Effect
# ---------------------------------------------------------------------------

def _fit_fe(y: np.ndarray, v: np.ndarray):
    """Fixed-effect model. tau2=0, p=1 parameter."""
    w = 1.0 / v
    mu, var = _iv_estimate(y, w)
    sigma2 = v  # study-level variance equals sampling variance
    loglik = _loglik_normal(y, mu, sigma2)
    return mu, var, 0.0, loglik


# ---------------------------------------------------------------------------
# Model 2: DerSimonian-Laird
# ---------------------------------------------------------------------------

def _dl_tau2(y: np.ndarray, v: np.ndarray) -> float:
    """Method-of-moments DerSimonian-Laird tau2 estimator (truncated at 0)."""
    k = len(y)
    w = 1.0 / v
    W = np.sum(w)
    W2 = np.sum(w ** 2)
    mu_fe = np.sum(w * y) / W
    Q = np.sum(w * (y - mu_fe) ** 2)
    c = W - W2 / W
    tau2 = max(0.0, (Q - (k - 1)) / c)
    return tau2


def _fit_dl(y: np.ndarray, v: np.ndarray):
    """DerSimonian-Laird random effects. p=2 parameters (mu, tau2)."""
    tau2 = _dl_tau2(y, v)
    w = 1.0 / (v + tau2)
    mu, var = _iv_estimate(y, w)
    sigma2 = v + tau2
    loglik = _loglik_normal(y, mu, sigma2)
    return mu, var, tau2, loglik


# ---------------------------------------------------------------------------
# Model 3: REML
# ---------------------------------------------------------------------------

def _reml_tau2(y: np.ndarray, v: np.ndarray,
               max_iter: int = 100, tol: float = 1e-8) -> float:
    """REML tau2 via Newton-Raphson (truncated at 0).

    Score: dREML/d(tau2) = 0
    Uses the standard iterative approach for normal-normal model.
    """
    # Start from DL estimate
    tau2 = max(0.0, _dl_tau2(y, v))

    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        W = np.sum(w)
        # REML first derivative
        mu = np.sum(w * y) / W
        r = y - mu
        # P = diag(w) - w w^T / W  (residual projection)
        # score = -0.5 * sum(P_ii) + 0.5 * r^T P^2 r  in scalar terms:
        Pw = w - w ** 2 / W
        score = -0.5 * np.sum(w - w ** 2 / W) + 0.5 * np.sum(w ** 2 * r ** 2) - 0.5 * (np.sum(w ** 2 * r)) ** 2 / W
        # Fisher information (expected second derivative, negated)
        info = 0.5 * np.sum(Pw ** 2)
        if info <= 0:
            break
        delta = score / info
        tau2_new = max(0.0, tau2 + delta)
        if abs(tau2_new - tau2) < tol:
            tau2 = tau2_new
            break
        tau2 = tau2_new

    return tau2


def _fit_reml(y: np.ndarray, v: np.ndarray):
    """REML random effects. p=2 parameters."""
    tau2 = _reml_tau2(y, v)
    w = 1.0 / (v + tau2)
    mu, var = _iv_estimate(y, w)
    sigma2 = v + tau2
    loglik = _loglik_normal(y, mu, sigma2)
    return mu, var, tau2, loglik


# ---------------------------------------------------------------------------
# Model 4: Paule-Mandel
# ---------------------------------------------------------------------------

def _pm_tau2(y: np.ndarray, v: np.ndarray,
             max_iter: int = 200, tol: float = 1e-8) -> float:
    """Paule-Mandel iterative Q-based tau2 estimator (truncated at 0)."""
    k = len(y)
    tau2 = max(0.0, _dl_tau2(y, v))  # warm start

    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        W = np.sum(w)
        mu = np.sum(w * y) / W
        Q = np.sum(w * (y - mu) ** 2)
        # Update: solve Q(tau2) = k-1
        # Gradient dQ/d(tau2) = -sum(w_i^2 * (y_i - mu)^2) (approx, ignoring mu change)
        dQ = -np.sum(w ** 2 * (y - mu) ** 2)
        if abs(dQ) < 1e-14:
            break
        delta = (Q - (k - 1)) / (-dQ)
        tau2_new = max(0.0, tau2 + delta)
        if abs(tau2_new - tau2) < tol:
            tau2 = tau2_new
            break
        tau2 = tau2_new

    # Clamp result
    return max(0.0, tau2)


def _fit_pm(y: np.ndarray, v: np.ndarray):
    """Paule-Mandel random effects. p=2 parameters."""
    tau2 = _pm_tau2(y, v)
    w = 1.0 / (v + tau2)
    mu, var = _iv_estimate(y, w)
    sigma2 = v + tau2
    loglik = _loglik_normal(y, mu, sigma2)
    return mu, var, tau2, loglik


# ---------------------------------------------------------------------------
# BIC computation
# ---------------------------------------------------------------------------

def _bic(loglik: float, p: int, k: int) -> float:
    """BIC = -2 * loglik + p * log(k)."""
    return -2.0 * loglik + p * np.log(k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bayesian_model_average(data: MAData) -> MAResult:
    """Bayesian Model Averaging over FE, DL, REML, and PM models.

    Parameters
    ----------
    data : MAData
        Input meta-analysis data (effect sizes + standard errors).

    Returns
    -------
    MAResult
        BMA-combined estimate, 95 % CI, and per-model details.
    """
    y = np.asarray(data.effect, dtype=float)
    se = np.asarray(data.se, dtype=float)
    v = se ** 2
    k = len(y)

    # ------------------------------------------------------------------ #
    # Edge case: single study — return it directly with FE weight = 1.0   #
    # ------------------------------------------------------------------ #
    if k == 1:
        est = float(y[0])
        ci_lo = est - 1.96 * float(se[0])
        ci_hi = est + 1.96 * float(se[0])
        models = [
            {"name": "FE",   "estimate": est, "tau2": 0.0, "bic": 0.0, "weight": 1.0},
            {"name": "DL",   "estimate": est, "tau2": 0.0, "bic": np.inf, "weight": 0.0},
            {"name": "REML", "estimate": est, "tau2": 0.0, "bic": np.inf, "weight": 0.0},
            {"name": "PM",   "estimate": est, "tau2": 0.0, "bic": np.inf, "weight": 0.0},
        ]
        return MAResult(
            estimate=est,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            method="BMA",
            details={
                "models": models,
                "bma_variance": float(v[0]),
            },
        )

    # ------------------------------------------------------------------ #
    # Fit the four models                                                  #
    # ------------------------------------------------------------------ #
    fe_mu,   fe_var,   fe_tau2,   fe_ll   = _fit_fe(y, v)
    dl_mu,   dl_var,   dl_tau2,   dl_ll   = _fit_dl(y, v)
    reml_mu, reml_var, reml_tau2, reml_ll = _fit_reml(y, v)
    pm_mu,   pm_var,   pm_tau2,   pm_ll   = _fit_pm(y, v)

    # p = 1 for FE (only mu), p = 2 for RE models (mu + tau2)
    fe_bic   = _bic(fe_ll,   p=1, k=k)
    dl_bic   = _bic(dl_ll,   p=2, k=k)
    reml_bic = _bic(reml_ll, p=2, k=k)
    pm_bic   = _bic(pm_ll,   p=2, k=k)

    bics = np.array([fe_bic, dl_bic, reml_bic, pm_bic])

    # ------------------------------------------------------------------ #
    # BIC weights via log-sum-exp for numerical stability                  #
    # ------------------------------------------------------------------ #
    log_unnorm = -0.5 * bics
    log_denom = np.logaddexp.reduce(log_unnorm)
    log_weights = log_unnorm - log_denom
    weights = np.exp(log_weights)
    # Guard against any residual numerical noise
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum()

    w_fe, w_dl, w_reml, w_pm = weights

    mus  = np.array([fe_mu,   dl_mu,   reml_mu, pm_mu])
    vrs  = np.array([fe_var,  dl_var,  reml_var, pm_var])
    tau2s = np.array([fe_tau2, dl_tau2, reml_tau2, pm_tau2])

    # ------------------------------------------------------------------ #
    # BMA combined estimate                                                #
    # ------------------------------------------------------------------ #
    bma_est = float(np.sum(weights * mus))

    # BMA variance (law of total variance)
    bma_var = float(np.sum(weights * (vrs + (mus - bma_est) ** 2)))

    bma_se = np.sqrt(bma_var)
    ci_lo = bma_est - 1.96 * bma_se
    ci_hi = bma_est + 1.96 * bma_se

    # ------------------------------------------------------------------ #
    # Build per-model summary                                              #
    # ------------------------------------------------------------------ #
    names = ["FE", "DL", "REML", "PM"]
    models = [
        {
            "name":     names[i],
            "estimate": float(mus[i]),
            "tau2":     float(tau2s[i]),
            "bic":      float(bics[i]),
            "weight":   float(weights[i]),
        }
        for i in range(4)
    ]

    return MAResult(
        estimate=bma_est,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        method="BMA",
        details={
            "models":       models,
            "bma_variance": bma_var,
        },
    )
