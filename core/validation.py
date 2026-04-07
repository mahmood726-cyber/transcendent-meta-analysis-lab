import numpy as np
from scipy.special import logsumexp as _logsumexp

def require_min_studies(data, k):
    if len(data.effect) < k:
        raise ValueError(f"Method requires at least {k} studies, got {len(data.effect)}")

def safe_exp(x):
    return np.exp(np.clip(x, -700, 700))

def safe_log(x):
    return np.log(np.clip(x, 1e-300, None))

def logsumexp_mean(x):
    return _logsumexp(x) - np.log(len(x))

def ensure_positive_definite(M, epsilon=1e-8):
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def check_convergence(values, tol, n=3):
    if len(values) < max(n, 2):
        return False
    recent = values[-n:]
    return max(recent) - min(recent) < tol
