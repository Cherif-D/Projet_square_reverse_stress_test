from __future__ import annotations

import numpy as np
from scipy.stats import norm

PROB_EPS = 1e-12
RHO_EPS = 1e-10


def validate_probability_scalar(value: float, name: str) -> float:
    """Validate that a scalar probability lies strictly inside (0, 1)."""
    val = float(value)
    if not np.isfinite(val) or not (0.0 < val < 1.0):
        raise ValueError(f"{name} doit etre strictement dans (0,1), recu={value!r}")
    return val


def bounded_probability(values, eps: float = PROB_EPS) -> np.ndarray:
    """Clip probability inputs away from 0 and 1 to keep ppf/cdf finite."""
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def bounded_correlation(values, eps: float = RHO_EPS) -> np.ndarray:
    """Clip asset correlations away from 0 and 1 for numerical stability."""
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def tail_default_prob(pd_vec: np.ndarray, rho_vec: np.ndarray, q: float) -> np.ndarray:
    """
    Queue default probability in the ASRF / Vasicek approximation.

    Phi((Phi^{-1}(PD) + sqrt(rho) * Phi^{-1}(q)) / sqrt(1-rho))
    """
    q_val = validate_probability_scalar(q, "q")
    pd_arr = bounded_probability(pd_vec)
    rho_arr = bounded_correlation(rho_vec)

    zq = norm.ppf(np.clip(q_val, PROB_EPS, 1.0 - PROB_EPS))
    arg = (norm.ppf(pd_arr) + np.sqrt(rho_arr) * zq) / np.sqrt(1.0 - rho_arr)
    return bounded_probability(norm.cdf(arg))
