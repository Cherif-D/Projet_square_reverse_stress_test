from __future__ import annotations

# ============================================================
# ESTIMATION DE LA COVARIANCE SIGMA
# ============================================================
#
# Ce module estime la matrice de covariance des chocs scénario.
#
# On utilise Ledoit-Wolf par défaut pour stabiliser l'estimation
# quand le nombre d'observations est limité.
#
# La régularisation +1e-12 * I garantit que Sigma est
# strictement définie positive (SPD) et inversible.
# ============================================================

import json

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.config import USE_LEDOIT_WOLF
from src.paths import REPORTS, TABLES


def estimate_covariance(X: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estime Sigma, Sigma_inv et L (Cholesky) à partir de la matrice
    d'observations X (n_obs x n_features).

    Retourne :
        Sigma     : matrice de covariance (n x n)
        Sigma_inv : inverse de Sigma
        L         : facteur de Cholesky de Sigma (tel que Sigma = L @ L.T)
    """
    if USE_LEDOIT_WOLF:
        lw = LedoitWolf()
        lw.fit(X)
        Sigma = lw.covariance_
    else:
        Sigma = np.cov(X, rowvar=False)

    Sigma = Sigma + 1e-12 * np.eye(len(feature_cols))
    Sigma_inv = np.linalg.inv(Sigma)
    L = np.linalg.cholesky(Sigma)

    return Sigma, Sigma_inv, L


def save_sigma(Sigma: np.ndarray, feature_cols: list[str], n_obs: int) -> None:
    """
    Sauvegarde Sigma.csv et sigma_report.json.
    """
    pd.DataFrame(Sigma, index=feature_cols, columns=feature_cols).to_csv(TABLES / "Sigma.csv")

    eigvals = np.linalg.eigvalsh(Sigma)
    cond_number = float(eigvals.max() / eigvals.min())

    sigma_report = {
        "feature_cols"   : feature_cols,
        "min_eigenvalue" : float(eigvals.min()),
        "max_eigenvalue" : float(eigvals.max()),
        "condition_number": cond_number,
        "is_spd"         : bool(np.all(eigvals > 0)),
        "n_observations" : int(n_obs),
    }

    with open(REPORTS / "sigma_report.json", "w", encoding="utf-8") as f:
        json.dump(sigma_report, f, indent=2, ensure_ascii=False)
