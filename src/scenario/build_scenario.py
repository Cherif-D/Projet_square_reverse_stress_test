from __future__ import annotations

# ============================================================
# ETAPE 1 — CONSTRUIRE LE SCENARIO s = (g, x)
# ============================================================
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 7 janvier 2026)
# ------------------------------------------------------------
# - eq. (5) : s = (g, x)^T
# - Section 6.1, étape 1 :
#   construire les drivers géopolitiques / macro-financiers
#   puis la covariance de référence Sigma.
# ============================================================

import pandas as pd

from src.config import TRANSFORMS
from src.paths import INTERIM, TABLES
from src.scenario.covariance import estimate_covariance, save_sigma
from src.scenario.transforms import safe_diff, safe_log_diff


def build_scenario_reference():
    """
    Construit le scénario standardisé et la covariance de référence Sigma.

    Quatre étapes :
      1. Charger `data/interim/macro.csv` (table harmonisée du tuteur).
      2. Appliquer les transformations de `TRANSFORMS` (logdiff ou diff
         selon la variable) pour produire les chocs `shock_*`.
      3. Standardiser (centrer / réduire) puis recentrer sur le dernier
         trimestre observé pour que `s = 0` corresponde à la baseline.
      4. Estimer Sigma sur les observations standardisées (Ledoit-Wolf
         si activé dans `config.USE_LEDOIT_WOLF`) et calculer son inverse
         ainsi que sa factorisation de Cholesky `Sigma = L L^T`.

    LIEN AVEC LE PAPIER
    -------------------
    - eq. (5) : s = (g, x)^T
    - Section 6.1, étape 1 :
      construire les drivers géopolitiques / macro-financiers
      puis la covariance de référence Sigma.

    Retourne
    --------
    tuple
        `z`, `feature_cols`, `Sigma`, `Sigma_inv`, `L`.
        La fonction écrit aussi `Sigma.csv`, `sigma_report.json` et
        `scenario_standardized.csv` dans `outputs/tables/`.
    """

    macro_path = INTERIM / "macro.csv"
    if not macro_path.exists():
        raise FileNotFoundError(
            f"{macro_path} introuvable. "
            "Le script principal s'attend à trouver la table harmonisée du tuteur dans data/interim/macro.csv."
        )

    macro = pd.read_csv(macro_path)
    if "Date" not in macro.columns:
        raise ValueError("macro.csv doit contenir une colonne Date")

    macro["Date"] = pd.to_datetime(macro["Date"])
    macro["quarter"] = macro["Date"].dt.to_period("Q")

    scenario = pd.DataFrame({"quarter": macro["quarter"]})

    for col, kind in TRANSFORMS.items():
        if col not in macro.columns:
            raise ValueError(f"Colonne manquante dans macro.csv : {col}")

        if kind == "logdiff":
            scenario[f"shock_{col}"] = safe_log_diff(macro[col])
        elif kind == "diff":
            scenario[f"shock_{col}"] = safe_diff(macro[col])
        else:
            raise ValueError(f"Transformation inconnue pour {col}: {kind}")

    scenario = scenario.dropna().reset_index(drop=True)
    feature_cols = [c for c in scenario.columns if c.startswith("shock_")]

    # --------------------------------------------------------
    # Standardisation
    # --------------------------------------------------------
    # On standardise toutes les composantes du scénario.
    #
    # Pourquoi ?
    # - pour rendre les coefficients de sector_params interprétables
    #   comme effets pour 1 écart-type de choc ;
    # - pour éviter qu'une variable domine juste à cause de son unité.
    # --------------------------------------------------------
    mu = scenario[feature_cols].mean()
    sd = scenario[feature_cols].std(ddof=1).replace(0.0, 1.0)

    z = scenario.copy()
    z[feature_cols] = (scenario[feature_cols] - mu) / sd

    # --------------------------------------------------------
    # Recentrage sur l'état courant
    # --------------------------------------------------------
    # On prend le dernier trimestre observé comme baseline.
    # Après recentrage :
    #   s = 0 correspond à l'état de départ de la banque.
    # --------------------------------------------------------
    s0 = z.iloc[-1][feature_cols]
    z[feature_cols] = z[feature_cols].sub(s0, axis=1)

    X = z[feature_cols].values

    # --------------------------------------------------------
    # Estimation de Sigma
    # --------------------------------------------------------
    Sigma, Sigma_inv, L = estimate_covariance(X, feature_cols)

    save_sigma(Sigma, feature_cols, n_obs=len(z))
    z.to_csv(TABLES / "scenario_standardized.csv", index=False)

    return z, feature_cols, Sigma, Sigma_inv, L
