from __future__ import annotations

# ============================================================
# UTILITAIRES PARTAGÉS POUR LES VISUALISATIONS
# ============================================================
#
# Ce module fournit les fonctions communes à tous les graphiques :
# - chargement des données de référence
# - calcul vectorisé de R(g, x) sur une grille 2D
# - construction de la grille 2D centrée sur le design point
# ============================================================

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from src.config import G_COL
from src.paths import INPUTS, REPORTS, TABLES


# --------------------------------------------------------
# Fonctions utilitaires (identiques MVP / visualize_frontier)
# --------------------------------------------------------

def sigmoid(z):
    """Sigmoïde standard."""
    return 1.0 / (1.0 + np.exp(-z))


def smooth_unit_interval(val, center=0.5, scale=0.15):
    """Projection lisse dans (0,1)."""
    return 0.02 + 0.96 * sigmoid((val - center) / scale)


def inverse_smooth_unit_interval(u, center=0.5, scale=0.15):
    """Inverse de smooth_unit_interval."""
    u = np.clip(u, 0.020001, 0.979999)
    return center + scale * logit((u - 0.02) / 0.96)


def tail_default_prob(pd_arr, rho_arr, q):
    """
    Probabilité de défaut de queue ASRF / Vasicek.
        Phi((Phi^{-1}(PD) + sqrt(rho) * Phi^{-1}(q)) / sqrt(1-rho))
    """
    return norm.cdf(
        (norm.ppf(pd_arr) + np.sqrt(rho_arr) * norm.ppf(q)) / np.sqrt(1.0 - rho_arr)
    )


# --------------------------------------------------------
# Chargement des données nécessaires aux graphiques
# --------------------------------------------------------

def load_viz_data():
    """
    Charge les inputs et outputs nécessaires à toutes les visualisations.

    Retourne :
        exposures, capital, sector_params, design_point, sigma, summary
    """
    exposures     = pd.read_csv(INPUTS / "exposures.csv")
    capital       = pd.read_csv(INPUTS / "capital.csv").iloc[0]
    sector_params = pd.read_csv(INPUTS / "sector_params.csv")

    # Sigma
    sigma = pd.read_csv(TABLES / "Sigma.csv", index_col=0)
    feature_cols = list(sigma.columns)

    # Design point — on filtre sur les colonnes de Sigma
    dp_raw = pd.read_csv(TABLES / "design_point.csv", index_col=0, header=None)
    design_point = dp_raw.loc[dp_raw.index.isin(feature_cols)].squeeze().astype(float)

    # Summary
    import json
    with open(REPORTS / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    return exposures, capital, sector_params, design_point, sigma, summary


def compute_ratio_grid(s_dict, exposures, capital, sector_params):
    """
    Calcul vectorisé de R(s) pour un scénario s_dict.

    s_dict : dict {col_name: valeur} représentant le vecteur scénario.

    Retourne le ratio de capital R = CET1 / RWA.
    """
    params = sector_params.set_index("sector")

    PD0      = exposures["PD0"].values
    LGD0     = exposures["LGD0"].values
    EAD      = exposures["EAD"].values
    rho      = exposures["rho"].values
    alpha_rwa = exposures["alpha_rwa"].values
    sectors  = exposures["sector"].values

    delta_g = params.loc[sectors, "delta_g"].values
    eta_g   = params.loc[sectors, "eta_g"].values

    g_val = s_dict.get(G_COL, 0.0)

    z_pd  = logit(np.clip(PD0, 1e-12, 1 - 1e-12)) + delta_g * g_val
    z_lgd = inverse_smooth_unit_interval(LGD0) + eta_g * g_val

    for col, val in s_dict.items():
        if col == G_COL:
            continue
        b_col = f"b_{col}"
        c_col = f"c_{col}"
        if b_col in params.columns:
            z_pd  += np.nan_to_num(params.loc[sectors, b_col].values) * val
        if c_col in params.columns:
            z_lgd += np.nan_to_num(params.loc[sectors, c_col].values) * val

    PD_stress  = expit(z_pd)
    LGD_stress = smooth_unit_interval(z_lgd)

    tail_pd   = tail_default_prob(PD_stress, rho, capital["q"])
    Lq_stress = np.sum(EAD * LGD_stress * tail_pd)

    delta_Lq = Lq_stress - capital["Lq0_abs"]
    CET1     = capital["CET1_0"] - delta_Lq + capital["delta_non_credit"]
    RWA      = max(capital["RWA_0"] + np.sum(alpha_rwa * (PD_stress - PD0)), 1e-6)

    return CET1 / RWA


def make_grid(design_point, x_driver, y_driver, span=4.0, n_pts=80):
    """
    Construit une grille 2D (x_driver, y_driver) centrée sur le design point.

    Retourne :
        xx, yy, X_grid, Y_grid, inv_sigma (sur les colonnes de design_point)
    """
    x_center = float(design_point[x_driver])
    y_center = float(design_point[y_driver])
    span     = max(span, abs(x_center) * 3, abs(y_center) * 3)

    xx = np.linspace(-1, x_center + span, n_pts)
    yy = np.linspace(y_center - span, y_center + span, n_pts)

    return xx, yy, np.meshgrid(xx, yy)


def compute_Z_ratio_and_dist(X_grid, Y_grid, design_point, x_driver, y_driver,
                              exposures, capital, sector_params, inv_sigma):
    """
    Calcule R(g, x) et d²(g, x) sur la grille 2D.

    Les autres composantes du scénario sont fixées à leur valeur du design point.
    """
    Z_ratio = np.zeros_like(X_grid)
    Z_dist  = np.zeros_like(X_grid)

    feature_cols = list(design_point.index)

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            s_vec = design_point.copy()
            s_vec[x_driver] = X_grid[i, j]
            s_vec[y_driver] = Y_grid[i, j]

            s_dict = dict(s_vec)
            Z_ratio[i, j] = compute_ratio_grid(s_dict, exposures, capital, sector_params)

            s_vals = s_vec.values
            Z_dist[i, j] = s_vals @ inv_sigma @ s_vals

    return Z_ratio, Z_dist
