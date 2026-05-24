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

import json

import numpy as np
import pandas as pd

from src.config import G_COL
from src.model.engine import build_reverse_stress_engine
from src.paths import REPORTS, TABLES
from src.portfolio.loaders import load_inputs


# --------------------------------------------------------
# Chargement des données nécessaires aux graphiques
# --------------------------------------------------------

def load_viz_context():
    """Load the central engine plus the saved optimization outputs used by figures."""
    exposures, capital, sector_params = load_inputs()
    sigma = pd.read_csv(TABLES / "Sigma.csv", index_col=0)
    feature_cols = list(sigma.columns)

    dp_raw = pd.read_csv(TABLES / "design_point.csv", index_col=0, header=None)
    design_point = dp_raw.loc[dp_raw.index.isin(feature_cols)].squeeze().astype(float)

    with open(REPORTS / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    engine = build_reverse_stress_engine(feature_cols, exposures, capital, sector_params)
    return engine, design_point, sigma, summary


def load_viz_data():
    """
    Backward-compatible helper kept for figures that only need raw inputs.
    """
    engine, design_point, sigma, summary = load_viz_context()
    return engine.exposures.copy(), engine.capital.copy(), engine.sector_params.copy(), design_point, sigma, summary


def compute_ratio_grid(s_dict, engine, anchor=None):
    """
    Calcul vectorisé de R(s) pour un scénario s_dict.

    s_dict : dict {col_name: valeur} représentant le vecteur scénario.

    Retourne le ratio de capital R = CET1 / RWA.
    """
    s_vec = engine.scenario_from_mapping(s_dict, anchor=anchor)
    return float(engine.evaluate(s_vec, include_exposures=False)["R"])


def make_grid(design_point, x_driver, y_driver, span=4.0, n_pts=80):
    """
    Construit une grille 2D (x_driver, y_driver) centrée sur le design point.

    Le rayon de la grille est élargi à `max(span, 3 * |x_center|, 3 * |y_center|)`
    pour garantir que le design point reste visible même quand ses coordonnées
    sont grandes.

    Retourne
    -------
    xx : ndarray (n_pts,)
        Vecteur 1D des abscisses (de -1 à x_center + span).
    yy : ndarray (n_pts,)
        Vecteur 1D des ordonnées (de y_center - span à y_center + span).
    meshgrid : tuple (X_grid, Y_grid)
        Tuple retourné par `np.meshgrid(xx, yy)`. Les appelants peuvent
        unpacker via `xx, yy, (X_grid, Y_grid) = make_grid(...)`.
    """
    x_center = float(design_point[x_driver])
    y_center = float(design_point[y_driver])
    span     = max(span, abs(x_center) * 3, abs(y_center) * 3)

    xx = np.linspace(-1, x_center + span, n_pts)
    yy = np.linspace(y_center - span, y_center + span, n_pts)

    return xx, yy, np.meshgrid(xx, yy)


def compute_Z_ratio_and_dist(X_grid, Y_grid, design_point, x_driver, y_driver, engine, inv_sigma):
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
            Z_ratio[i, j] = compute_ratio_grid(s_dict, engine, anchor=design_point.values)

            s_vals = s_vec.values
            Z_dist[i, j] = s_vals @ inv_sigma @ s_vals

    return Z_ratio, Z_dist
