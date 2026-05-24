from __future__ import annotations

# ============================================================
# SAUVEGARDE DES TABLES CSV
# ============================================================
#
# Ce module centralise toutes les écritures de fichiers CSV
# dans outputs/tables/.
# ============================================================

import numpy as np
import pandas as pd

from src.paths import TABLES


def save_sigma_csv(Sigma: np.ndarray, feature_cols: list[str]) -> None:
    """Sauvegarde la matrice de covariance Sigma.csv."""
    pd.DataFrame(Sigma, index=feature_cols, columns=feature_cols).to_csv(TABLES / "Sigma.csv")


def save_scenario_standardized(z: pd.DataFrame) -> None:
    """Sauvegarde le scénario standardisé."""
    z.to_csv(TABLES / "scenario_standardized.csv", index=False)


def save_baseline_exposure_metrics(baseline_stressed: pd.DataFrame) -> None:
    """Sauvegarde les métriques d'exposition à la baseline (s=0)."""
    baseline_stressed.to_csv(TABLES / "baseline_exposure_metrics.csv", index=False)


def save_design_point(s_star: np.ndarray, feature_cols: list[str]) -> None:
    """Sauvegarde le design point (vecteur s*)."""
    pd.Series(s_star, index=feature_cols).to_csv(TABLES / "design_point.csv")


def save_exposure_stress_at_design_point(stressed_df: pd.DataFrame) -> None:
    """Sauvegarde les expositions stressées au design point."""
    stressed_df.to_csv(TABLES / "exposure_stress_at_design_point.csv", index=False)


def save_sector_diagnostics(sector_diag: pd.DataFrame) -> None:
    """Sauvegarde les diagnostics sectoriels au design point."""
    sector_diag.to_csv(TABLES / "sector_diagnostics_at_design_point.csv", index=False)


def save_candidate_pool(pool: pd.DataFrame) -> None:
    """Sauvegarde le pool de scénarios candidats."""
    pool.to_csv(TABLES / "candidate_pool.csv", index=False)


def save_shortlist(shortlist: pd.DataFrame) -> None:
    """Sauvegarde la shortlist de scénarios de gouvernance."""
    shortlist.to_csv(TABLES / "scenario_shortlist.csv", index=False)
