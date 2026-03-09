from __future__ import annotations

# ============================================================
# ETAPE 2 — CHARGER LES INPUTS CREDIT / CAPITAL
# ============================================================
#
# On charge :
# - exposures.csv
# - capital.csv
# - sector_params.csv
#
# Ces trois fichiers sont la couche d'inputs qu'on a construite avant.
# ============================================================

import pandas as pd

from src.paths import INPUTS


def load_inputs():
    """
    On charge :
    - exposures.csv
    - capital.csv
    - sector_params.csv

    Ces trois fichiers sont la couche d'inputs qu'on a construite avant.
    """

    exp_path = INPUTS / "exposures.csv"
    cap_path = INPUTS / "capital.csv"
    sec_path = INPUTS / "sector_params.csv"

    if not exp_path.exists():
        raise FileNotFoundError(f"{exp_path} introuvable")
    if not cap_path.exists():
        raise FileNotFoundError(f"{cap_path} introuvable")
    if not sec_path.exists():
        raise FileNotFoundError(f"{sec_path} introuvable")

    exposures    = pd.read_csv(exp_path)
    capital      = pd.read_csv(cap_path).iloc[0]
    sector_params = pd.read_csv(sec_path)

    required_exp = {"id", "sector", "EAD", "PD0", "LGD0", "rho", "M", "alpha_rwa"}
    required_cap = {"CET1_0", "RWA_0", "R0", "R_omega", "q", "delta_non_credit", "Lq0_abs"}
    required_sec = {"sector", "delta_g", "eta_g"}

    missing_exp = required_exp - set(exposures.columns)
    missing_cap = required_cap - set(capital.index)
    missing_sec = required_sec - set(sector_params.columns)

    if missing_exp:
        raise ValueError(f"Colonnes manquantes dans exposures.csv : {missing_exp}")
    if missing_cap:
        raise ValueError(f"Colonnes manquantes dans capital.csv : {missing_cap}")
    if missing_sec:
        raise ValueError(f"Colonnes manquantes dans sector_params.csv : {missing_sec}")

    return exposures, capital, sector_params
