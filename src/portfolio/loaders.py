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

import numpy as np
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

    exposures = pd.read_csv(exp_path)
    capital = pd.read_csv(cap_path).iloc[0]
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

    exp_numeric_cols = ["EAD", "PD0", "LGD0", "rho", "M", "alpha_rwa"]
    cap_numeric_cols = ["CET1_0", "RWA_0", "R0", "R_omega", "q", "delta_non_credit", "Lq0_abs"]
    sec_numeric_cols = [
        c for c in sector_params.columns
        if c in {"delta_g", "eta_g"} or c.startswith("b_") or c.startswith("c_")
    ]

    exposures[exp_numeric_cols] = exposures[exp_numeric_cols].apply(pd.to_numeric, errors="coerce")
    capital.loc[cap_numeric_cols] = pd.to_numeric(capital.loc[cap_numeric_cols], errors="coerce")
    if sec_numeric_cols:
        sector_params[sec_numeric_cols] = sector_params[sec_numeric_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

    if exposures[exp_numeric_cols].isna().any().any():
        raise ValueError("Valeurs manquantes ou non numeriques dans les colonnes quantitatives de exposures.csv")
    if capital.loc[cap_numeric_cols].isna().any():
        raise ValueError("Valeurs manquantes ou non numeriques dans les colonnes quantitatives de capital.csv")
    if sec_numeric_cols and sector_params[sec_numeric_cols].isna().any().any():
        raise ValueError("Valeurs manquantes ou non numeriques dans les colonnes quantitatives de sector_params.csv")

    if exposures["sector"].isna().any() or exposures["id"].isna().any():
        raise ValueError("Les colonnes id et sector de exposures.csv doivent etre renseignees")
    if sector_params["sector"].isna().any():
        raise ValueError("La colonne sector de sector_params.csv doit etre renseignee")
    if sector_params["sector"].duplicated().any():
        duplicates = sorted(sector_params.loc[sector_params["sector"].duplicated(), "sector"].unique())
        raise ValueError(f"Secteurs dupliques dans sector_params.csv : {duplicates}")

    if not np.isfinite(exposures["EAD"]).all() or (exposures["EAD"] <= 0.0).any():
        raise ValueError("EAD doit etre strictement positif pour toutes les expositions")
    if not exposures["PD0"].between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("PD0 doit etre strictement dans (0,1)")
    if not exposures["LGD0"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("LGD0 doit etre dans [0,1]")
    if not exposures["rho"].between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("rho doit etre strictement dans (0,1)")
    if not np.isfinite(exposures["M"]).all() or (exposures["M"] <= 0.0).any():
        raise ValueError("M doit etre strictement positif")
    if not np.isfinite(exposures["alpha_rwa"]).all() or (exposures["alpha_rwa"] < 0.0).any():
        raise ValueError("alpha_rwa doit etre fini et non negatif")

    q_val = float(capital["q"])
    if not np.isfinite(q_val) or not (0.0 < q_val < 1.0):
        raise ValueError("q doit etre strictement dans (0,1)")
    for col in ["CET1_0", "RWA_0", "R0", "R_omega", "delta_non_credit", "Lq0_abs"]:
        val = float(capital[col])
        if not np.isfinite(val):
            raise ValueError(f"{col} doit etre fini")
    if float(capital["RWA_0"]) <= 0.0:
        raise ValueError("RWA_0 doit etre strictement positif")

    return exposures, capital, sector_params
