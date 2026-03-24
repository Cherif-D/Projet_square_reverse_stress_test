from __future__ import annotations

# ============================================================
# ETAPE 2 BIS — MAPPINGS PD_i(g,x) ET LGD_i(g,x)
# ============================================================
#
# LIEN AVEC LE PAPIER
# -------------------
# - eq. (7) / (8) : PD stressée en logit
# - eq. (10)      : LGD stressée en affine
# - Annexe A.2    : version sectorielle (b_k, d_k, c_k, e_k)
#
# Ici :
# - chaque exposition i hérite des sensibilités de son secteur k(i)
# - on reste donc dans une version parcimonieuse et robuste
# ============================================================

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.config import ENFORCE_STRESS_ONLY_FLOOR, G_COL


def sigmoid(z):
    """
    Sigmoïde standard.
    On l'utilise pour des transformations lisses.
    """
    return 1.0 / (1.0 + np.exp(-z))


def smooth_unit_interval(raw_value, center: float = 0.5, scale: float = 0.15):
    """
    Projection lisse dans (0,1).

    Pourquoi ?
    Parce que pour la LGD, on veut éviter un simple clip brutal
    qui casse les dérivées et peut gêner l'optimisation.

    La transformation renvoie une valeur dans (0.02, 0.98),
    ce qui évite les problèmes numériques sur les bords.
    """
    raw_arr = np.asarray(raw_value, dtype=float)
    smoothed = 0.02 + 0.96 * sigmoid((raw_arr - center) / scale)
    if raw_arr.ndim == 0:
        return float(smoothed)
    return smoothed


def inverse_smooth_unit_interval(u: float, center: float = 0.5, scale: float = 0.15) -> float:
    """
    Inverse de smooth_unit_interval.

    Pourquoi cette fonction est importante ?
    Parce qu'on veut que, à choc nul :
        LGD_i(s=0) = LGD0_i
    exactement.

    Si on appliquait directement smooth_unit_interval(LGD0_i),
    on ne retrouverait pas LGD0_i à la baseline.

    Donc :
    - on transforme d'abord LGD0_i en "latent anchor"
    - puis on ajoute les chocs en espace latent
    - puis on re-projette dans (0,1)
    """
    u = float(np.clip(u, 0.020001, 0.979999))
    z = (u - 0.02) / 0.96
    return float(center + scale * logit(z))


def build_stressed_exposures_fn(feature_cols, exposures, sector_params):
    """
    On construit PD_i(g,x) et LGD_i(g,x).

    LIEN AVEC LE PAPIER
    -------------------
    - eq. (7) / (8) : PD stressée en logit
    - eq. (10)      : LGD stressée en affine
    - Annexe A.2    : version sectorielle (b_k, d_k, c_k, e_k)

    Ici :
    - chaque exposition i hérite des sensibilités de son secteur k(i)
    - on reste donc dans une version parcimonieuse et robuste
    """

    params_idx = sector_params.set_index("sector")
    g_col      = G_COL
    x_cols     = [c for c in feature_cols if c != g_col]

    def stressed_exposures(s_vec: np.ndarray) -> pd.DataFrame:
        s = pd.Series(s_vec, index=feature_cols)

        out = exposures.copy()
        pd_vals  = []
        lgd_vals = []

        for _, row in out.iterrows():
            sec = row["sector"]
            if sec not in params_idx.index:
                raise KeyError(f"Secteur absent de sector_params.csv : {sec}")

            p = params_idx.loc[sec]

            # ------------------------------------------------
            # PD stressée
            # ------------------------------------------------
            # eq. (7) :
            #   logit(PD_i(g,x)) = logit(PD0_i) + beta_k(i)^T x + delta_k(i) g
            # ------------------------------------------------
            pd0  = float(np.clip(row["PD0"], 1e-12, 1 - 1e-12))
            z_pd = logit(pd0)

            # effet géopolitique direct
            z_pd += float(p["delta_g"]) * float(s[g_col])

            # effets macro-financiers
            for c in x_cols:
                key = f"b_{c}"
                if key in p.index and pd.notna(p[key]):
                    z_pd += float(p[key]) * float(s[c])

            pd_stress = float(np.clip(expit(z_pd), 1e-12, 1 - 1e-12))

            if ENFORCE_STRESS_ONLY_FLOOR:
                pd_stress = max(pd_stress, float(row["PD0"]))

            # ------------------------------------------------
            # LGD stressée
            # ------------------------------------------------
            # eq. (10) :
            #   LGD_i(g,x) = LGD0_i + gamma_k(i)^T x + eta_k(i) g
            #
            # Pour préserver exactement LGD_i(0)=LGD0_i, on travaille
            # en espace latent puis on re-projette dans (0,1).
            # ------------------------------------------------
            lgd0           = float(row["LGD0"])
            lgd_anchor_raw = inverse_smooth_unit_interval(lgd0)

            raw_lgd = lgd_anchor_raw + float(p["eta_g"]) * float(s[g_col])

            for c in x_cols:
                key = f"c_{c}"
                if key in p.index and pd.notna(p[key]):
                    raw_lgd += float(p[key]) * float(s[c])

            lgd_stress = smooth_unit_interval(raw_lgd)

            if ENFORCE_STRESS_ONLY_FLOOR:
                lgd_stress = max(lgd_stress, float(row["LGD0"]))

            pd_vals.append(pd_stress)
            lgd_vals.append(lgd_stress)

        out["PD_stress"]  = pd_vals
        out["LGD_stress"] = lgd_vals
        return out

    return stressed_exposures
