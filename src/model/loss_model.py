from __future__ import annotations

# ============================================================
# ETAPE 3 — L_q, CET1, RWA, R(s)
# ============================================================
#
# LIEN AVEC LE PAPIER
# -------------------
# - eq. (16) : L_q(g,x)
# - eq. (17) : CET1(g,x)
# - eq. (21) : RWA(g,x)
# - eq. (22) : R(s)
#
# CORRECTION DE BASELINE
# ----------------------
# On travaille ici avec :
#     delta_Lq(s) = Lq_abs(s) - Lq_baseline
#
# de façon à garantir :
#     R(0) = R0
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.model.capital_model import compute_cet1, compute_ratio, compute_rwa


def tail_default_prob(pd_vec: np.ndarray, rho_vec: np.ndarray, q: float) -> np.ndarray:
    """
    Probabilité de défaut de queue dans l'approximation ASRF / Vasicek.

    C'est le terme :
        Phi((Phi^{-1}(PD) + sqrt(rho) * Phi^{-1}(q)) / sqrt(1-rho))

    Il apparaît dans :
    - eq. (16) du papier pour L_q(g,x)
    - eq. (59) de l'annexe sectorielle
    """
    zq  = norm.ppf(q)
    arg = (norm.ppf(pd_vec) + np.sqrt(rho_vec) * zq) / np.sqrt(1.0 - rho_vec)
    return norm.cdf(arg)


def build_loss_rwa_ratio_fn(exposures, capital, stressed_exposures_fn):
    """
    On construit les fonctions :
    - Lq_abs(s)
    - delta_Lq(s)
    - CET1(s)
    - RWA(s)
    - R(s)

    LIEN AVEC LE PAPIER
    -------------------
    - eq. (16) : L_q(g,x)
    - eq. (17) : CET1(g,x)
    - eq. (21) : RWA(g,x)
    - eq. (22) : R(s)

    CORRECTION DE BASELINE
    ----------------------
    On travaille ici avec :
        delta_Lq(s) = Lq_abs(s) - Lq_baseline

    de façon à garantir :
        R(0) = R0
    """

    q                = float(capital["q"])
    CET1_0           = float(capital["CET1_0"])
    RWA_0            = float(capital["RWA_0"])
    delta_non_credit = float(capital["delta_non_credit"])

    PD0_base  = exposures["PD0"].to_numpy(dtype=float)
    alpha_rwa = exposures["alpha_rwa"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Baseline s = 0
    # --------------------------------------------------------
    s_zero = np.zeros(1)  # placeholder overwritten below in caller? no, we rebuild with len via closure outside? we need len(feature cols) unavailable
    # On ne peut pas utiliser s_zero ici sans connaître la dimension du scénario.
    # Donc on calcule Lq_baseline dans une fonction dédiée à partir des inputs baseline.

    baseline_tail_pd = tail_default_prob(
        exposures["PD0"].to_numpy(dtype=float),
        exposures["rho"].to_numpy(dtype=float),
        q,
    )

    Lq_baseline = float(
        np.sum(
            exposures["EAD"].to_numpy(dtype=float)
            * exposures["LGD0"].to_numpy(dtype=float)
            * baseline_tail_pd
        )
    )

    def loss_rwa_ratio(s_vec: np.ndarray):
        stressed = stressed_exposures_fn(s_vec)

        EAD = stressed["EAD"].to_numpy(dtype=float)
        PD  = stressed["PD_stress"].to_numpy(dtype=float)
        LGD = stressed["LGD_stress"].to_numpy(dtype=float)
        rho = stressed["rho"].to_numpy(dtype=float)

        # ----------------------------------------------------
        # L_q(s) absolu — eq. (16)
        # ----------------------------------------------------
        tail_pd = tail_default_prob(PD, rho, q)
        stressed["tail_PD"] = tail_pd

        stressed["loss_q_i"] = EAD * LGD * tail_pd
        Lq_abs = float(stressed["loss_q_i"].sum())

        # ----------------------------------------------------
        # Correction baseline : delta_Lq(s)
        # ----------------------------------------------------
        delta_Lq = float(Lq_abs - Lq_baseline)

        # ----------------------------------------------------
        # CET1(s) corrigé baseline
        # ----------------------------------------------------
        CET1 = compute_cet1(CET1_0, delta_Lq, delta_non_credit)

        # ----------------------------------------------------
        # RWA(s) — eq. (21)
        # ----------------------------------------------------
        RWA = compute_rwa(RWA_0, alpha_rwa, PD, PD0_base)

        # ----------------------------------------------------
        # Ratio — eq. (22)
        # ----------------------------------------------------
        R = compute_ratio(CET1, RWA)

        # Diagnostics enrichis
        stressed["delta_loss_q_i"] = stressed["loss_q_i"] - (
            stressed["EAD"].to_numpy(dtype=float)
            * exposures["LGD0"].to_numpy(dtype=float)
            * baseline_tail_pd
        )

        return {
            "R"           : R,
            "Lq_abs"      : Lq_abs,
            "delta_Lq"    : delta_Lq,
            "RWA"         : RWA,
            "CET1"        : CET1,
            "Lq_baseline" : Lq_baseline,
            "stressed"    : stressed,
        }

    return loss_rwa_ratio
