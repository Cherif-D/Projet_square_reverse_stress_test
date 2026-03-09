from __future__ import annotations

# ============================================================
# MODELE DE CAPITAL — CET1, RWA, R(s)
# ============================================================
#
# LIEN AVEC LE PAPIER
# -------------------
# - eq. (17) : CET1(g,x) = CET1_0 - pertes + delta_non_credit
# - eq. (21) : RWA(g,x)  = RWA_0 + sum_i alpha_i * (PD_i(g,x) - PD0_i)
# - eq. (22) : R(s)       = CET1(g,x) / RWA(g,x)
#
# CORRECTION DE BASELINE
# ----------------------
# On travaille avec delta_Lq(s) = Lq_abs(s) - Lq_baseline
# de façon à garantir R(0) = R0.
# ============================================================

import numpy as np


def compute_cet1(CET1_0: float, delta_Lq: float, delta_non_credit: float) -> float:
    """
    CET1(s) corrigé baseline — eq. (17).

    CET1(s) = CET1_0 - delta_Lq(s) + delta_non_credit
    """
    return float(CET1_0 - delta_Lq + delta_non_credit)


def compute_rwa(RWA_0: float, alpha_rwa: np.ndarray, PD_stress: np.ndarray, PD0_base: np.ndarray) -> float:
    """
    RWA(s) — eq. (21).

    RWA(s) = RWA_0 + sum_i alpha_i * (PD_i(s) - PD0_i)
    """
    RWA = float(RWA_0 + np.sum(alpha_rwa * (PD_stress - PD0_base)))
    return max(RWA, 1e-12)


def compute_ratio(CET1: float, RWA: float) -> float:
    """
    Ratio de capital — eq. (22).

    R(s) = CET1(s) / RWA(s)
    """
    return CET1 / RWA
