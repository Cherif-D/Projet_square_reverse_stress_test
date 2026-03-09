from __future__ import annotations

# ============================================================
# ETAPE 5 — S_eta ET N_phi
# ============================================================
#
# LIEN AVEC LE PAPIER
# -------------------
# - eq. (38) : near-optimal set N_phi
# - Section 5 : explorer un ensemble plausible, pas seulement le point unique
#
# On construit un pool de scénarios admissibles :
# - S_eta : voisinage local autour du design point
# - N_phi : near-optimal set avec d² <= d²* + phi
# ============================================================

import numpy as np
import pandas as pd

from src.config import ETA_LOCAL, G_COL, PHI_NEAR, POOL_SIZE, RNG


def generate_candidate_pool(best, feature_cols, L, Sigma_inv, capital, loss_rwa_ratio_fn,
                            eta_local=ETA_LOCAL, phi_near=PHI_NEAR, n=POOL_SIZE):
    """
    On construit un pool de scénarios admissibles :

    - S_eta : voisinage local autour du design point
    - N_phi : near-optimal set avec d² <= d²* + phi

    Cela correspond à l'esprit de la Section 5 du papier :
    ne pas s'arrêter au point unique, mais explorer un ensemble plausible.
    """
    y_star  = best["y_star"]
    d2_star = best["d2"]
    R_omega = float(capital["R_omega"])
    g_idx   = feature_cols.index(G_COL)

    rows = []

    for _ in range(n):
        if RNG.uniform() < 0.5:
            y = y_star + RNG.normal(0.0, 0.35, len(feature_cols))
        else:
            y = RNG.normal(0.0, 1.0, len(feature_cols))
            y[g_idx] = abs(y[g_idx]) + 0.1

        s          = L @ y
        d2         = float(s.T @ Sigma_inv @ s)
        local_dist2 = float(np.dot(y - y_star, y - y_star))

        out = loss_rwa_ratio_fn(s)

        if s[g_idx] < 0:
            continue
        if out["R"] > R_omega:
            continue

        in_local = local_dist2 <= eta_local
        in_near  = d2 <= d2_star + phi_near

        if in_local or in_near:
            rows.append({
                "set_type"  : "local" if in_local else "near",
                "d2"        : d2,
                "local_dist2": local_dist2,
                "R"         : out["R"],
                "Lq_abs"    : out["Lq_abs"],
                "delta_Lq"  : out["delta_Lq"],
                "RWA"       : out["RWA"],
                "CET1"      : out["CET1"],
                **{c: float(s[i]) for i, c in enumerate(feature_cols)},
            })

    pool = pd.DataFrame(rows)
    return pool
