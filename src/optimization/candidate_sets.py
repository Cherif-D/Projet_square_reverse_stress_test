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
# - N_phi : scénarios admissibles avec d² <= d²* + phi
# ============================================================

import numpy as np
import pandas as pd

from src.config import ETA_LOCAL, G_COL, PHI_NEAR, POOL_SIZE, RNG
from src.model.engine import ReverseStressEngine


def _sample_local_candidate(y_star: np.ndarray, eta_local: float) -> np.ndarray:
    direction = RNG.normal(0.0, 1.0, len(y_star))
    direction /= max(np.linalg.norm(direction), 1e-12)
    radius = np.sqrt(RNG.uniform(0.0, eta_local))
    return y_star + radius * direction


def _sample_near_candidate(y_star: np.ndarray, d2_cap: float) -> np.ndarray:
    y = y_star + RNG.normal(0.0, 0.45, len(y_star))
    norm2 = float(np.dot(y, y))
    if norm2 > d2_cap:
        y = y * np.sqrt(d2_cap / max(norm2, 1e-12))
    return y


def _pool_row(feature_cols, s, out, d2, local_dist2, sample_source, in_local, in_near, is_design_point):
    if is_design_point:
        set_type = "design"
    elif in_local and in_near:
        set_type = "both"
    elif in_local:
        set_type = "local"
    else:
        set_type = "near"

    return {
        "sample_source": sample_source,
        "set_type": set_type,
        "is_design_point": bool(is_design_point),
        "in_local": bool(in_local),
        "in_near": bool(in_near),
        "d2": d2,
        "local_dist2": local_dist2,
        "distance_to_design_y": float(np.sqrt(max(local_dist2, 0.0))),
        "R": out["R"],
        "break_slack": out["break_slack"],
        "Lq_abs": out["Lq_abs"],
        "delta_Lq": out["delta_Lq"],
        "RWA": out["RWA"],
        "CET1": out["CET1"],
        **{c: float(s[i]) for i, c in enumerate(feature_cols)},
    }


def generate_candidate_pool(
    best,
    feature_cols,
    L,
    Sigma_inv,
    engine: ReverseStressEngine,
    eta_local=ETA_LOCAL,
    phi_near=PHI_NEAR,
    n=POOL_SIZE,
    dedup_decimals: int = 10,
):
    """
    On construit un pool de scénarios admissibles :

    - S_eta : voisinage local autour du design point
    - N_phi : scénarios admissibles avec d² <= d²* + phi

    Cela correspond à l'esprit de la Section 5 du papier :
    ne pas s'arrêter au point unique, mais explorer un ensemble plausible.
    Les candidats restent filtrés par la contrainte de capital et par
    la contrainte géopolitique `g >= 0`.
    """
    y_star  = best["y_star"]
    d2_star = best["d2"]
    g_idx   = feature_cols.index(G_COL)
    d2_cap = d2_star + phi_near

    rows = [
        _pool_row(
            feature_cols=feature_cols,
            s=best["s_star"],
            out=best,
            d2=d2_star,
            local_dist2=0.0,
            sample_source="design_point",
            in_local=True,
            in_near=True,
            is_design_point=True,
        )
    ]

    for sample_source, sampler in (
        ("local_shell", lambda: _sample_local_candidate(y_star, eta_local)),
        ("near_projection", lambda: _sample_near_candidate(y_star, d2_cap)),
    ):
        for _ in range(n):
            y = sampler()

            s = L @ y
            d2 = float(s.T @ Sigma_inv @ s)
            local_dist2 = float(np.dot(y - y_star, y - y_star))
            in_local = local_dist2 <= eta_local + 1e-12
            in_near = d2 <= d2_cap + 1e-12

            if not (in_local or in_near):
                continue

            out = engine.evaluate(s, include_exposures=False)

            if s[g_idx] < 0:
                continue
            if out["break_slack"] < -1e-8:
                continue

            rows.append(
                _pool_row(
                    feature_cols=feature_cols,
                    s=s,
                    out=out,
                    d2=d2,
                    local_dist2=local_dist2,
                    sample_source=sample_source,
                    in_local=in_local,
                    in_near=in_near,
                    is_design_point=False,
                )
            )

    pool = pd.DataFrame(rows)
    if pool.empty:
        return pool

    rounded = pool[feature_cols].round(dedup_decimals)
    pool = (
        pool.assign(_dedup_key=rounded.astype(str).agg("|".join, axis=1))
        .sort_values(["is_design_point", "d2", "break_slack", "local_dist2"], ascending=[False, True, True, True])
        .drop_duplicates(subset="_dedup_key", keep="first")
        .drop(columns="_dedup_key")
        .sort_values(["d2", "break_slack", "local_dist2"])
        .reset_index(drop=True)
    )
    return pool
