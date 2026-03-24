from __future__ import annotations

# ============================================================
# ETAPE 6 — SHORTLIST MAXIMIN
# ============================================================
#
# LIEN AVEC LE PAPIER
# -------------------
# - eq. (49) : réduction maximin / farthest-point
#
# On réduit le pool à une shortlist non redondante.
# On travaille en y-space car les distances y sont directement
# cohérentes avec la géométrie de Mahalanobis.
# ============================================================

import numpy as np
import pandas as pd

from src.config import SHORTLIST_SIZE


def farthest_point_selection(Y: np.ndarray, P: int, start_idx: int = 0) -> list[int]:
    """
    Réduction maximin / farthest-point.

    Logique :
    - on part du scénario le plus plausible
    - puis on ajoute itérativement le point le plus éloigné
      des scénarios déjà sélectionnés en y-space

    Cela correspond à l'esprit de l'eq. (49) du papier :
    produire une petite shortlist non redondante.
    """
    selected = [start_idx]
    dmin     = np.linalg.norm(Y - Y[start_idx], axis=1)

    for _ in range(1, min(P, len(Y))):
        j = int(np.argmax(dmin))
        selected.append(j)
        dnew = np.linalg.norm(Y - Y[j], axis=1)
        dmin = np.minimum(dmin, dnew)

    return selected


def build_shortlist(pool: pd.DataFrame, feature_cols: list[str], L: np.ndarray, P: int = SHORTLIST_SIZE):
    """
    On réduit le pool à une shortlist non redondante.

    On travaille en y-space car les distances y sont directement
    cohérentes avec la géométrie de Mahalanobis.
    """
    if pool.empty:
        return pool.copy()

    Y = np.linalg.solve(L, pool[feature_cols].values.T).T
    if "is_design_point" in pool.columns and pool["is_design_point"].any():
        start_idx = int(np.flatnonzero(pool["is_design_point"].to_numpy())[0])
    else:
        start_idx = int(np.argmin(pool["d2"].values))
    idx       = farthest_point_selection(Y, P=min(P, len(pool)), start_idx=start_idx)
    shortlist = pool.iloc[idx].copy()
    return shortlist
