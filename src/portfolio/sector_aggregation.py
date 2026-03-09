from __future__ import annotations

# ============================================================
# DIAGNOSTICS SECTORIELS AU DESIGN POINT
# ============================================================
#
# Intuition :
# - le script principal optimise au niveau exposition-level ;
# - mais pour le reporting, on veut aussi une lecture sectorielle.
#
# On calcule :
# - EAD sectorielle
# - PD moyenne pondérée EAD
# - LGD moyenne pondérée EAD
# - Tail PD moyenne pondérée
# - perte de queue sectorielle
# ============================================================

import numpy as np
import pandas as pd


def build_sector_diagnostics(stressed_exposures_df: pd.DataFrame) -> pd.DataFrame:
    """
    On agrège les résultats par secteur.

    Intuition :
    - le script principal optimise au niveau exposition-level ;
    - mais pour le reporting, on veut aussi une lecture sectorielle.

    On calcule :
    - EAD sectorielle
    - PD moyenne pondérée EAD
    - LGD moyenne pondérée EAD
    - Tail PD moyenne pondérée
    - perte de queue sectorielle
    """
    df = stressed_exposures_df.copy()

    grp = df.groupby("sector", as_index=False).apply(
        lambda g: pd.Series({
            "EAD_sector"         : g["EAD"].sum(),
            "PD_sector_stress"   : np.average(g["PD_stress"], weights=g["EAD"]),
            "LGD_sector_stress"  : np.average(g["LGD_stress"], weights=g["EAD"]),
            "TailPD_sector"      : np.average(g["tail_PD"], weights=g["EAD"]),
            "LossQ_sector"       : g["loss_q_i"].sum(),
            "DeltaLossQ_sector"  : g["delta_loss_q_i"].sum(),
        })
    ).reset_index(drop=True)

    return grp
