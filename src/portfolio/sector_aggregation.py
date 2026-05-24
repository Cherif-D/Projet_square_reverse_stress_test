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


def build_sector_diagnostics(
    baseline_exposures_df: pd.DataFrame,
    stressed_exposures_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrège les résultats exposition-level en une vue sectorielle.

    Le script principal optimise au niveau exposition. Pour le reporting,
    on a aussi besoin d'une lecture sectorielle (EAD, PD, LGD, tail PD,
    perte de queue) pour alimenter la Table 2 de la note et les figures
    de l'impact sectoriel.

    Pour chaque secteur, le DataFrame retourné contient les colonnes
    suivantes (les moyennes sont pondérées par EAD) :

      - `sector` : nom du secteur.
      - `Exposure_count` : nombre d'expositions du secteur.
      - `EAD_sector` : somme des EAD.
      - `PD_sector_baseline`, `PD_sector_stress`,
        `Delta_PD_sector_bp` : PD avant / après stress et écart en bps.
      - `LGD_sector_baseline`, `LGD_sector_stress`,
        `Delta_LGD_sector_pp` : LGD avant / après stress et écart en pp.
      - `TailPD_sector_baseline`, `TailPD_sector_stress`,
        `TailPD_sector` (alias de stress), `Delta_TailPD_sector_bp` :
        tail PD ASRF avant / après stress et écart en bps.
      - `LossQ_sector_baseline`, `LossQ_sector` (= stress),
        `DeltaLossQ_sector` : perte de queue baseline, stressée, et
        contribution sectorielle à la variation totale.
      - `DeltaLossQ_share_pct` : part de chaque secteur dans le
        `DeltaLq` global (en %).

    Le DataFrame est trié par `LossQ_sector` décroissant. L'invariant
    `sum(DeltaLossQ_sector) == sum(delta_loss_q_i)` est testé dans
    `tests/test_sector_aggregation.py`.
    """
    baseline = baseline_exposures_df[[
        "id", "sector", "EAD", "PD_stress", "LGD_stress", "tail_PD", "loss_q_i",
    ]].rename(columns={
        "PD_stress": "PD_baseline",
        "LGD_stress": "LGD_baseline",
        "tail_PD": "TailPD_baseline",
        "loss_q_i": "LossQ_baseline",
    })

    stressed = stressed_exposures_df[[
        "id", "sector", "EAD", "PD_stress", "LGD_stress", "tail_PD", "loss_q_i", "delta_loss_q_i",
    ]].rename(columns={
        "PD_stress": "PD_stress",
        "LGD_stress": "LGD_stress",
        "tail_PD": "TailPD_stress",
        "loss_q_i": "LossQ_stress",
    })

    merged = baseline.merge(
        stressed,
        on=["id", "sector", "EAD"],
        how="inner",
        validate="one_to_one",
    )

    rows = []
    for sector, grp in merged.groupby("sector", sort=True):
        weights = grp["EAD"].to_numpy(dtype=float)
        pd_baseline = float(np.average(grp["PD_baseline"], weights=weights))
        pd_stress = float(np.average(grp["PD_stress"], weights=weights))
        lgd_baseline = float(np.average(grp["LGD_baseline"], weights=weights))
        lgd_stress = float(np.average(grp["LGD_stress"], weights=weights))
        tailpd_baseline = float(np.average(grp["TailPD_baseline"], weights=weights))
        tailpd_stress = float(np.average(grp["TailPD_stress"], weights=weights))
        lossq_baseline = float(grp["LossQ_baseline"].sum())
        lossq_stress = float(grp["LossQ_stress"].sum())
        delta_lossq = float(grp["delta_loss_q_i"].sum())

        rows.append({
            "sector": sector,
            "Exposure_count": int(len(grp)),
            "EAD_sector": float(grp["EAD"].sum()),
            "PD_sector_baseline": pd_baseline,
            "PD_sector_stress": pd_stress,
            "Delta_PD_sector_bp": 10000.0 * (pd_stress - pd_baseline),
            "LGD_sector_baseline": lgd_baseline,
            "LGD_sector_stress": lgd_stress,
            "Delta_LGD_sector_pp": 100.0 * (lgd_stress - lgd_baseline),
            "TailPD_sector_baseline": tailpd_baseline,
            "TailPD_sector_stress": tailpd_stress,
            "TailPD_sector": tailpd_stress,
            "Delta_TailPD_sector_bp": 10000.0 * (tailpd_stress - tailpd_baseline),
            "LossQ_sector_baseline": lossq_baseline,
            "LossQ_sector": lossq_stress,
            "DeltaLossQ_sector": delta_lossq,
        })

    sector_diag = pd.DataFrame(rows).sort_values("LossQ_sector", ascending=False).reset_index(drop=True)
    total_delta = float(sector_diag["DeltaLossQ_sector"].sum()) if not sector_diag.empty else 0.0
    if abs(total_delta) > 1e-12:
        sector_diag["DeltaLossQ_share_pct"] = 100.0 * sector_diag["DeltaLossQ_sector"] / total_delta
    else:
        sector_diag["DeltaLossQ_share_pct"] = 0.0

    return sector_diag
