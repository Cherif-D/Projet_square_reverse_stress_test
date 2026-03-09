from __future__ import annotations

# ============================================================
# SAUVEGARDE DES RAPPORTS JSON / MARKDOWN
# ============================================================
#
# Ce module centralise toutes les écritures de rapports
# dans outputs/reports/.
# ============================================================

import json

import numpy as np
import pandas as pd

from src.paths import REPORTS


def save_summary(
    feature_cols: list[str],
    baseline_out: dict,
    capital,
    best: dict,
    all_optima: list,
    pool: pd.DataFrame,
    shortlist: pd.DataFrame,
    design_point: pd.Series,
    eta_local: float,
    phi_near: float,
) -> None:
    """
    Sauvegarde summary.json et summary.md.
    """
    top_drivers = design_point.abs().sort_values(ascending=False).head(8).to_dict()

    summary = {
        "feature_cols"            : feature_cols,
        "baseline_ratio_model"    : float(baseline_out["R"]),
        "baseline_Lq_abs"         : float(baseline_out["Lq_abs"]),
        "baseline_delta_Lq"       : float(baseline_out["delta_Lq"]),
        "baseline_RWA_model"      : float(baseline_out["RWA"]),
        "baseline_CET1_model"     : float(baseline_out["CET1"]),
        "baseline_Lq_reference"   : float(baseline_out["Lq_baseline"]),
        "R_omega"                 : float(capital["R_omega"]),
        "R0_input"                : float(capital["R0"]),
        "q"                       : float(capital["q"]),
        "design_point_ratio"      : float(best["R"]),
        "design_point_Lq_abs"     : float(best["Lq_abs"]),
        "design_point_delta_Lq"   : float(best["delta_Lq"]),
        "design_point_RWA"        : float(best["RWA"]),
        "design_point_CET1"       : float(best["CET1"]),
        "design_point_d2"         : float(best["d2"]),
        "design_point_pvalue"     : float(best["pval"]),
        "n_local_optima_found"    : int(len(all_optima)),
        "n_candidate_pool"        : int(len(pool)),
        "n_shortlist"             : int(len(shortlist)),
        "top_drivers_abs"         : {k: float(v) for k, v in top_drivers.items()},
        "eta_local"               : eta_local,
        "phi_near"                : phi_near,
        "baseline_check_R_equals_R0": float(baseline_out["R"]),
    }

    with open(REPORTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(REPORTS / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Reverse Stress Test — script principal final\n\n")
        f.write("## Baseline\n")
        f.write(f"- Ratio baseline modèle : {baseline_out['R']:.6f}\n")
        f.write(f"- Ratio initial input R0 : {float(capital['R0']):.6f}\n")
        f.write(f"- Lq absolu baseline : {baseline_out['Lq_abs']:.6f}\n")
        f.write(f"- DeltaLq baseline : {baseline_out['delta_Lq']:.6f}\n")
        f.write(f"- RWA baseline : {baseline_out['RWA']:.6f}\n")
        f.write(f"- CET1 baseline : {baseline_out['CET1']:.6f}\n")
        f.write(f"- Lq référence (s=0) : {baseline_out['Lq_baseline']:.6f}\n\n")

        f.write("## Design point\n")
        f.write(f"- Seuil de rupture R_omega : {float(capital['R_omega']):.6f}\n")
        f.write(f"- Ratio design point : {best['R']:.6f}\n")
        f.write(f"- Lq absolu design point : {best['Lq_abs']:.6f}\n")
        f.write(f"- DeltaLq design point : {best['delta_Lq']:.6f}\n")
        f.write(f"- RWA design point : {best['RWA']:.6f}\n")
        f.write(f"- CET1 design point : {best['CET1']:.6f}\n")
        f.write(f"- d² Mahalanobis : {best['d2']:.6f}\n")
        f.write(f"- p-value plausibilité : {best['pval']:.8f}\n\n")

        f.write("## Ensembles de scénarios\n")
        f.write(f"- Nombre d'optima faisables trouvés : {len(all_optima)}\n")
        f.write(f"- Taille du pool candidat : {len(pool)}\n")
        f.write(f"- Taille de la shortlist : {len(shortlist)}\n\n")

        f.write("## Top drivers du design point\n")
        for k, v in design_point.abs().sort_values(ascending=False).head(8).items():
            f.write(f"- {k}: {float(v):.4f}\n")
