from __future__ import annotations

# ============================================================
# SAUVEGARDE DES RAPPORTS JSON / MARKDOWN
# ============================================================
#
# Ce module centralise toutes les écritures de rapports
# dans outputs/reports/.
# ============================================================

import json

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
    sector_diag: pd.DataFrame,
    eta_local: float,
    phi_near: float,
) -> None:
    """
    Sauvegarde summary.json et summary.md.
    """
    top_drivers = design_point.abs().sort_values(ascending=False).head(8).to_dict()

    baseline_gap = float(baseline_out["R"] - float(capital["R0"]))
    sector_delta_lq_sum = float(sector_diag["DeltaLossQ_sector"].sum()) if not sector_diag.empty else 0.0
    sector_reconciliation_error = float(sector_delta_lq_sum - best["delta_Lq"])

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
        "design_point_break_slack": float(best["break_slack"]),
        "design_point_constraint_saturated": bool(best.get("constraint_saturated", False)),
        "design_point_d2"         : float(best["d2"]),
        "design_point_pvalue"     : float(best["pval"]),
        "n_local_optima_found"    : int(len(all_optima)),
        "n_candidate_pool"        : int(len(pool)),
        "n_candidate_pool_local"  : int(pool["in_local"].sum()) if "in_local" in pool.columns else 0,
        "n_candidate_pool_near"   : int(pool["in_near"].sum()) if "in_near" in pool.columns else 0,
        "n_shortlist"             : int(len(shortlist)),
        "top_drivers_abs"         : {k: float(v) for k, v in top_drivers.items()},
        "eta_local"               : eta_local,
        "phi_near"                : phi_near,
        "baseline_check_R_equals_R0": float(baseline_out["R"]),
        "baseline_check_R_equals_R0_gap": baseline_gap,
        "sector_delta_Lq_sum": sector_delta_lq_sum,
        "sector_delta_Lq_reconciliation_error": sector_reconciliation_error,
    }

    with open(REPORTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(REPORTS / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Reverse Stress Test — synthèse du run\n\n")
        f.write("## Baseline\n")
        f.write(f"- Ratio baseline modèle : {baseline_out['R']:.6f}\n")
        f.write(f"- Ratio initial input R0 : {float(capital['R0']):.6f}\n")
        f.write(f"- Lq absolu baseline : {baseline_out['Lq_abs']:.6f}\n")
        f.write(f"- DeltaLq baseline : {baseline_out['delta_Lq']:.6f}\n")
        f.write(f"- RWA baseline : {baseline_out['RWA']:.6f}\n")
        f.write(f"- CET1 baseline : {baseline_out['CET1']:.6f}\n")
        f.write(f"- Lq référence (s=0) : {baseline_out['Lq_baseline']:.6f}\n\n")
        f.write(f"- Ecart R(0) - R0 : {baseline_gap:.6e}\n\n")

        f.write("## Design point\n")
        f.write(f"- Seuil de rupture R_omega : {float(capital['R_omega']):.6f}\n")
        f.write(f"- Ratio design point : {best['R']:.6f}\n")
        f.write(f"- Slack de rupture : {best['break_slack']:.6e}\n")
        f.write(f"- Lq absolu design point : {best['Lq_abs']:.6f}\n")
        f.write(f"- DeltaLq design point : {best['delta_Lq']:.6f}\n")
        f.write(f"- RWA design point : {best['RWA']:.6f}\n")
        f.write(f"- CET1 design point : {best['CET1']:.6f}\n")
        f.write(f"- d² Mahalanobis : {best['d2']:.6f}\n")
        f.write(f"- p-value plausibilité : {best['pval']:.8f}\n\n")

        f.write("## Ensembles de scénarios\n")
        f.write(f"- Nombre d'optima faisables trouvés : {len(all_optima)}\n")
        f.write(f"- Taille du pool candidat : {len(pool)}\n")
        if "in_local" in pool.columns and "in_near" in pool.columns:
            f.write(f"- Scénarios dans S_eta : {int(pool['in_local'].sum())}\n")
            f.write(f"- Scénarios dans N_phi : {int(pool['in_near'].sum())}\n")
        f.write(f"- Taille de la shortlist : {len(shortlist)}\n\n")

        f.write("## Reconciliation sectorielle\n")
        f.write(f"- Somme sectorielle DeltaLq : {sector_delta_lq_sum:.6f}\n")
        f.write(f"- Ecart vs DeltaLq global : {sector_reconciliation_error:.6e}\n\n")

        f.write("## Top drivers du design point\n")
        for k, v in design_point.abs().sort_values(ascending=False).head(8).items():
            f.write(f"- {k}: {float(v):.4f}\n")


def save_calibration_governance_note(sector_params: pd.DataFrame) -> None:
    """Document the empirical calibration gap and the required backtesting workflow."""
    origins = sorted(sector_params.get("param_origin", pd.Series(dtype=str)).dropna().unique().tolist())
    statuses = sorted(
        sector_params.get("calibration_status", pd.Series(dtype=str)).dropna().unique().tolist()
    )
    is_stylized = bool(
        "param_origin" in sector_params.columns
        and sector_params["param_origin"].astype(str).str.contains("stylized", case=False).any()
    )

    note_lines = [
        "# Calibration, Backtesting and Sensitivity Note",
        "",
        "## Current status",
        f"- Parameter origins observed in sector_params.csv: {origins or ['not documented']}",
        f"- Calibration statuses observed in sector_params.csv: {statuses or ['not documented']}",
        f"- Current implementation still uses stylized coefficients: {'yes' if is_stylized else 'no'}",
        "",
        "## Empirical calibration target",
        "1. Estimate sector-level PD elasticities with a panel logit or quasi-binomial model on internal defaults.",
        "2. Estimate LGD elasticities on workout / recovery data with bounded-response regression in latent space.",
        "3. Re-estimate the model on rolling windows and retain only signs and magnitudes that remain stable out-of-sample.",
        "",
        "## Minimum backtesting package",
        "1. One-year ahead PD ranking accuracy by sector and by risk bucket.",
        "2. Directional backtest: adverse GDP, VIX, unemployment and GPR shocks must worsen sector PD/LGD in expected directions.",
        "3. Stability backtest: coefficients must remain inside challenger bands across rolling windows.",
        "4. Portfolio reconciliation: stressed sector losses must aggregate exactly to total DeltaLq.",
        "",
        "## Sensitivity package for governance",
        "1. Global scale on all sector coefficients: 0.75x / 1.00x / 1.25x.",
        "2. Geopolitical-only scale on delta_g and eta_g: 0.75x / 1.00x / 1.25x.",
        "3. Alternative covariance/reference law: Gaussian versus Student-t.",
        "4. Capital sensitivity: q in {99.5%, 99.9%} and RWA slope multipliers +/-10%.",
        "",
        "## Governance rule",
        "Until empirical calibration is available, outputs should be labelled prototype / academic and not model-approved.",
        "",
    ]

    with open(REPORTS / "calibration_backtesting_note.md", "w", encoding="utf-8") as f:
        f.write("\n".join(note_lines))
