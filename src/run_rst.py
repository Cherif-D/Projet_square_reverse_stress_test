from __future__ import annotations

# ============================================================
# ORCHESTRATION DU PIPELINE RST
# ============================================================
#
# Ce module est l'équivalent de main() dans MVP.py.
# Il assemble toutes les étapes du reverse stress test dans l'ordre.
#
# LOGIQUE METHODOLOGIQUE
# ----------------------
# On suit la chaîne du papier :
#
#   s = (g, x)
#     -> PD_i(g,x), LGD_i(g,x)
#     -> L_q(g,x)
#     -> CET1(g,x), RWA(g,x)
#     -> R(s)
#     -> design point s_omega
#     -> ensembles plausibles + shortlist gouvernance-ready
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 7 janvier 2026)
# ------------------------------------------------------------
# - eq. (5)  : scénario s = (g, x)^T
# - eq. (7)  : logit(PD_i(g,x)) = logit(PD0_i) + beta_k(i)^T x + delta_k(i) g
# - eq. (10) : LGD_i(g,x) via transmission affine en espace latent
#              (smooth projection sigmoïde pour garantir LGD ∈ (0,1) et
#               LGD_i(0) = LGD0_i exactement ; approche l'affine du papier
#               pour les petits chocs)
# - eq. (16) : L_q(g,x) ≈ sum_i EAD_i * LGD_i(g,x) * tail_PD_i(g,x)
# - eq. (17) : CET1(g,x) = CET1_0 - pertes + delta_non_credit
# - eq. (21) : RWA(g,x) = RWA_0 + sum_i alpha_i * (PD_i(g,x) - PD0_i)
# - eq. (22) : R(s) = CET1(g,x) / RWA(g,x)
# - eq. (24) : design point = scénario cassant le plus plausible
# - eq. (38) : near-optimal set N_phi
# - eq. (49) : réduction maximin / farthest-point
# ============================================================

import numpy as np
import pandas as pd

from src.config import ETA_LOCAL, PHI_NEAR, POOL_SIZE, SHORTLIST_SIZE
from src.model.baseline import inspect_tutor_data_context
from src.model.loss_model import build_loss_rwa_ratio_fn
from src.model.stress_mappings import build_stressed_exposures_fn
from src.optimization.candidate_sets import generate_candidate_pool
from src.optimization.design_point import solve_design_point
from src.optimization.shortlist import build_shortlist
from src.paths import OUTPUTS
from src.portfolio.loaders import load_inputs
from src.portfolio.sector_aggregation import build_sector_diagnostics
from src.reporting.save_reports import save_summary
from src.reporting.save_tables import (
    save_baseline_exposure_metrics,
    save_candidate_pool,
    save_design_point,
    save_exposure_stress_at_design_point,
    save_sector_diagnostics,
    save_shortlist,
)
from src.scenario.build_scenario import build_scenario_reference


def main():
    # --------------------------------------------------------
    # Contexte données tuteur
    # --------------------------------------------------------
    inspect_tutor_data_context()

    # --------------------------------------------------------
    # Étape 1 : scénario / Sigma
    # --------------------------------------------------------
    scenario_std, feature_cols, Sigma, Sigma_inv, L = build_scenario_reference()

    # --------------------------------------------------------
    # Étape 2 : inputs crédit / capital
    # --------------------------------------------------------
    exposures, capital, sector_params = load_inputs()

    # --------------------------------------------------------
    # Étape 3 : fonctions PD/LGD
    # --------------------------------------------------------
    stressed_exposures_fn = build_stressed_exposures_fn(feature_cols, exposures, sector_params)

    # --------------------------------------------------------
    # Étape 4 : fonctions Lq / CET1 / RWA / ratio
    # --------------------------------------------------------
    loss_rwa_ratio_fn = build_loss_rwa_ratio_fn(exposures, capital, stressed_exposures_fn)

    # --------------------------------------------------------
    # Baseline s = 0
    # --------------------------------------------------------
    s_zero       = np.zeros(len(feature_cols))
    baseline_out = loss_rwa_ratio_fn(s_zero)

    baseline_stressed = baseline_out["stressed"].copy()
    save_baseline_exposure_metrics(baseline_stressed)

    # --------------------------------------------------------
    # Étape 5 : design point
    # --------------------------------------------------------
    best, all_optima = solve_design_point(feature_cols, Sigma_inv, L, capital, loss_rwa_ratio_fn)

    design_point = pd.Series(best["s_star"], index=feature_cols)
    save_design_point(best["s_star"], feature_cols)
    save_exposure_stress_at_design_point(best["stressed"])

    sector_diag = build_sector_diagnostics(best["stressed"])
    save_sector_diagnostics(sector_diag)

    # --------------------------------------------------------
    # Étape 6 : S_eta / N_phi
    # --------------------------------------------------------
    pool = generate_candidate_pool(
        best            = best,
        feature_cols    = feature_cols,
        L               = L,
        Sigma_inv       = Sigma_inv,
        capital         = capital,
        loss_rwa_ratio_fn = loss_rwa_ratio_fn,
        eta_local       = ETA_LOCAL,
        phi_near        = PHI_NEAR,
        n               = POOL_SIZE,
    )
    save_candidate_pool(pool)

    # --------------------------------------------------------
    # Étape 7 : shortlist
    # --------------------------------------------------------
    shortlist = build_shortlist(pool, feature_cols, L, P=SHORTLIST_SIZE)
    save_shortlist(shortlist)

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------
    save_summary(
        feature_cols  = feature_cols,
        baseline_out  = baseline_out,
        capital       = capital,
        best          = best,
        all_optima    = all_optima,
        pool          = pool,
        shortlist     = shortlist,
        design_point  = design_point,
        eta_local     = ETA_LOCAL,
        phi_near      = PHI_NEAR,
    )

    print("OK - pipeline RST terminé.")
    print("Sorties disponibles dans :", OUTPUTS)
    print(f"Baseline check : R(0) = {baseline_out['R']:.6f} vs R0 input = {float(capital['R0']):.6f}")


if __name__ == "__main__":
    main()
