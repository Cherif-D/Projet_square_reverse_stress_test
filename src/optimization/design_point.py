from __future__ import annotations

# ============================================================
# ETAPE 4 — DESIGN POINT EN ESPACE BLANCHI
# ============================================================
#
# LIEN AVEC LE PAPIER
# -------------------
# - eq. (24) : design point = scénario le plus plausible conduisant à R(s) <= R_omega
#
# Sous hypothèse gaussienne :
#     maximiser la plausibilité
# revient à
#     minimiser la distance de Mahalanobis
#
# En espace blanchi :
#     s = L y
# la distance devient :
#     d²(s) = ||y||²
#
# On résout donc :
#     min_y  1/2 ||y||²
#     s.c.   R(Ly) <= R_omega
#            g(Ly) >= 0
#            (relaxation numérique de la contrainte théorique g > 0)
# ============================================================

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from src.config import G_COL, RNG
from src.model.engine import ReverseStressEngine


def build_starting_points(feature_cols, L):
    """
    On construit un multi-start robuste :
    - random starts en y-space
    - ladder sur l'intensité géopolitique

    Pourquoi ?
    Parce que l'optimisation inverse peut être sensible au point de départ.
    """
    g_idx  = feature_cols.index(G_COL)
    starts = []

    # starts aléatoires
    for _ in range(20):
        y0 = RNG.normal(0.0, 0.5, len(feature_cols))
        y0[g_idx] = abs(y0[g_idx]) + 0.25
        starts.append(y0)

    # ladder sur g
    for level in [0.25, 0.50, 1.00, 1.50, 2.00, 3.00]:
        s0 = np.zeros(len(feature_cols))
        s0[g_idx] = level
        y0 = np.linalg.solve(L, s0)
        starts.append(y0)

    return starts


def solve_design_point(feature_cols, Sigma_inv, L, engine: ReverseStressEngine):
    """
    On résout le design point.

    Sous hypothèse gaussienne :
        maximiser la plausibilité
    revient à
        minimiser la distance de Mahalanobis

    En espace blanchi :
        s = L y
    la distance devient :
        d²(s) = ||y||²

    On résout donc :
        min_y  1/2 ||y||²
        s.c.   R(Ly) <= R_omega
               g(Ly) >= 0

        La contrainte `g > 0` du papier est écrite `g >= 0` pour SLSQP.
        Le design point courant est strictement positif en pratique.
    """
    g_idx   = feature_cols.index(G_COL)
    R_omega = float(engine.R_omega)

    def s_from_y(y):
        return L @ y

    def objective_y(y):
        return 0.5 * float(np.dot(y, y))

    def cst_break(y):
        s   = s_from_y(y)
        out = engine.evaluate(s, include_exposures=False)
        return out["break_slack"]

    def cst_g_positive(y):
        s = s_from_y(y)
        return s[g_idx]

    constraints = [
        {"type": "ineq", "fun": cst_break},
        {"type": "ineq", "fun": cst_g_positive},
    ]

    starts = build_starting_points(feature_cols, L)
    sols   = []

    for y0 in starts:
        res = minimize(
            objective_y,
            y0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-9, "disp": False},
        )

        if not res.success:
            continue

        y_star = res.x
        s_star = s_from_y(y_star)
        out = engine.evaluate(s_star, include_exposures=True)

        if out["R"] <= R_omega + 1e-6 and s_star[g_idx] >= -1e-8:
            d2   = float(s_star.T @ Sigma_inv @ s_star)
            pval = float(1.0 - chi2.cdf(d2, df=len(feature_cols)))

            sols.append({
                "y_star"     : y_star,
                "s_star"     : s_star,
                "R"          : out["R"],
                "Lq_abs"     : out["Lq_abs"],
                "delta_Lq"   : out["delta_Lq"],
                "RWA"        : out["RWA"],
                "CET1"       : out["CET1"],
                "Lq_baseline": out["Lq_baseline"],
                "break_slack": out["break_slack"],
                "constraint_saturated": bool(abs(out["break_slack"]) <= 1e-6),
                "d2"         : d2,
                "pval"       : pval,
                "stressed"   : out["stressed"],
                "success"    : True,
                "nit"        : getattr(res, "nit", None),
            })

    if not sols:
        raise RuntimeError("Aucune solution faisable trouvée. Vérifie les inputs ou la sévérité des paramètres.")

    best = min(sols, key=lambda z: (z["d2"], abs(z["break_slack"])))
    return best, sols
