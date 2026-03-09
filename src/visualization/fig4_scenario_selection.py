from __future__ import annotations

# ============================================================
# FIGURE 4 — PROCEDURE DE SELECTION DE SCENARIOS FINIS
# ============================================================
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 2026)
# ---------------------------------------------------
# Figure 4 (p. 23-24) : "Illustration of the finite-scenario
#   selection procedure in the (g, x) plane."
#
# Ce graphique illustre la réduction en 3 étapes du pool infini
# vers une liste finie de scénarios gouvernance-ready :
#
# - Courbe rouge : frontière de rupture R(g, x) = R̄
# - Zone hachurée : ensemble admissible E (S_rho ou N_epsilon)
# - Points gris : pool de candidats C_N (multi-start + exploration locale)
# - Marqueurs verts : ancres géopolitiques {(g_j, x*(g_j))}
#   sur une grille d'intensités géopolitiques (mild/moderate/severe)
# - Marqueurs noirs : liste réduite C_P via règle farthest-point
#   (P = SHORTLIST_SIZE scénarios non redondants)
#
# Règle farthest-point (eq. 49) :
#   s^[p] = argmax_{s in C_N} min_{p'<p} ||L^{-1}(s - s^[p'])||_2
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import G_COL
from src.paths import OUTPUTS, TABLES
from src.visualization.helpers import (
    compute_Z_ratio_and_dist,
    compute_ratio_grid,
    load_viz_data,
    make_grid,
)

PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

# Style global
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def plot_fig4_scenario_selection(n_pts: int = 70, span: float = 3.5) -> None:
    """
    Reproduit la Figure 4 du papier :
    Procédure de sélection de scénarios finis dans le plan (g, x).

    Affiche :
    - Frontière de rupture (rouge épaisse)
    - Ensemble admissible N_epsilon (zone hachurée jaune)
    - Pool de candidats (points gris semi-transparents)
    - Ancres géopolitiques (pentagones verts)
    - Shortlist farthest-point (losanges noirs avec étiquettes)
    - Design point s* (étoile rouge)
    """
    exposures, capital, sector_params, design_point, sigma, summary = load_viz_data()

    pool_path      = TABLES / "candidate_pool.csv"
    shortlist_path = TABLES / "scenario_shortlist.csv"
    if not pool_path.exists() or not shortlist_path.exists():
        print("[Fig 4] candidate_pool.csv ou scenario_shortlist.csv introuvable. "
              "Lancez d'abord run_project.py.")
        return

    pool      = pd.read_csv(pool_path)
    shortlist = pd.read_csv(shortlist_path)

    macro_drivers = design_point.drop(G_COL).abs().sort_values(ascending=False)
    x_driver = G_COL
    y_driver = macro_drivers.index[0]
    print(f"[Fig 4] Axes : X={x_driver} / Y={y_driver}")

    inv_sigma = np.linalg.inv(sigma.values)
    L         = np.linalg.cholesky(sigma.values)

    xx, yy, (X_grid, Y_grid) = make_grid(
        design_point, x_driver, y_driver, span=span, n_pts=n_pts
    )

    Z_ratio, Z_dist = compute_Z_ratio_and_dist(
        X_grid, Y_grid, design_point, x_driver, y_driver,
        exposures, capital, sector_params, inv_sigma
    )

    threshold = float(capital["R_omega"])
    d2_star   = float(summary["design_point_d2"])
    gx        = float(design_point[x_driver])
    gy        = float(design_point[y_driver])
    x_rng     = X_grid.max() - X_grid.min()
    y_rng     = Y_grid.max() - Y_grid.min()

    # --------------------------------------------------------
    # Ancres géopolitiques sur une grille d'intensités
    # On balaye g de 0.02 à gx * 2, autres coords = design point
    # --------------------------------------------------------
    g_levels   = np.linspace(0.02, gx * 2.5, 8)
    anchor_pts = []
    for g_val in g_levels:
        s_anchor         = design_point.copy()
        s_anchor[x_driver] = g_val
        r_val = compute_ratio_grid(dict(s_anchor), exposures, capital, sector_params)
        if r_val <= threshold + 0.005:
            anchor_pts.append((g_val, float(s_anchor[y_driver])))

    # --------------------------------------------------------
    # Tracé
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("white")

    # 1. Zone de rupture S_red (fond très clair)
    ax.contourf(X_grid, Y_grid, Z_ratio,
                levels=np.linspace(Z_ratio.min(), threshold, 20),
                cmap="Reds_r", alpha=0.20)

    # 2. N_epsilon = S_red ∩ {d² <= d²* + 1} (hachuré jaune)
    N_eps_mask = ((Z_ratio <= threshold) & (Z_dist <= d2_star + 1.0)).astype(float)
    ax.contourf(X_grid, Y_grid, N_eps_mask,
                levels=[0.5, 1.5], colors=["#FFF176"], alpha=0.70, hatches=["///"])
    ax.contour(X_grid, Y_grid, N_eps_mask,
               levels=[0.5], colors=["#F9A825"], linewidths=1.0, linestyles=":")

    # 3. Ellipses de Mahalanobis (repère)
    levels_ell = sorted(set([round(d2_star, 3), 1.0, round(d2_star + 1.0, 3)]))
    ax.contour(X_grid, Y_grid, Z_dist,
               levels=levels_ell,
               colors=["#90CAF9"], linestyles="--", linewidths=1.4, alpha=0.7)

    # 4. Frontière de rupture (rouge épaisse)
    cs = ax.contour(X_grid, Y_grid, Z_ratio,
                    levels=[threshold], colors=["#C62828"], linewidths=3.0)
    ax.clabel(cs, fmt=r"$R(s)=\bar{R}$", inline=True, fontsize=12, colors="#C62828")

    # 5. Pool de candidats (points gris semi-transparents)
    if x_driver in pool.columns and y_driver in pool.columns:
        ax.scatter(pool[x_driver], pool[y_driver],
                   c="#BDBDBD", s=18, alpha=0.55, zorder=4,
                   label=f"Pool candidats  ($N={len(pool)}$)")

    # 6. Ancres géopolitiques (pentagones verts)
    if anchor_pts:
        gxs_a = [p[0] for p in anchor_pts]
        gys_a = [p[1] for p in anchor_pts]
        ax.scatter(gxs_a, gys_a,
                   c="#00C853", s=200, marker="p", zorder=8,
                   edgecolors="#1B5E20", linewidths=1.5,
                   label=f"Ancres géopolitiques  ($g_j$, $N={len(anchor_pts)}$)")
        for k, (gxa, gya) in enumerate(anchor_pts):
            ax.annotate(f"$g_{k+1}$",
                        xy=(gxa, gya),
                        xytext=(gxa + x_rng * 0.025, gya + y_rng * 0.025),
                        fontsize=9, color="#1B5E20", zorder=9,
                        fontweight="bold")

    # 7. Shortlist farthest-point (losanges noirs)
    if x_driver in shortlist.columns and y_driver in shortlist.columns:
        ax.scatter(shortlist[x_driver], shortlist[y_driver],
                   c="#212121", s=200, marker="D", zorder=10,
                   edgecolors="white", linewidths=1.5,
                   label=f"Shortlist farthest-point  ($P={len(shortlist)}$)")
        for rank, (_, row) in enumerate(shortlist.iterrows(), start=1):
            ax.annotate(
                f"  $s^{{({rank})}}$",
                xy=(row[x_driver], row[y_driver]),
                xytext=(row[x_driver] + x_rng * 0.025,
                        row[y_driver] + y_rng * 0.028),
                fontsize=9.5, color="#212121", zorder=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#555", lw=0.8)
            )

    # 8. Axes de référence
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.35, zorder=1)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.35, zorder=1)

    # 9. Baseline s = 0
    ax.scatter([0], [0], color="#2E7D32", s=140, zorder=12, marker="o",
               edgecolors="white", linewidths=1.5, label="Baseline  $s=0$")

    # 10. Design point s* (étoile rouge)
    ax.scatter([gx], [gy], color="#C62828", s=450, marker="*", zorder=13,
               edgecolors="white", linewidths=1.0,
               label=r"Design point  $s^*$")
    ax.annotate(r"$s^*$",
                xy=(gx, gy),
                xytext=(gx + x_rng * 0.07, gy + y_rng * 0.08),
                fontsize=12, color="#C62828", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C62828", alpha=0.85))

    ax.set_xlabel(f"Choc géopolitique  $g$  [{x_driver}]  [z-score]", fontsize=13, labelpad=10)
    ax.set_ylabel(f"Choc macro  $x$  [{y_driver}]  [z-score]", fontsize=13, labelpad=10)
    ax.set_title(
        "Figure 4 — Procédure de sélection de scénarios finis\n"
        r"Pool $C_N$ $\to$ Ancres géopolitiques $g_j$ $\to$ Shortlist farthest-point $C_P$",
        fontsize=14, fontweight="bold", pad=15
    )

    leg = ax.legend(loc="lower right", fontsize=10, framealpha=0.92,
                    edgecolor="#999", fancybox=True)
    leg.get_frame().set_linewidth(1.2)
    ax.grid(True, linestyle=":", alpha=0.4, color="#aaa")
    ax.set_facecolor("#FAFAFA")

    out_path = PLOTS / "fig4_scenario_selection.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 4] Sauvegardée : {out_path}")


if __name__ == "__main__":
    plot_fig4_scenario_selection()
