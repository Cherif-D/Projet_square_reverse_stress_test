from __future__ import annotations

# ============================================================
# FIGURE 1 — GEOMETRIE DU PROBLEME RST
# ============================================================
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 2026)
# ---------------------------------------------------
# Figure 1 (p. 14-15) : "Geometry of the reverse-stress problem
#   in the (g, x) plane."
#
# Ce graphique illustre l'interprétation géométrique du RST en 2D :
#
# - Zone rouge (S_red) : R(g, x) <= R_omega (breach region)
# - Frontière (rouge pleine) : R(g, x) = R_omega (breakdown frontier)
# - Ellipses de plausibilité : iso-niveaux de d²(s) = s'Sigma^{-1}s
#   centrées en s = 0 (distribution de référence)
# - Point de design s* : tangence entre la plus petite ellipse
#   et la frontière, sous contrainte g >= 0
# - Boule locale B_eta(s*) : ellipse de rayon eta autour de s*
# - Voisinage local S_eta = S_red ∩ B_eta(s*) : zone hachurée
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

from src.config import ETA_LOCAL, G_COL
from src.paths import OUTPUTS
from src.visualization.helpers import (
    compute_Z_ratio_and_dist,
    load_viz_context,
    make_grid,
)

PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

# Style global
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def plot_fig1_geometry(n_pts: int = 90, span: float = 3.5) -> None:
    """
    Reproduit la Figure 1 du papier :
    Géométrie du problème de reverse stress test en plan (g, x).

    Affiche :
    - Zone de rupture S_red (rouge pâle)
    - Frontière R(s) = R_omega (courbe rouge)
    - Ellipses de Mahalanobis centrées en 0 (contours bleus)
    - Design point s* (étoile rouge)
    - Boule locale B_eta(s*) autour de s* (ellipse verte pointillée)
    - Voisinage local S_eta (zone hachurée verte)
    """
    engine, design_point, sigma, summary = load_viz_context()

    # Axes : géopolitique (X) vs premier driver macro (Y)
    macro_drivers = design_point.drop(G_COL).abs().sort_values(ascending=False)
    x_driver = G_COL
    y_driver = macro_drivers.index[0]
    print(f"[Fig 1] Axes : X={x_driver} / Y={y_driver}")

    inv_sigma    = np.linalg.inv(sigma.values)
    feature_cols = list(design_point.index)
    L            = np.linalg.cholesky(sigma.values)

    xx, yy, (X_grid, Y_grid) = make_grid(
        design_point, x_driver, y_driver, span=span, n_pts=n_pts
    )

    Z_ratio, Z_dist = compute_Z_ratio_and_dist(
        X_grid, Y_grid, design_point, x_driver, y_driver,
        engine, inv_sigma
    )

    threshold = float(engine.R_omega)
    d2_star   = float(summary["design_point_d2"])
    rho_local = ETA_LOCAL

    # --------------------------------------------------------
    # Boule locale en y-space : ||y - y*||² <= eta
    # --------------------------------------------------------
    s_star = design_point.values.copy()
    y_star = np.linalg.solve(L, s_star)

    x_idx = feature_cols.index(x_driver)
    y_idx = feature_cols.index(y_driver)

    Z_dist_local = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            s_vec          = design_point.values.copy()
            s_vec[x_idx]   = X_grid[i, j]
            s_vec[y_idx]   = Y_grid[i, j]
            y_vec          = np.linalg.solve(L, s_vec)
            Z_dist_local[i, j] = np.dot(y_vec - y_star, y_vec - y_star)

    # --------------------------------------------------------
    # Tracé
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("white")

    x_min, x_max = X_grid.min(), X_grid.max()
    y_min, y_max = Y_grid.min(), Y_grid.max()

    # 1. Zone de rupture S_red (fond rouge clair)
    ax.contourf(X_grid, Y_grid, Z_ratio,
                levels=np.linspace(Z_ratio.min(), threshold, 30),
                cmap="Reds_r", alpha=0.35)

    # 2. Frontière R(s) = R_omega (ligne rouge épaisse)
    cs = ax.contour(X_grid, Y_grid, Z_ratio,
                    levels=[threshold],
                    colors=["#C62828"], linewidths=3.0)
    ax.clabel(cs, fmt=r"$R(s)=R_\omega$", inline=True, fontsize=12, colors="#C62828")

    # 3. Ellipses de Mahalanobis centrées en 0
    levels_ell = sorted(set([round(d2_star, 3), 1.0, 4.0]))
    cse = ax.contour(X_grid, Y_grid, Z_dist,
                     levels=levels_ell,
                     colors=["#1565C0"], linestyles="--", linewidths=1.8, alpha=0.8)
    ax.clabel(cse, fmt={v: f"$d^2={v:.2f}$" for v in levels_ell},
              inline=True, fontsize=11, colors="#1565C0")

    # 4. Voisinage local S_eta = S_red ∩ B_eta(s*) (hachuré vert)
    S_rho_mask = ((Z_ratio <= threshold) & (Z_dist_local <= rho_local)).astype(float)
    ax.contourf(X_grid, Y_grid, S_rho_mask,
                levels=[0.5, 1.5], colors=["#43A047"], alpha=0.45, hatches=["///"])
    ax.contour(X_grid, Y_grid, S_rho_mask,
               levels=[0.5], colors=["#2E7D32"], linewidths=0.8, linestyles=":")

    # 5. Boule locale B_eta(s*) (ellipse verte pointillée)
    ax.contour(X_grid, Y_grid, Z_dist_local,
               levels=[rho_local],
               colors=["#2E7D32"], linestyles="--", linewidths=2.2)

    # 6. Axes de référence
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4, zorder=1)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4, zorder=1)

    # 7. Baseline s = 0
    ax.scatter([0], [0], color="#2E7D32", s=140, zorder=12, marker="o",
               edgecolors="white", linewidths=1.5, label="Baseline $s=0$")

    # 8. Design point s* (grande étoile rouge)
    gx = float(design_point[x_driver])
    gy = float(design_point[y_driver])
    ax.scatter([gx], [gy], color="#C62828", s=400, marker="*", zorder=13,
               edgecolors="white", linewidths=1.0,
               label=r"Design point $s^*$")

    # --------------------------------------------------------
    # Annotations des zones en coordonnées axes (stables)
    # --------------------------------------------------------
    ax.text(0.72, 0.82, r"$S_{\rm red}$",
            transform=ax.transAxes, fontsize=18, color="#C62828",
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C62828", alpha=0.7))

    ax.text(0.62, 0.45, r"$\mathcal{S}_\eta$",
            transform=ax.transAxes, fontsize=16, color="#2E7D32",
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2E7D32", alpha=0.7))

    ax.text(0.60, 0.30, r"$\mathcal{B}_\eta(s^*)$",
            transform=ax.transAxes, fontsize=13, color="#2E7D32",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#2E7D32", alpha=0.6))

    # Flèche du design point vers son label
    ax.annotate(
        rf"$s^*=({gx:.3f},\,{gy:.3f})$",
        xy=(gx, gy),
        xytext=(gx + (x_max - x_min) * 0.12, gy + (y_max - y_min) * 0.10),
        fontsize=11, color="#C62828",
        arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C62828", alpha=0.8)
    )

    ax.set_xlabel(f"Choc géopolitique  $g$  [{x_driver}]  [z-score]", fontsize=13, labelpad=10)
    ax.set_ylabel(f"Choc macro  $x$  [{y_driver}]  [z-score]", fontsize=13, labelpad=10)
    ax.set_title(
        "Figure 1 — Géométrie du problème de reverse stress test\n"
        r"Région de rupture $S_{\rm red}$,  frontière $R(s)=R_\omega$,  voisinage local $\mathcal{S}_\eta$",
        fontsize=14, fontweight="bold", pad=15
    )

    leg = ax.legend(loc="lower right", fontsize=12, framealpha=0.9,
                    edgecolor="#999", fancybox=True)
    leg.get_frame().set_linewidth(1.2)
    ax.grid(True, linestyle=":", alpha=0.4, color="#aaa")
    ax.set_facecolor("#FAFAFA")

    out_path = PLOTS / "fig1_geometry.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 1] Sauvegardée : {out_path}")


if __name__ == "__main__":
    plot_fig1_geometry()
