from __future__ import annotations

# ============================================================
# FIGURE 2 — ENSEMBLE NEAR-OPTIMAL N_epsilon
# ============================================================
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 2026)
# ---------------------------------------------------
# Figure 2 (p. 15-16) : "Geometry of the plausible reverse stress
#   scenario set N_epsilon in the (g, x) plane."
#
# Ce graphique montre la géométrie de l'ensemble epsilon-near-optimal :
#
# - Zone rouge (S_red) : R(g, x) <= R_omega
# - Frontière (rouge pleine) : R(g, x) = R_omega
# - Ellipses de plausibilité centrées en 0
# - Design point s* (étoile)
# - Contour externe relaxé : d²(s) = d²(s*) + epsilon
# - Zone hachurée N_epsilon = S_red ∩ {s : d²(s) <= d²(s*) + epsilon}
#
# Différence clé avec Figure 1 :
#   Fig 1 utilise la boule B_rho(s*) en y-space (voisinage local).
#   Fig 2 utilise le contour relaxé en Mahalanobis d²(s) <= d²* + eps
#   (near-optimal set), qui peut capturer des régions disconnectées.
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

from src.config import G_COL, PHI_NEAR
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


def plot_fig2_near_optimal(n_pts: int = 90, span: float = 3.5) -> None:
    """
    Reproduit la Figure 2 du papier :
    Géométrie de l'ensemble plausible N_epsilon en plan (g, x).

    Affiche :
    - Zone de rupture S_red (rouge pâle)
    - Frontière R(s) = R_omega (courbe rouge épaisse)
    - Ellipses de Mahalanobis centrées en 0 (contours bleus)
    - Design point s* (étoile rouge)
    - Contour relaxé d²(s) = d²* + epsilon (ellipse orange pointillée)
    - Ensemble N_epsilon (zone hachurée orange) = S_red ∩ {d² <= d²* + eps}
    """
    engine, design_point, sigma, summary = load_viz_context()

    macro_drivers = design_point.drop(G_COL).abs().sort_values(ascending=False)
    x_driver = G_COL
    y_driver = macro_drivers.index[0]
    print(f"[Fig 2] Axes : X={x_driver} / Y={y_driver}")

    inv_sigma = np.linalg.inv(sigma.values)

    xx, yy, (X_grid, Y_grid) = make_grid(
        design_point, x_driver, y_driver, span=span, n_pts=n_pts
    )

    Z_ratio, Z_dist = compute_Z_ratio_and_dist(
        X_grid, Y_grid, design_point, x_driver, y_driver,
        engine, inv_sigma
    )

    threshold = float(engine.R_omega)
    d2_star   = float(summary["design_point_d2"])
    eps       = PHI_NEAR

    gx = float(design_point[x_driver])
    gy = float(design_point[y_driver])
    x_min, x_max = X_grid.min(), X_grid.max()
    y_min, y_max = Y_grid.min(), Y_grid.max()

    # --------------------------------------------------------
    # Tracé
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("white")

    # 1. Zone de rupture S_red (fond rouge clair)
    ax.contourf(X_grid, Y_grid, Z_ratio,
                levels=np.linspace(Z_ratio.min(), threshold, 30),
                cmap="Reds_r", alpha=0.35)

    # 2. N_epsilon = S_red ∩ {d² <= d²* + eps} (hachuré orange)
    N_eps_mask = ((Z_ratio <= threshold) & (Z_dist <= d2_star + eps)).astype(float)
    ax.contourf(X_grid, Y_grid, N_eps_mask,
                levels=[0.5, 1.5], colors=["#FF8F00"], alpha=0.50, hatches=["\\\\\\"])
    ax.contour(X_grid, Y_grid, N_eps_mask,
               levels=[0.5], colors=["#E65100"], linewidths=0.8, linestyles=":")

    # 3. Frontière R(s) = R_omega (ligne rouge épaisse)
    cs = ax.contour(X_grid, Y_grid, Z_ratio,
                    levels=[threshold], colors=["#C62828"], linewidths=3.0)
    ax.clabel(cs, fmt=r"$R(s)=\bar{R}$", inline=True, fontsize=12, colors="#C62828")

    # 4. Ellipses de Mahalanobis centrées en 0
    levels_ell = sorted(set([round(d2_star, 3), 1.0, 4.0]))
    cse = ax.contour(X_grid, Y_grid, Z_dist,
                     levels=levels_ell,
                     colors=["#1565C0"], linestyles="--", linewidths=1.8, alpha=0.85)
    ax.clabel(cse, fmt={v: f"$d^2={v:.2f}$" for v in levels_ell},
              inline=True, fontsize=11, colors="#1565C0")

    # 5. Contour relaxé d²(s) = d²* + eps (orange épais pointillé)
    ax.contour(X_grid, Y_grid, Z_dist,
               levels=[d2_star + eps],
               colors=["#E65100"], linestyles="--", linewidths=2.8)

    # 6. Axes de référence
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4, zorder=1)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4, zorder=1)

    # 7. Baseline s = 0
    ax.scatter([0], [0], color="#2E7D32", s=140, zorder=12, marker="o",
               edgecolors="white", linewidths=1.5, label="Baseline $s=0$")

    # 8. Design point s*
    ax.scatter([gx], [gy], color="#C62828", s=400, marker="*", zorder=13,
               edgecolors="white", linewidths=1.0, label=r"Design point $s^*$")

    # --------------------------------------------------------
    # Annotations des zones en coordonnées axes (stables)
    # --------------------------------------------------------
    ax.text(0.72, 0.82, r"$S_{\rm red}$",
            transform=ax.transAxes, fontsize=18, color="#C62828",
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C62828", alpha=0.7))

    ax.text(0.63, 0.42, r"$\mathcal{N}_\varepsilon$",
            transform=ax.transAxes, fontsize=18, color="#E65100",
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E65100", alpha=0.7))

    # Annotation du contour relaxé
    ax.text(0.28, 0.68,
            rf"$d^2_\Sigma(s)=d^2_\Sigma(s^*)+\varepsilon={d2_star + eps:.2f}$",
            transform=ax.transAxes, fontsize=10, color="#E65100",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E65100", alpha=0.75))

    # Flèche s*
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
        r"Figure 2 — Ensemble plausible $\mathcal{N}_\varepsilon$ des scénarios near-optimal" + "\n"
        r"$\mathcal{N}_\varepsilon = S_{\rm red} \cap \{s : d^2_\Sigma(s) \leq d^2_\Sigma(s^*)+\varepsilon\}$",
        fontsize=14, fontweight="bold", pad=15
    )

    leg = ax.legend(loc="lower right", fontsize=12, framealpha=0.9,
                    edgecolor="#999", fancybox=True)
    leg.get_frame().set_linewidth(1.2)
    ax.grid(True, linestyle=":", alpha=0.4, color="#aaa")
    ax.set_facecolor("#FAFAFA")

    out_path = PLOTS / "fig2_near_optimal.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 2] Sauvegardée : {out_path}")


if __name__ == "__main__":
    plot_fig2_near_optimal()
