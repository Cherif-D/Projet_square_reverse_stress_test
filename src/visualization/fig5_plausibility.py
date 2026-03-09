from __future__ import annotations

# ============================================================
# FIGURE 5 — PLAUSIBILITE GAUSSIENNE VS. STUDENT-t
# ============================================================
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 2026)
# ---------------------------------------------------
# Figure 5 (Annexe A.1, p. 30) : "Gaussian vs. Student-t reference
#   plausibility on a common colour scale (scenario s = (g, x))."
#
# Ce graphique compare l'évaluation de la rareté des scénarios selon
# deux hypothèses de distribution de référence :
#
# Panneau gauche — Distribution Gaussienne :
#   Rareté : -log10(P(d²(S) >= d²(s)))
#   où d²(S) ~ chi²_d
#
# Panneau droit — Distribution Student-t (nu = 6 d.d.l.) :
#   Même métrique de rareté, mais d²(s) a une distribution à
#   queues plus épaisses sous Student-t.
#   Pour multivariate Student-t(nu) avec même Sigma :
#   d²(s)/d ~ (nu-2)/nu * F(d, nu)  [approximation]
#
# Différence clé :
#   - Sous Gaussienne, les scénarios distants sont très rares.
#   - Sous Student-t, ils sont relativement moins rares (queues épaisses).
#   Cela affecte directement le classement des scénarios candidats
#   et justifie l'utilisation d'une distribution de référence robuste.
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, f as f_dist

from src.config import G_COL
from src.paths import OUTPUTS
from src.visualization.helpers import load_viz_data

PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

# Degrés de liberté Student-t (nu = 6 comme dans le papier Annexe A.1)
NU_STUDENT = 6


def _gaussian_rarity(d2_grid: np.ndarray, d: int) -> np.ndarray:
    """
    Rareté Gaussienne : -log10(P(chi²_d >= d²)).

    Sous hypothèse gaussienne :
        d²(S) = s'Sigma^{-1}s ~ chi²_d
    """
    p_val = 1.0 - chi2.cdf(d2_grid, df=d)
    p_val = np.clip(p_val, 1e-15, 1.0)
    return -np.log10(p_val)


def _student_rarity(d2_grid: np.ndarray, d: int, nu: int) -> np.ndarray:
    """
    Rareté Student-t multivariée : -log10(P(d²(S) >= d²)).

    Pour une distribution Student-t(nu) multivariée de dimension d,
    rescalée pour avoir la même matrice de covariance Sigma :

        d²(s) * (nu - 2) / (nu * d) ~ F(d, nu)

    Donc : P(d²(S) >= d²) = 1 - F_{d,nu}(d²*(nu-2)/(nu*d))

    Cette formule est conforme à l'Annexe A.1 du papier.
    """
    scale = (nu - 2.0) / (nu * d)
    f_val = d2_grid * scale
    p_val = 1.0 - f_dist.cdf(f_val, dfn=d, dfd=nu)
    p_val = np.clip(p_val, 1e-15, 1.0)
    return -np.log10(p_val)


def plot_fig5_plausibility(n_pts: int = 100, span: float = 4.5,
                           nu: int = NU_STUDENT) -> None:
    """
    Reproduit la Figure 5 du papier :
    Carte de chaleur comparative Gaussienne vs. Student-t.

    Deux panneaux côte à côte montrant -log10(rareté) sur la même
    échelle de couleur, mettant en évidence l'effet queues épaisses.
    """
    exposures, capital, sector_params, design_point, sigma, summary = load_viz_data()

    macro_drivers = design_point.drop(G_COL).abs().sort_values(ascending=False)
    x_driver = G_COL
    y_driver = macro_drivers.index[0]
    print(f"[Fig 5] Axes : X={x_driver} / Y={y_driver}")

    inv_sigma    = np.linalg.inv(sigma.values)
    feature_cols = list(design_point.index)
    d            = len(feature_cols)

    # --------------------------------------------------------
    # Grille 2D symétrique centrée en 0
    # --------------------------------------------------------
    xx_center = float(design_point[x_driver])
    yy_center = float(design_point[y_driver])
    span_eff  = max(span, abs(xx_center) * 4, abs(yy_center) * 4)

    xx = np.linspace(-span_eff, span_eff, n_pts)
    yy = np.linspace(-span_eff, span_eff, n_pts)
    X_grid, Y_grid = np.meshgrid(xx, yy)

    # --------------------------------------------------------
    # Calcul de d²(g, x) — autres composantes fixées à 0
    # --------------------------------------------------------
    x_idx = feature_cols.index(x_driver)
    y_idx = feature_cols.index(y_driver)

    Z_d2 = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            s_vec         = np.zeros(d)
            s_vec[x_idx]  = X_grid[i, j]
            s_vec[y_idx]  = Y_grid[i, j]
            Z_d2[i, j]    = float(s_vec @ inv_sigma @ s_vec)

    # --------------------------------------------------------
    # Rareté Gaussienne et Student-t
    # --------------------------------------------------------
    Z_gauss   = _gaussian_rarity(Z_d2, d)
    Z_student = _student_rarity(Z_d2, d, nu)

    # Colorbar partagée — percentile 97 pour ne pas saturer
    vmin = 0.0
    vmax = max(float(np.nanpercentile(Z_gauss, 97)),
               float(np.nanpercentile(Z_student, 97)))
    n_levels = 25

    # --------------------------------------------------------
    # Tracé : deux panneaux côte à côte
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    fig.patch.set_facecolor("white")

    titles = [
        f"Distribution Gaussienne\n"
        f"$d^2_\\Sigma(S) \\sim \\chi^2_{{{d}}}$",
        f"Distribution Student-$t$  ($\\nu={nu}$)\n"
        f"$d^2_\\Sigma(S)\\cdot(\\nu-2)/(\\nu d) \\sim F({d},{nu})$",
    ]
    Z_maps = [Z_gauss, Z_student]
    panel_labels = ["(a)", "(b)"]

    im = None
    for ax, title, Z_map, panel in zip(axes, titles, Z_maps, panel_labels):
        # Fond coloré
        im = ax.contourf(X_grid, Y_grid, Z_map,
                         levels=np.linspace(vmin, vmax, n_levels),
                         cmap="RdYlBu_r", extend="max")

        # Iso-contours Mahalanobis (noirs pointillés)
        d2_star_val = float(summary["design_point_d2"])
        mah_levels  = sorted(set([1.0, 4.0, 9.0, round(d2_star_val, 4)]))
        cse = ax.contour(X_grid, Y_grid, Z_d2,
                         levels=mah_levels,
                         colors="black", linestyles="dashed",
                         linewidths=1.2, alpha=0.6)
        ax.clabel(cse, fmt={v: f"$d^2={v:.2f}$" for v in mah_levels},
                  inline=True, fontsize=9, colors="black")

        # Axes de référence
        ax.axhline(0, color="white", linewidth=0.6, alpha=0.5, zorder=2)
        ax.axvline(0, color="white", linewidth=0.6, alpha=0.5, zorder=2)

        # Baseline
        ax.scatter([0], [0], color="#00E676", s=150, zorder=10, marker="o",
                   edgecolors="black", linewidths=1.2, label="Baseline $s=0$")

        # Design point s*
        ax.scatter([xx_center], [yy_center], color="white", s=350,
                   marker="*", zorder=11, edgecolors="black", linewidths=1.0,
                   label=r"$s^*$")

        ax.set_xlabel(f"$g$  [{x_driver}]  [z-score]", fontsize=12, labelpad=8)
        ax.set_title(f"{panel}  {title}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.8,
                  edgecolor="#999", fancybox=True)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel(f"$x$  [{y_driver}]  [z-score]", fontsize=12, labelpad=8)

    # Colorbar commune (à droite)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015, aspect=30)
    cbar.set_label(
        r"Rareté du scénario $-\log_{10}\,\mathbb{P}(d^2_\Sigma(S) \geq d^2_\Sigma(s))$",
        fontsize=11, labelpad=12
    )
    cbar.ax.tick_params(labelsize=10)

    # Annotation explicative
    fig.text(
        0.50, 0.01,
        "Note : les queues épaisses de la Student-t rendent les scénarios extrêmes "
        "moins rares (couleurs plus claires) que sous Gaussienne.",
        ha="center", va="bottom", fontsize=9, color="#555",
        style="italic"
    )

    fig.suptitle(
        "Figure 5 — Plausibilité Gaussienne vs. Student-$t$  "
        r"($-\log_{10}$ de la p-valeur)  sur une échelle commune",
        fontsize=14, fontweight="bold", y=1.02
    )

    out_path = PLOTS / "fig5_plausibility.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Fig 5] Sauvegardée : {out_path}")


if __name__ == "__main__":
    plot_fig5_plausibility()
