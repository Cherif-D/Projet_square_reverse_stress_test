from __future__ import annotations

# ============================================================
# FIGURE 3 — FEUILLE DE ROUTE D'IMPLEMENTATION
# ============================================================
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 2026)
# ---------------------------------------------------
# Figure 3 (p. 19) : "Implementation roadmap of the geopolitical
#   reverse stress testing framework."
#
# Ce graphique présente le diagramme de flux en 4 étapes :
#
# Étape 1 : Définition du scénario et modèle de référence
#           - Choix de l'indicateur géopolitique et des facteurs macro
#           - Estimation de la structure de covariance Sigma
#
# Étape 2 : Calibration des mappings de transmission
#           - Sensibilités PD(g, x) : coefficients delta_k et beta_k
#           - Sensibilités LGD(g, x) : coefficients eta_k et gamma_k
#
# Étape 3 : Spécification du cadre capital
#           - Agrégation du portefeuille L_q(g, x)
#           - Calcul des RWA et du ratio CET1
#
# Étape 4 : Résolution numérique du RST
#           - Optimisation contrainte -> design point s*
#           - Construction de S_rho et N_epsilon
#           - Réduction maximin -> shortlist gouvernance-ready
# ============================================================

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.paths import OUTPUTS

PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

# Palette couleurs par étape
STEP_COLORS  = ["#1565C0", "#2E7D32", "#6A1B9A", "#C62828"]
ARROW_COLOR  = "#424242"
BG_COLOR     = "#F8F9FA"
TITLE_COLOR  = "#1A1A1A"


def _box(ax, x, y, w, h, text, fc, fontsize=11, ec="white"):
    """Dessine une boîte arrondie colorée avec texte centré."""
    patch = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=ec,
        linewidth=2.5, alpha=0.92, zorder=3
    )
    ax.add_patch(patch)
    ax.text(x, y, text,
            ha="center", va="center",
            fontsize=fontsize, color="white",
            fontweight="bold",
            zorder=4, multialignment="center")


def _content_box(ax, x, y, w, h, text, color, fontsize=9.5):
    """Dessine une boîte de contenu blanche avec bordure colorée."""
    patch = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor="white", edgecolor=color,
        linewidth=2.2, alpha=0.97, zorder=3
    )
    ax.add_patch(patch)
    ax.text(x, y, text,
            ha="center", va="center",
            fontsize=fontsize, color="#1A1A1A",
            zorder=4, multialignment="left",
            linespacing=1.55)


def _arrow(ax, x1, y1, x2, y2):
    """Flèche entre deux boîtes."""
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ARROW_COLOR,
                    lw=2.2,
                    mutation_scale=18,
                ),
                zorder=2)


def plot_fig3_roadmap() -> None:
    """
    Reproduit la Figure 3 du papier :
    Feuille de route d'implémentation du framework RST géopolitique.

    Diagramme de flux en 4 étapes séquentielles avec inputs/outputs.
    """
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis("off")

    # --------------------------------------------------------
    # Titre principal
    # --------------------------------------------------------
    ax.text(9, 10.4,
            "Figure 3 — Feuille de route d'implémentation\n"
            "du cadre de Reverse Stress Test géopolitique",
            ha="center", va="center",
            fontsize=16, fontweight="bold", color=TITLE_COLOR)

    # --------------------------------------------------------
    # Coordonnées des 4 étapes
    # --------------------------------------------------------
    xs         = [2.2, 6.5, 11.0, 15.5]
    box_w      = 3.8
    h_title    = 1.0
    h_content  = 3.2
    y_title    = 7.8
    y_content  = 5.0

    steps = [
        (
            "Étape 1\nScénario &\nModèle de référence",
            "• Choc géopolitique : g (GPRD)\n"
            "• Facteurs macro : x = (GDP, VIX,\n"
            "   SP500, WTI, Curve, Unrate, EPU)\n"
            "• Estimation Σ — Ledoit-Wolf\n"
            "• Standardisation (z-scores)\n"
            "• Recentrage sur état courant",
        ),
        (
            "Étape 2\nMappings de\ntransmission",
            "• Sensibilités PD (eq. 7) :\n"
            "   logit(PD_i) = logit(PD⁰_i)\n"
            "              + β_kᵀ x + δ_k g\n"
            "• Sensibilités LGD (eq. 10) :\n"
            "   LGD_i = LGD⁰_i + γ_kᵀ x + η_k g\n"
            "• Calibration par secteur k",
        ),
        (
            "Étape 3\nCadre capital",
            "• Perte queue (eq. 16) :\n"
            "   L_q = Σ EAD_i · LGD_i · Φ_tail\n"
            "• RWA (eq. 21) :\n"
            "   RWA = RWA₀ + Σ α_i · ΔPD_i\n"
            "• Ratio (eq. 22) :\n"
            "   R(s) = CET1(s) / RWA(s)",
        ),
        (
            "Étape 4\nRésolution\nnumérique",
            "• Optimisation SLSQP (eq. 24) :\n"
            "   min ½‖y‖²  s.c.  R(Ly)≤R̄, g≥0\n"
            "• Multi-start → design point s*\n"
            "• Pool S_ρ ∪ N_ε (538 scénarios)\n"
            "• Shortlist maximin (eq. 49)\n"
            "   → 8 scénarios gouvernance-ready",
        ),
    ]

    inputs_labels = [
        "data/interim/\nmacro.csv",
        "inputs/\nsector_params.csv",
        "inputs/exposures.csv\ninputs/capital.csv",
        "outputs/tables/\ndesign_point.csv …",
    ]
    outputs_labels = [
        "Sigma.csv\nscenario_standardized.csv",
        "PD_stress(s)\nLGD_stress(s)",
        "R(s) = CET1(s)/RWA(s)",
        "summary.json\nsummary.md",
    ]

    for i, (x, (title, content)) in enumerate(zip(xs, steps)):
        color = STEP_COLORS[i]

        # Boîte titre (colorée)
        _box(ax, x, y_title, box_w, h_title, title, color, fontsize=11)

        # Boîte contenu (blanche / bordure colorée)
        _content_box(ax, x, y_content, box_w, h_content, content, color, fontsize=9.5)

        # Flèche vers étape suivante
        if i < len(xs) - 1:
            _arrow(ax,
                   x + box_w / 2, y_title,
                   xs[i + 1] - box_w / 2, y_title)

        # Input (en haut)
        ax.annotate(inputs_labels[i],
                    xy=(x, y_title + h_title / 2 + 0.05),
                    xytext=(x, 9.3),
                    ha="center", va="bottom", fontsize=8.5,
                    color="#444",
                    arrowprops=dict(arrowstyle="->", color="#999", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#EEE", ec="#CCC"),
                    zorder=5)

        # Output (en bas)
        ax.annotate(outputs_labels[i],
                    xy=(x, y_content - h_content / 2 - 0.05),
                    xytext=(x, 1.8),
                    ha="center", va="top", fontsize=8.5,
                    color="#444",
                    arrowprops=dict(arrowstyle="->", color="#999", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#EEE", ec="#CCC"),
                    zorder=5)

    # --------------------------------------------------------
    # Étiquettes "INPUTS" et "OUTPUTS"
    # --------------------------------------------------------
    ax.text(0.5, 9.3, "Inputs ↑",
            ha="left", va="center", fontsize=10, color="#444",
            fontstyle="italic")
    ax.text(0.5, 1.8, "Outputs ↓",
            ha="left", va="center", fontsize=10, color="#444",
            fontstyle="italic")

    # Ligne de séparation horizontale
    for y_sep in [8.45, 3.65]:
        ax.axhline(y_sep, color="#CCC", linewidth=0.8, linestyle="--", zorder=1)

    out_path = PLOTS / "fig3_roadmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[Fig 3] Sauvegardée : {out_path}")


if __name__ == "__main__":
    plot_fig3_roadmap()
