from __future__ import annotations

# ============================================================
# GRAPHIQUE : PROFIL DU SCENARIO DE RUPTURE (DESIGN POINT)
# ============================================================
#
# Bar chart horizontal des chocs en z-scores au design point.
# Rouge = choc négatif (baisse PIB, etc.)
# Bleu  = choc positif (hausse géopolitique, VIX, etc.)
#
# Ce graphique est issu de visualize_results.py (plot_scenario).
# Il est ici intégré dans l'architecture src/visualization/.
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.paths import OUTPUTS, TABLES

PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

# Correspondance des noms de variables pour l'affichage
VAR_LABELS = {
    "shock_GPRD"   : "Géopolitique (GPRD)",
    "shock_gdp"    : "Croissance PIB",
    "shock_vix"    : "Volatilité (VIX)",
    "shock_sp500"  : "Marchés actions (SP500)",
    "shock_wti"    : "Prix pétrole (WTI)",
    "shock_t10Y2Y" : "Pente courbe (10Y-2Y)",
    "shock_unrate" : "Taux chômage",
    "shock_epu"    : "Incertitude politique (EPU)",
}


def plot_scenario_profile() -> None:
    """
    Graphique du scénario de rupture :
    Bar chart horizontal des chocs (z-scores) au design point.

    Chaque barre représente l'intensité du choc (en écarts-types)
    pour une variable du scénario s = (g, x).
    """
    dp_path = TABLES / "design_point.csv"
    if not dp_path.exists():
        print("[Scenario profile] design_point.csv introuvable.")
        return

    # Lecture robuste : on ne garde que les lignes "shock_*"
    df_raw = pd.read_csv(dp_path, header=None, index_col=0)
    df_raw = df_raw[
        df_raw.index.to_series().apply(
            lambda x: isinstance(x, str) and x.startswith("shock_")
        )
    ].copy()
    df_raw.columns   = ["choc"]
    df_raw["choc"]   = pd.to_numeric(df_raw["choc"], errors="coerce")
    df_raw            = df_raw.dropna()

    # Labels lisibles
    df_raw["label"] = df_raw.index.map(lambda k: VAR_LABELS.get(k, k))

    # Tri par valeur absolue
    df_raw["abs"] = df_raw["choc"].abs()
    df_raw = df_raw.sort_values("abs", ascending=True).drop(columns="abs")

    # Couleurs : rouge si choc négatif, bleu si positif
    colors = ["#C62828" if v < 0 else "#1565C0" for v in df_raw["choc"]]

    # --------------------------------------------------------
    # Tracé
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    bars = ax.barh(df_raw["label"], df_raw["choc"],
                   color=colors, alpha=0.85,
                   edgecolor="white", linewidth=1.2,
                   height=0.65)

    # Ligne zéro
    ax.axvline(0, color="#424242", linewidth=1.5, zorder=3)

    # Valeurs au bout des barres (toujours lisibles)
    for bar, val in zip(bars, df_raw["choc"]):
        w     = bar.get_width()
        x_txt = w + 0.003 if w >= 0 else w - 0.003
        ha    = "left" if w >= 0 else "right"
        ax.text(x_txt,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}σ",
                va="center", ha=ha, fontsize=11,
                color="#212121", fontweight="bold")

    # Bande colorée en arrière-plan pour g (géopolitique)
    gprd_label = VAR_LABELS.get("shock_GPRD", "shock_GPRD")
    if gprd_label in df_raw["label"].values:
        gprd_idx = list(df_raw["label"]).index(gprd_label)
        ax.axhspan(gprd_idx - 0.45, gprd_idx + 0.45,
                   color="#FFF9C4", alpha=0.5, zorder=0)
        ax.annotate("Choc géopolitique principal",
                    xy=(0, gprd_idx), xytext=(0.02, gprd_idx + 0.6),
                    fontsize=9, color="#F57F17",
                    arrowprops=dict(arrowstyle="->", color="#F57F17", lw=1))

    ax.set_xlabel("Intensité du choc [écarts-types par rapport à la baseline]",
                  fontsize=13, labelpad=10)
    ax.set_title(
        "Profil du scénario de rupture — Design point $s^*$\n"
        "Décomposition par variable  (unité : z-score)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#ccc")
    ax.set_xlim(df_raw["choc"].min() * 1.35, df_raw["choc"].max() * 1.35)

    # Légende couleur
    legend_elements = [
        mpatches.Patch(facecolor="#C62828", alpha=0.85, label="Choc négatif  (ex. baisse PIB)"),
        mpatches.Patch(facecolor="#1565C0", alpha=0.85, label="Choc positif  (ex. hausse géopolitique)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11,
              framealpha=0.9, edgecolor="#999")

    out_path = PLOTS / "scenario_profile.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Scenario profile] Sauvegardée : {out_path}")


if __name__ == "__main__":
    plot_scenario_profile()
