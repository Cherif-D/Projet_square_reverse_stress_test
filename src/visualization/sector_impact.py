from __future__ import annotations

# ============================================================
# GRAPHIQUE : IMPACT SECTORIEL AU DESIGN POINT
# ============================================================
#
# Ce module regroupe deux visualisations sectorielles :
#
# 1. plot_sector_losses() :
#    Bar chart des pertes de queue L_q par secteur (depuis visualize_results.py)
#
# 2. plot_sector_pd_lgd() :
#    Graphique à bulles PD_stress vs LGD_stress par secteur,
#    taille proportionnelle à l'EAD sectorielle.
#    Visualise la double transmission géopolitique->PD et LGD.
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from src.paths import OUTPUTS, TABLES

PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

# Correspondance secteur -> couleur distincte
SECTOR_COLORS = {
    "Energy"       : "#E53935",
    "Transport"    : "#FB8C00",
    "Manufacturing": "#FDD835",
    "Consumer"     : "#43A047",
    "Tech"         : "#1E88E5",
    "Utilities"    : "#8E24AA",
    "Defense"      : "#00ACC1",
    "RealEstate"   : "#6D4C41",
}

SECTOR_LABELS = {
    "Energy"       : "Énergie",
    "Transport"    : "Transport",
    "Manufacturing": "Industrie",
    "Consumer"     : "Consommation",
    "Tech"         : "Technologie",
    "Utilities"    : "Services publics",
    "Defense"      : "Défense",
    "RealEstate"   : "Immobilier",
}


def _sector_color(name: str) -> str:
    return SECTOR_COLORS.get(name, "#78909C")


def _sector_label(name: str) -> str:
    return SECTOR_LABELS.get(name, name)


def plot_sector_losses() -> None:
    """
    Bar chart horizontal des pertes de queue L_q par secteur.
    Reproduit plot_sector_impact() de visualize_results.py.
    """
    diag_path = TABLES / "sector_diagnostics_at_design_point.csv"
    if not diag_path.exists():
        print("[Sector losses] sector_diagnostics_at_design_point.csv introuvable.")
        return

    df = pd.read_csv(diag_path).sort_values("LossQ_sector", ascending=True)
    labels = [_sector_label(s) for s in df["sector"]]
    colors = [_sector_color(s) for s in df["sector"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    bars = ax.barh(labels, df["LossQ_sector"],
                   color=colors, alpha=0.88,
                   edgecolor="white", linewidth=1.2, height=0.65)

    # Valeurs au bout
    for bar, val in zip(bars, df["LossQ_sector"]):
        ax.text(bar.get_width() + df["LossQ_sector"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", fontsize=11,
                color="#212121", fontweight="bold")

    # Delta-loss en annotation secondaire
    if "DeltaLossQ_sector" in df.columns:
        for bar, dval in zip(bars, df["DeltaLossQ_sector"]):
            sign = "+" if dval >= 0 else ""
            ax.text(bar.get_width() * 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{sign}{dval:.4f}",
                    va="center", ha="center", fontsize=8.5,
                    color="white", alpha=0.9)

    ax.axvline(0, color="#424242", linewidth=1.2, zorder=3)
    ax.set_xlabel("Perte de queue  $L_q$  (contribution sectorielle)", fontsize=13, labelpad=10)
    ax.set_title(
        "Répartition des pertes de queue par secteur — Design point $s^*$\n"
        "(valeurs blanches = $\\Delta L_q$ vs baseline)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#ccc")
    ax.set_xlim(0, df["LossQ_sector"].max() * 1.20)

    out_path = PLOTS / "sector_impact.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Sector losses] Sauvegardée : {out_path}")


def plot_sector_pd_lgd() -> None:
    """
    Graphique à bulles : PD_stress vs LGD_stress par secteur.

    - Axe X : PD moyenne pondérée EAD sous stress
    - Axe Y : LGD moyenne pondérée EAD sous stress
    - Taille bulle : EAD sectorielle totale
    - Couleur : perte de queue sectorielle

    Ce graphique illustre la double transmission géopolitique
    sur les deux paramètres de crédit simultanément.
    """
    diag_path = TABLES / "sector_diagnostics_at_design_point.csv"
    if not diag_path.exists():
        print("[Sector PD/LGD] sector_diagnostics_at_design_point.csv introuvable.")
        return

    df = pd.read_csv(diag_path)

    # Normalisation taille bulles
    ead_min = df["EAD_sector"].min()
    ead_max = df["EAD_sector"].max()
    sizes   = 300 + 2200 * (df["EAD_sector"] - ead_min) / max(ead_max - ead_min, 1e-12)

    colors  = [_sector_color(s) for s in df["sector"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    for _, row in df.iterrows():
        sz  = float(300 + 2200 * (row["EAD_sector"] - ead_min) / max(ead_max - ead_min, 1e-12))
        col = _sector_color(row["sector"])
        ax.scatter([row["PD_sector_stress"] * 100],
                   [row["LGD_sector_stress"] * 100],
                   s=sz, color=col, alpha=0.80,
                   edgecolors="white", linewidths=1.5, zorder=5)

    # Annotations secteurs
    for _, row in df.iterrows():
        ax.annotate(
            _sector_label(row["sector"]),
            xy=(row["PD_sector_stress"] * 100, row["LGD_sector_stress"] * 100),
            xytext=(6, 5), textcoords="offset points",
            fontsize=11, fontweight="bold",
            color=_sector_color(row["sector"]),
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec=_sector_color(row["sector"]), alpha=0.85)
        )

    # Ligne de tendance (régression linéaire indicative)
    pd_vals  = df["PD_sector_stress"].values * 100
    lgd_vals = df["LGD_sector_stress"].values * 100
    if len(pd_vals) > 2:
        z    = np.polyfit(pd_vals, lgd_vals, 1)
        p    = np.poly1d(z)
        x_lr = np.linspace(pd_vals.min(), pd_vals.max(), 100)
        ax.plot(x_lr, p(x_lr),
                linestyle="--", color="#9E9E9E",
                linewidth=1.5, alpha=0.7, label="Tendance linéaire")

    ax.set_xlabel("$PD_{\\rm stress}$ pondérée EAD par secteur  [%]",
                  fontsize=13, labelpad=10)
    ax.set_ylabel("$LGD_{\\rm stress}$ pondérée EAD par secteur  [%]",
                  fontsize=13, labelpad=10)
    ax.set_title(
        "Double transmission géopolitique : $PD_{\\rm stress}$ vs $LGD_{\\rm stress}$ par secteur\n"
        "Taille bulle ∝ EAD sectorielle — Design point $s^*$",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle=":", alpha=0.5, color="#ccc")

    # Légende taille bulle
    for ead_val, sz_val, lbl in [
        (ead_min, 300,  f"EAD min ({ead_min:.1f})"),
        ((ead_min + ead_max) / 2, 1400, f"EAD médian ({(ead_min+ead_max)/2:.1f})"),
        (ead_max, 2500, f"EAD max ({ead_max:.1f})"),
    ]:
        ax.scatter([], [], s=sz_val, c="#BDBDBD",
                   alpha=0.7, edgecolors="white", label=lbl)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9,
              edgecolor="#999", fancybox=True, title="EAD sectorielle",
              title_fontsize=10)

    out_path = PLOTS / "sector_pd_lgd.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Sector PD/LGD] Sauvegardée : {out_path}")


if __name__ == "__main__":
    plot_sector_losses()
    plot_sector_pd_lgd()
