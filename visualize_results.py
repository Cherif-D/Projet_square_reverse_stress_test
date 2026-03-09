import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
OUTPUTS = Path("outputs")
TABLES = OUTPUTS / "tables"
PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(exist_ok=True)

# 1. Graphique du Scénario (Bar Chart des Chocs)
def plot_scenario():
    try:
        # Charger le Design Point (chocs standardisés en z-score)
        df = pd.read_csv(TABLES / "design_point.csv", header=None, index_col=0)
        df.columns = ["Choc (Ecart-type)"]
        
        # Trier par amplitude
        df["abs_val"] = df["Choc (Ecart-type)"].abs()
        df = df.sort_values("abs_val", ascending=True)
        
        plt.figure(figsize=(10, 6))
        # Couleurs : Rouge si négatif (baisse PIB), Bleu si positif (Hausse Pétrole/GPRD)
        colors = ['red' if x < 0 else 'blue' for x in df["Choc (Ecart-type)"]]
        
        bars = plt.barh(df.index, df["Choc (Ecart-type)"], color=colors, alpha=0.7)
        
        plt.axvline(0, color='black', linewidth=0.8)
        plt.title("Profil du Scénario de Rupture (Design Point)\n(Unité : Écarts-types par rapport à la baseline)")
        plt.xlabel("Intensité du choc (z-score)")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Ajouter les valeurs au bout des barres
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else width
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                     va='center', ha='left' if width > 0 else 'right', fontsize=9)

        plt.tight_layout()
        plt.savefig(PLOTS / "scenario_profile.png", dpi=300)
        print(f"Graphique généré : {PLOTS / 'scenario_profile.png'}")
        
    except Exception as e:
        print(f"Erreur graphe scénario : {e}")

# 2. Graphique des Pertes par Secteur
def plot_sector_impact():
    try:
        df = pd.read_csv(TABLES / "sector_diagnostics_at_design_point.csv")
        
        # On trie par perte totale
        df = df.sort_values("LossQ_sector", ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="LossQ_sector", y="sector", palette="viridis")
        
        plt.title("Répartition des Pertes par Secteur (Design Point)")
        plt.xlabel("Montant de la Perte de Queue (Lq)")
        plt.ylabel("Secteur")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(PLOTS / "sector_impact.png", dpi=300)
        print(f"Graphique généré : {PLOTS / 'sector_impact.png'}")
        
    except Exception as e:
        print(f"Erreur graphe secteur : {e}")

# 3. Comparaison Avant/Après (Waterfall simplifié)
def plot_capital_bridge():
    # Nécessite de lire le JSON pour avoir les agrégats, ou on le fait "en dur" pour l'exemple
    pass 

if __name__ == "__main__":
    print("--- Génération des graphiques ---")
    plot_scenario()
    plot_sector_impact()
    print("Terminé.")