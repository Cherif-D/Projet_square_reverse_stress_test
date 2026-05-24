import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import expit, logit
from scipy.stats import norm, chi2

# ============================================================
# CONFIGURATION
# ============================================================
INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
TABLES = OUTPUTS / "tables"
PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

# Facteur géopolitique (Axe X)
G_COL = "shock_GPRD" 

# ============================================================
# 1. RECONSTRUCTION DU MODELE
# ============================================================
def load_data():
    print("Chargement des données...")
    try:
        exposures = pd.read_csv(INPUTS / "exposures.csv")
        capital = pd.read_csv(INPUTS / "capital.csv").iloc[0]
        sector_params = pd.read_csv(INPUTS / "sector_params.csv")
        
        # --- CORRECTION CRITIQUE ICI ---
        # 1. On charge d'abord Sigma pour connaitre les vraies variables (8)
        sigma = pd.read_csv(TABLES / "Sigma.csv", index_col=0)
        
        # 2. On charge le Design Point en ignorant l'en-tête potentielle
        # On lit tout, puis on ne garde que les lignes qui existent dans Sigma
        dp_raw = pd.read_csv(TABLES / "design_point.csv", index_col=0, header=None)
        
        # On filtre : on ne garde que les index qui sont des colonnes de Sigma
        # (Ça élimine la ligne ",0" ou "nan")
        design_point = dp_raw.loc[dp_raw.index.isin(sigma.columns)].squeeze().astype(float)
        
        print(f"Variables alignées : {len(design_point)} trouvées (attendu: {len(sigma.columns)})")
        
        return exposures, capital, sector_params, design_point, sigma
        
    except FileNotFoundError as e:
        print(f"ERREUR CRITIQUE : Fichier manquant ({e}). Avez-vous lancé MVP.py ?")
        exit()

# Fonctions utilitaires (identiques MVP)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def smooth_unit_interval(val, center=0.5, scale=0.15):
    return 0.02 + 0.96 * sigmoid((val - center) / scale)

def inverse_smooth_unit_interval(u, center=0.5, scale=0.15):
    u = np.clip(u, 0.020001, 0.979999)
    return center + scale * logit((u - 0.02) / 0.96)

def tail_default_prob(pd, rho, q):
    return norm.cdf((norm.ppf(pd) + np.sqrt(rho) * norm.ppf(q)) / np.sqrt(1 - rho))

# Calcul du Ratio de Capital
def compute_ratio_for_scenario(s_dict, exposures, capital, sector_params):
    params = sector_params.set_index("sector")
    
    PD0 = exposures["PD0"].values
    LGD0 = exposures["LGD0"].values
    EAD = exposures["EAD"].values
    rho = exposures["rho"].values
    alpha_rwa = exposures["alpha_rwa"].values
    sectors = exposures["sector"].values
    
    delta_g = params.loc[sectors, "delta_g"].values
    eta_g = params.loc[sectors, "eta_g"].values
    
    g_val = s_dict.get(G_COL, 0.0)
    
    z_pd = logit(np.clip(PD0, 1e-12, 1-1e-12)) + delta_g * g_val
    lgd_anchor = inverse_smooth_unit_interval(LGD0)
    z_lgd = lgd_anchor + eta_g * g_val
    
    for col, val in s_dict.items():
        if col == G_COL: continue
        b_col = f"b_{col}"
        c_col = f"c_{col}"
        
        if b_col in params.columns:
            beta = np.nan_to_num(params.loc[sectors, b_col].values)
            z_pd += beta * val
            
        if c_col in params.columns:
            gamma = np.nan_to_num(params.loc[sectors, c_col].values)
            z_lgd += gamma * val

    PD_stress = expit(z_pd)
    LGD_stress = smooth_unit_interval(z_lgd)
    
    tail_pd = tail_default_prob(PD_stress, rho, capital["q"])
    Lq_stress = np.sum(EAD * LGD_stress * tail_pd)
    
    # Lq_baseline est lu directement depuis capital.csv (champ Lq0_abs
    # rempli par Simulations/simulate_capital.py). Le pipeline principal
    # recalcule cette valeur depuis les colonnes baseline ; ici on s'appuie
    # sur la valeur figée dans le CSV pour rester compatible legacy.
    Lq_baseline = capital["Lq0_abs"]
    delta_Lq = Lq_stress - Lq_baseline
    CET1 = capital["CET1_0"] - delta_Lq + capital["delta_non_credit"]
    
    RWA = capital["RWA_0"] + np.sum(alpha_rwa * (PD_stress - PD0))
    RWA = max(RWA, 1e-6)
    
    return CET1 / RWA

# ============================================================
# 2. GENERATION DU GRAPHIQUE
# ============================================================
def plot_frontier():
    exposures, capital, sector_params, design_point, sigma = load_data()
    
    # Choix des Axes
    # Axe X : Géopolitique
    # Axe Y : Le facteur macro le plus impactant
    macro_drivers = design_point.drop(G_COL).abs().sort_values(ascending=False)
    X_driver = G_COL
    Y_driver = macro_drivers.index[0]
    
    print(f"Génération du graphique : Axe X={X_driver} / Axe Y={Y_driver}")
    
    # Grille
    x_center, y_center = design_point[X_driver], design_point[Y_driver]
    # On zoome un peu plus si les chocs sont petits
    span = max(4.0, abs(x_center)*3) 
    
    xx = np.linspace(-1, x_center + span, 100)
    yy = np.linspace(y_center - span, y_center + span, 100)
    X_grid, Y_grid = np.meshgrid(xx, yy)
    
    Z_ratio = np.zeros_like(X_grid)
    Z_dist = np.zeros_like(X_grid)
    
    inv_sigma = np.linalg.inv(sigma.values)
    
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            s_vec = design_point.copy()
            s_vec[X_driver] = X_grid[i, j]
            s_vec[Y_driver] = Y_grid[i, j]
            
            Z_ratio[i, j] = compute_ratio_for_scenario(s_vec, exposures, capital, sector_params)
            
            s_vals = s_vec.values
            Z_dist[i, j] = s_vals @ inv_sigma @ s_vals

    # TRACÉ
    plt.figure(figsize=(10, 8))
    
    threshold = capital["R_omega"]
    
    # 1. Zone de Mort
    plt.contourf(X_grid, Y_grid, Z_ratio, levels=[-np.inf, threshold], colors=['#ffebee'], alpha=0.5)
    
    # 2. Frontière (Ligne Rouge)
    cs_frontier = plt.contour(X_grid, Y_grid, Z_ratio, levels=[threshold], colors='red', linewidths=2.5)
    plt.clabel(cs_frontier, fmt='Frontière', inline=True, fontsize=10)
    
    # 3. Ellipses de probabilité
    d2_design = design_point.values @ inv_sigma @ design_point.values
    levels_dist = sorted([1.0, 4.0, 9.0, d2_design]) 
    plt.contour(X_grid, Y_grid, Z_dist, levels=levels_dist, colors='blue', linestyles='dashed', linewidths=0.8, alpha=0.6)
    
    # 4. Points
    plt.scatter([x_center], [y_center], color='red', s=150, zorder=10, marker='*', label="Design Point (Choc Fatal)")
    plt.scatter([0], [0], color='green', s=80, zorder=10, label="Baseline (Calme)")
    
    plt.title(f"Frontière de Rupture : {X_driver} vs {Y_driver}", fontsize=14, fontweight='bold')
    plt.xlabel(f"Choc Géopolitique ({X_driver}) [Z-score]", fontsize=11)
    plt.ylabel(f"Choc Macro ({Y_driver}) [Z-score]", fontsize=11)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    out_file = PLOTS / "frontier_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Graphique terminé : {out_file}")

if __name__ == "__main__":
    plot_frontier()