from __future__ import annotations

# ============================================================
# SIMULATION SOLIDE DE inputs/exposures.csv
# ============================================================
#
# BUT
# ---
# Créer une table d'expositions compatible avec la voie exposure-level
# du papier "Reverse Stress Testing Geopolitical Risk in Corporate
# Credit Portfolios" (Hurlin, Lajaunie, Pull, 7 janvier 2026).
#
# Colonnes produites :
#   id, sector, EAD, PD0, LGD0, rho, M, alpha_rwa, K0_proxy, RW0_exact_proxy
#
# LIEN AVEC LE PAPIER
# -------------------
# - EAD_i          → eq. (1) : EAD = sum_i EAD_i
# - PD0_i, LGD0_i  → Section 3.1 : paramètres baseline non stressés
# - rho_i          → eq. (11) : corrélation d'actif dans le modèle
#                     à facteur latent gaussien (ASRF/IRB, Vasicek 2002, Gordy 2003)
# - M_i            → eq. (18)-(19) : maturité effective dans la formule
#                     de poids de risque h(PD,LGD,M)
# - alpha_rwa_i    → eq. (21) : approximation linéaire des RWA
#                     RWA(g,x) = RWA0 + sum_i alpha_i*(PD_i(g,x) - PD0_i)
# - RW0_exact_proxy→ proxy de densité RWA baseline obtenu à partir de
#                     K0_proxy = LGD*[Phi((Phi^{-1}(PD)+sqrt(rho)*Phi^{-1}(q))/sqrt(1-rho)) - PD]*gamma(M)
#                     puis RW0_exact_proxy = 12.5 * K0_proxy
#
# LOGIQUE METHODOLOGIQUE
# ----------------------
# Comme on n'a pas de vraies données internes banque, on SIMULE
# les entrées manquantes de manière économiquement cohérente.
# Le dictionnaire SECTOR_PROFILE est ancré sur des sources publiques.
#
# ============================================================
#
# SOURCES DES PARAMETRES pd_mean / lgd_mean
# ==========================================
#
# (A) ANCRE DOCUMENTAIRE PRINCIPALE : HSBC Continental Europe,
#     "Capital and Risk Management Pillar 3 Disclosures
#     at 31 December 2024" (publié mars 2025).
#     URL : https://www.hsbc.com/-/files/hsbc/investors/hsbc-results/2024/annual/pdfs/hsbc-continental-europe/250313-hsbc-continental-europe-pillar-3-at-2024-dec-31.pdf
#     Ce document sert ici d'ancre de structure et d'ordre de grandeur.
#     Il ne publie pas directement des PD/LGD sectorielles prêtes à l'emploi
#     pour notre modèle.
#     Nous l'utilisons surtout pour :
#     - la ventilation par industrie des expositions, of which defaulted,
#       impairment et provisions ;
#     - des ordres de grandeur corporate IRB servant de point d'ancrage global.
#     Les valeurs sectorielles retenues ci-dessous restent donc des proxys
#     calibrés, pas des extractions littérales d'une table HSBC.
#
# (B) CROSS-CHECK MARCHE : S&P Global Ratings,
#     "2024 Annual Global Corporate Default and Rating Transition Study"
#     (publiée mars 2025, données à fin 2024).
#     URL : https://www.spglobal.com/ratings/en/regulatory/article/default-transition-and-recovery-2024-annual-global-corporate-default-and-rating-transition-study-s13452126
#     Utilisé ici comme repère qualitatif de hiérarchie sectorielle :
#     - secteurs cycliques ou à forte intensité de financement plus fragiles ;
#     - utilities historiquement plus résilient ;
#     - real estate et consumer sensibles au cycle ;
#     - energy et transport soumis à des effets de marché plus volatils.
#     NOTE : univers global, plus risqué qu'un portefeuille banque européen
#     IG/crossover.
#
# (C) CONTEXTE EBA :
#     - EBA Risk Dashboard, Credit Risk Parameters Annex, Q3 2024 :
#       ordres de grandeur corporate IRB et dispersion des paramètres entre banques.
#       URL : https://www.eba.europa.eu/sites/default/files/2024-12/116c9bd7-60d2-4f9a-93f9-bb15f964a6d4/Credit%20Risk%20parameters%20annex%20-%20Q3%202024.pdf
#     - EBA, "Report on the 2023 Credit Risk Benchmarking Exercise"
#       (publié avril 2024) :
#       rôle de la collatéralisation et de l'hétérogénéité de portefeuille dans les LGD.
#     - Basel Foundation IRB (CRR Art. 161 ; BCBS RBC20, 15 Dec 2019) :
#       LGD prescrite = 45% senior unsecured corporate.
#       LGD prescrite = 75% subordinated.
#     Ces références servent à cadrer les niveaux plausibles, sans fournir
#     un mapping sectoriel direct sur nos 8 secteurs.
#
# (D) CORRELATION D'ACTIF : formule Bâle IRB, CRE31 (BCBS, 1 Jan 2023) :
#       rho_irb(PD) = 0.12*R + 0.24*(1-R), R = (1-exp(-50*PD))/(1-exp(-50))
#     Documenté en cross-check pour chaque secteur.
#     L'ajustement sectoriel autour de l'ancre IRB reste stylisé.
#
# (E) QUANTILE q = 0.999 :
#     99.9e percentile du cadre IRB Bâle (BCBS CRE31, 2023 ; Gordy 2003).
#     Utilisé dans l'eq. (16) du papier pour Lq(g,x).
#
# CONVENTIONS DE CALIBRATION
# ==========================
# 1) Les pd_mean / lgd_mean ci-dessous sont des valeurs de calibration
#    retenues pour produire un portefeuille plausible et lisible.
# 2) RW0_exact_proxy est exprimé en unité RWA via 12.5 * K0_proxy.
# 3) rho_i est ancré sur la formule IRB corporate, puis ajusté par secteur.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# 0. CONFIGURATION
# ============================================================

ROOT = Path(".")
INPUTS = ROOT / "inputs"
INPUTS.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# n=500 expositions : portefeuille assez granulaire pour l'approximation
# ASRF (Gordy 2003, Vasicek 2002), tout en restant rapide à optimiser.
N_EXPOSURES = 500

# EAD totale normalisée à 100. Correspond à l'eq. (1) du papier.
TOTAL_EAD = 100.0

# q = 0.999 : 99.9e percentile IRB (BCBS CRE31, 2023). Eq. (16) du papier.
Q = 0.999

# 8 secteurs couvrant les grandes classes d'un portefeuille corporate.
# Hiérarchie stylisée, ancrée sur l'ECB FSR (May 2024),
# l'EBA 2025 stress test et la S&P 2024 default study.
SECTORS = [
    "Energy", "Transport", "Manufacturing", "Consumer",
    "Tech", "Utilities", "Defense", "RealEstate",
]

# Poids sectoriels : portefeuille corporate européen diversifié.
# Manufacturing et RealEstate surpondérés : choix stylisé cohérent avec
# un portefeuille corporate européen diversifié ; HSBC P3 sert ici surtout
# d'ancre qualitative de structure, pas de preuve sectorielle stricte.
SECTOR_WEIGHTS = {
    "Energy": 0.12, "Transport": 0.10, "Manufacturing": 0.18, "Consumer": 0.15,
    "Tech": 0.12, "Utilities": 0.08, "Defense": 0.07, "RealEstate": 0.18,
}


def basel_irb_rho(pd: float) -> float:
    """Corrélation d'actif Bâle IRB (CRE31, BCBS 2023, §31.43).
    Corporate hors PME : rho = 0.12*R + 0.24*(1-R)
    avec R = (1-exp(-50*PD))/(1-exp(-50))."""
    e50 = np.exp(-50.0)
    ratio = (1.0 - np.exp(-50.0 * pd)) / (1.0 - e50)
    return 0.12 * ratio + 0.24 * (1.0 - ratio)


# ============================================================
# SECTOR_PROFILE — chaque valeur est documentée
# ============================================================
# [proxy calibré]   = valeur retenue à partir d'ordres de grandeur documentaires
#                     et d'un recalage de portefeuille ; pas une statistique
#                     sectorielle publiée telle quelle
# [ancre corporate] = valeur corporate de référence utilisée faute de lecture
#                     sectorielle suffisamment robuste
# [stylisé]         = hypothèse de modélisation
# ============================================================

SECTOR_PROFILE = {

    # ── Energy ──────────────────────────────────────────────
    # pd_mean = 0.445% [proxy calibré] : niveau bas retenu pour un
    #   portefeuille banque IG/crossover.
    # lgd_mean = 24.0% [proxy calibré] : niveau bas cohérent avec la présence
    #   d'actifs physiques et de recouvrements potentiellement meilleurs.
    # rho_shift = 0.000 [stylisé] : secteur laissé au voisinage de l'ancre IRB.
    # m_mean = 2.8 [stylisé] : maturité typique financements énergie mid-term.
    # alpha_mult = 1.15 [stylisé] : légère surpondération RWA.
    "Energy": {
        "pd_mean": 0.00445, "lgd_mean": 0.24,
        "rho_shift": 0.000, "m_mean": 2.8, "alpha_mult": 1.15,
    },

    # ── Transport ───────────────────────────────────────────
    # pd_mean = 0.116% [proxy calibré] : valeur volontairement basse à la baseline,
    #   distincte des taux de défaut de marché plus volatils.
    # lgd_mean = 58.0% [proxy calibré] : niveau élevé cohérent avec des expositions
    #   potentiellement moins collatéralisées.
    # rho_shift = -0.005 [stylisé] : léger décalage sous l'ancre IRB.
    # m_mean = 2.7 [stylisé] : maturité moyenne transport.
    # alpha_mult = 1.10 [stylisé] : légère surpondération RWA.
    "Transport": {
        "pd_mean": 0.00116, "lgd_mean": 0.58,
        "rho_shift": -0.005, "m_mean": 2.7, "alpha_mult": 1.10,
    },

    # ── Manufacturing ───────────────────────────────────────
    # pd_mean = 0.187% [proxy calibré] : niveau bas à modéré pour un secteur
    #   industriel diversifié.
    # lgd_mean = 46.0% [proxy calibré] : niveau intermédiaire cohérent avec un
    #   collatéral partiel et des structures de financement mixtes.
    # rho_shift = -0.010 [stylisé] : manufacturing légèrement sous l'ancre IRB.
    # m_mean = 2.6 [stylisé] : secteur de référence.
    # alpha_mult = 1.00 [stylisé] : secteur pivot du portefeuille.
    "Manufacturing": {
        "pd_mean": 0.00187, "lgd_mean": 0.46,
        "rho_shift": -0.010, "m_mean": 2.6, "alpha_mult": 1.00,
    },

    # ── Consumer ────────────────────────────────────────────
    # pd_mean = 0.421% [proxy calibré] : secteur placé au-dessus de la moyenne
    #   du portefeuille sur la composante défaut.
    # lgd_mean = 43.0% [proxy calibré] : niveau cohérent avec un collatéral
    #   physique souvent limité.
    # rho_shift = -0.015 [stylisé] : léger décalage sous l'ancre IRB.
    # m_mean = 2.4 [stylisé] : maturités courtes (revolving, trade finance).
    # alpha_mult = 1.00 [stylisé].
    "Consumer": {
        "pd_mean": 0.00421, "lgd_mean": 0.43,
        "rho_shift": -0.015, "m_mean": 2.4, "alpha_mult": 1.00,
    },

    # ── Tech ────────────────────────────────────────────────
    # pd_mean = 0.175% [proxy calibré] : niveau bas à modéré retenu pour un
    #   secteur tech de qualité relativement correcte.
    # lgd_mean = 52.0% [proxy calibré] : niveau plus élevé en raison d'un
    #   collatéral souvent moins tangible.
    # rho_shift = -0.020 [stylisé] : part idiosyncratique supposée plus forte.
    # m_mean = 2.3 [stylisé] : maturités courtes.
    # alpha_mult = 0.80 [stylisé] : sensibilité RWA un peu plus faible.
    "Tech": {
        "pd_mean": 0.00175, "lgd_mean": 0.52,
        "rho_shift": -0.020, "m_mean": 2.3, "alpha_mult": 0.80,
    },

    # ── Utilities ───────────────────────────────────────────
    # pd_mean = 0.150% [proxy calibré] : secteur positionné parmi les plus
    #   résilients du portefeuille.
    # lgd_mean = 45.0% [proxy calibré] : niveau intermédiaire retenu pour un
    #   secteur régulé, avec actifs réels mais recouvrements hétérogènes.
    # rho_shift = -0.025 [stylisé] : utilities placé sous l'ancre IRB.
    # m_mean = 3.0 [stylisé] : maturités longues (infra).
    # alpha_mult = 0.70 [stylisé] : sensibilité RWA plus faible.
    "Utilities": {
        "pd_mean": 0.00150, "lgd_mean": 0.45,
        "rho_shift": -0.025, "m_mean": 3.0, "alpha_mult": 0.70,
    },

    # ── Defense ─────────────────────────────────────────────
    # pd_mean = 0.430% [ancre corporate] : faute de lecture sectorielle robuste,
    #   on retient une ancre corporate prudente.
    # lgd_mean = 46.0% [ancre corporate] : niveau intermédiaire retenu pour un
    #   secteur à contrats souvent longs et recouvrements spécifiques.
    # rho_shift = -0.015 [stylisé] : léger décalage sous l'ancre IRB.
    # m_mean = 2.8 [stylisé] : contrats étatiques long terme.
    # alpha_mult = 0.90 [stylisé].
    "Defense": {
        "pd_mean": 0.00430, "lgd_mean": 0.46,
        "rho_shift": -0.015, "m_mean": 2.8, "alpha_mult": 0.90,
    },

    # ── RealEstate ──────────────────────────────────────────
    # pd_mean = 1.005% [proxy calibré] : secteur placé en haut de la distribution
    #   baseline, cohérent avec une exposition plus cyclique au financement.
    # lgd_mean = 42.0% [proxy calibré] : niveau intermédiaire cohérent avec un
    #   collatéral immobilier imparfaitement protecteur.
    # rho_shift = 0.000 [stylisé] : secteur maintenu au voisinage de l'ancre IRB.
    # m_mean = 3.2 [stylisé] : maturités longues prêts CRE.
    # alpha_mult = 1.20 [stylisé] : surpondération RWA marquée.
    "RealEstate": {
        "pd_mean": 0.01005, "lgd_mean": 0.42,
        "rho_shift": 0.000, "m_mean": 3.2, "alpha_mult": 1.20,
    },
}


# ============================================================
# 1. NOMBRE D'EXPOSITIONS PAR SECTEUR
# ============================================================
# On fait correspondre le nombre de lignes à la structure sectorielle
# du portefeuille. Eq. (1) du papier : EAD = sum_i EAD_i.
# ============================================================

sector_counts = {
    sec: int(round(N_EXPOSURES * w))
    for sec, w in SECTOR_WEIGHTS.items()
}
diff = N_EXPOSURES - sum(sector_counts.values())
if diff != 0:
    first_sec = next(iter(sector_counts))
    sector_counts[first_sec] += diff


# ============================================================
# 2. CREATION DE LA BASE BRUTE
# ============================================================
# EAD brutes : lognormale(0, 0.9).
# Pourquoi lognormale ? Support positif, quelques grosses lignes
# (concentration), beaucoup de petites (granularité).
# sigma=0.9 → ratio P95/P50 ≈ 4.4, réaliste pour un portefeuille corporate.
# ============================================================

rows = []
exposure_id = 1
for sector, n_sec in sector_counts.items():
    p = SECTOR_PROFILE[sector]
    raw_ead = RNG.lognormal(mean=0.0, sigma=0.9, size=n_sec)
    for raw in raw_ead:
        rows.append({
            "id": exposure_id, "sector": sector, "ead_raw": raw,
            "pd_mean_sector": p["pd_mean"], "lgd_mean_sector": p["lgd_mean"],
            "rho_shift_sector": p["rho_shift"], "m_mean_sector": p["m_mean"],
            "alpha_mult_sector": p["alpha_mult"],
        })
        exposure_id += 1

df = pd.DataFrame(rows)
# Normalisation → eq. (1) : EAD = sum_i EAD_i = TOTAL_EAD = 100.
df["EAD"] = TOTAL_EAD * df["ead_raw"] / df["ead_raw"].sum()


# ============================================================
# 3. SIMULATION DE PD0_i
# ============================================================
# Le papier part de PD0_i (Section 3.1) puis construit PD_i(g,x)
# via eq. (7)-(8) dans le script principal.
# Bruit MULTIPLICATIF lognormal : préserve PD>0, crée de l'hétérogénéité.
# sigma=0.25 → coefficient de variation ≈ 25% autour de la moyenne sectorielle.
# Bornes : lower=0.0005 (plancher), upper=0.20 (plafond).
# ============================================================

pd_noise = RNG.lognormal(mean=0.0, sigma=0.25, size=len(df))
df["PD0"] = df["pd_mean_sector"] * pd_noise
df["PD0"] = df["PD0"].clip(lower=0.0005, upper=0.20)


# ============================================================
# 4. SIMULATION DE LGD0_i
# ============================================================
# Le papier part de LGD0_i (Section 3.1) puis construit LGD_i(g,x)
# via eq. (9)-(10) dans le script principal.
# Bruit ADDITIF normal : LGD ∈ [0,1], additif plus contrôlable.
# scale=0.035 → ±3.5pp d'écart-type, modéré.
# Bornes : lower=0.10 (coûts minimaux de recouvrement), upper=0.90.
# Cette dispersion garde une LGD moyenne portefeuille assez stable.
# ============================================================

lgd_noise = RNG.normal(loc=0.0, scale=0.035, size=len(df))
df["LGD0"] = df["lgd_mean_sector"] + lgd_noise
df["LGD0"] = df["LGD0"].clip(lower=0.10, upper=0.90)


# ============================================================
# 5. SIMULATION DE rho_i
# ============================================================
# rho_i = corrélation d'actif, eq. (11) du papier :
#   Y_i = sqrt(rho_i)*Z + sqrt(1-rho_i)*epsilon_i
# Bruit additif normal, scale=0.010 (±1pp). Bornes : [0.08, 0.30].
# La formule Bâle IRB (CRE31) donne max ≈ 0.24 ; nos bornes restent plus larges
# pour flexibilité (le modèle de corrélation stylisé peut différer du réglementaire).
# rho est ancré sur la formule IRB corporate, puis ajusté
# par un rho_shift sectoriel et un bruit additif.
# ============================================================

rho_irb_base = df["pd_mean_sector"].apply(basel_irb_rho).to_numpy()
rho_noise = RNG.normal(loc=0.0, scale=0.010, size=len(df))
df["rho"] = rho_irb_base + df["rho_shift_sector"].to_numpy() + rho_noise
df["rho"] = df["rho"].clip(lower=0.08, upper=0.30)


# ============================================================
# 6. SIMULATION DE M_i (maturité effective)
# ============================================================
# M_i intervient dans eq. (18)-(19) du papier via h(PD,LGD,M).
# Bruit normal, scale=0.4 (±0.4 an). Bornes : [1.0, 5.0] ans.
# 5 ans = plafond standard dans les modèles IRB (BCBS CRE31).
# ============================================================

m_noise = RNG.normal(loc=0.0, scale=0.4, size=len(df))
df["M"] = df["m_mean_sector"] + m_noise
df["M"] = df["M"].clip(lower=1.0, upper=5.0)


# ============================================================
# 7. PROXY DE RW0_i A LA BASELINE
# ============================================================
# Eq. (19) du papier :
#   h(PD,LGD,M) = LGD * [Phi((Phi^{-1}(PD) + sqrt(rho)*Phi^{-1}(q))/sqrt(1-rho)) - PD] * gamma(M)
# gamma(M) = ajustement de maturité (BCBS CRE31, §31.44).
# On simplifie gamma(M) par une fonction linéaire centrée sur M=2.5, bornée [0.80,1.25].
#
# On explicite d'abord K0_proxy comme proxy de charge
# en capital baseline, puis on convertit en RWA via le facteur 12.5.
# Cela permet de garder une lecture claire :
#   K0_proxy        ≈ charge en capital
#   RW0_exact_proxy ≈ densité RWA
# ============================================================

def maturity_adjustment_proxy(M: np.ndarray) -> np.ndarray:
    """Proxy simplifié de gamma(M). Source : approx. BCBS CRE31 §31.44."""
    return np.clip(1.0 + 0.08 * (M - 2.5), 0.80, 1.25)

# tail_PD0 = Phi((Phi^{-1}(PD0) + sqrt(rho)*Phi^{-1}(q))/sqrt(1-rho))
# C'est le terme de queue ASRF, eq. (16) du papier.
tail_pd0 = norm.cdf(
    (norm.ppf(df["PD0"].to_numpy()) + np.sqrt(df["rho"].to_numpy()) * norm.ppf(Q))
    / np.sqrt(1.0 - df["rho"].to_numpy())
)
df["tail_PD0"] = tail_pd0

# K0_i ≈ LGD0_i * (tail_PD0_i - PD0_i) * gamma(M_i)
df["K0_proxy"] = (
    df["LGD0"].to_numpy()
    * (df["tail_PD0"].to_numpy() - df["PD0"].to_numpy())
    * maturity_adjustment_proxy(df["M"].to_numpy())
)
df["K0_proxy"] = df["K0_proxy"].clip(lower=0.0)

# RW0_i = 12.5 * K0_i
# Facteur réglementaire de conversion charge en capital → RWA.
df["RW0_exact_proxy"] = 12.5 * df["K0_proxy"]


# ============================================================
# 8. CONSTRUCTION DE alpha_rwa_i
# ============================================================
# Eq. (21) du papier : RWA(g,x) = RWA0 + sum_i alpha_i*(PD_i(g,x) - PD0_i).
# alpha_i = pente locale des RWA par rapport à PD_i.
# On calibre : alpha_i ∝ EAD_i * RW0_i * alpha_mult_k / max(PD0_i, 0.005).
# Le facteur 0.35 est un coefficient d'échelle [stylisé].
# La division par PD0 (avec plancher 0.005) capte la non-linéarité IRB
# (pente dRW/dPD plus forte pour PD basses).
# Ce coefficient est choisi pour garder alpha_rwa dans un ordre de
# grandeur raisonnable.
# ============================================================

pd_floor = np.maximum(df["PD0"].to_numpy(), 0.005)
df["alpha_rwa"] = (
    0.35 * df["alpha_mult_sector"].to_numpy() * df["EAD"].to_numpy()
    * df["RW0_exact_proxy"].to_numpy() / pd_floor
)
df["alpha_rwa"] = df["alpha_rwa"].clip(lower=1e-8)


# ============================================================
# 9. SORTIE FINALE
# ============================================================

final_cols = [
    "id", "sector", "EAD", "PD0", "LGD0", "rho", "M",
    "alpha_rwa", "K0_proxy", "RW0_exact_proxy",
]
df_final = df[final_cols].copy()
df_final = df_final.sort_values(["sector", "id"]).reset_index(drop=True)

out_path = INPUTS / "exposures.csv"
df_final.to_csv(out_path, index=False)

print("Fichier créé :", out_path)
print("Nombre d'expositions :", len(df_final))
print("EAD total :", round(df_final["EAD"].sum(), 6))
print("\nCross-check rho Bâle IRB CRE31 (ancre) :")
for sec, p in SECTOR_PROFILE.items():
    rho_irb = basel_irb_rho(p["pd_mean"])
    rho_target = rho_irb + p["rho_shift"]
    print(f"  {sec:15s}  pd={p['pd_mean']:.5f}  rho_cible~{rho_target:.4f}  rho_IRB={rho_irb:.4f}")
print("\nRésumé par secteur :")
print(
    df_final.groupby("sector").agg(
        n=("id", "count"), EAD_total=("EAD", "sum"),
        PD0_mean=("PD0", "mean"), LGD0_mean=("LGD0", "mean"),
        rho_mean=("rho", "mean"), M_mean=("M", "mean"),
        K0_mean=("K0_proxy", "mean"), RW0_mean=("RW0_exact_proxy", "mean"),
    ).round(4)
)

ead_weighted_pd = np.average(df_final["PD0"], weights=df_final["EAD"])
ead_weighted_lgd = np.average(df_final["LGD0"], weights=df_final["EAD"])
ead_weighted_rw0 = np.average(df_final["RW0_exact_proxy"], weights=df_final["EAD"])
print("\nDiagnostics portefeuille pondérés EAD :")
print(f"  PD moyenne pondérée   : {ead_weighted_pd:.6f}")
print(f"  LGD moyenne pondérée  : {ead_weighted_lgd:.6f}")
print(f"  RW0 moyen pondéré     : {ead_weighted_rw0:.6f}")
