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
#   id, sector, EAD, PD0, LGD0, rho, M, alpha_rwa, RW0_exact_proxy
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
# - RW0_exact_proxy→ eq. (19) : proxy du poids de risque IRB baseline
#                     h(PD,LGD,M) = LGD*[Phi((Phi^{-1}(PD)+sqrt(rho)*Phi^{-1}(q))/sqrt(1-rho)) - PD]*gamma(M)
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
# (A) ANCRE PRIMAIRE : HSBC Continental Europe,
#     "Capital and Risk Management Pillar 3 Disclosures
#     at 31 December 2024" (publié mars 2025).
#     URL : hsbc.com/-/files/hsbc/investors/hsbc-results/2024/annual/pdfs/
#           hsbc-continental-europe/250313-hsbc-continental-europe-pillar-3-at-2024-dec-31.pdf
#     Tables utilisées :
#     - Table 29 (CQ5), page 34 :
#       encours non financiers par industrie (NACE), non-performing, impairment.
#       On en dérive des proxies de PD/LGD sectoriels via le ratio
#       impairment/encours et non-performing/encours.
#     - Table 40 (CCR4), page 59 :
#       FIRB - Corporates subtotal :
#         PD moyenne pondérée = 0.43%
#         LGD moyenne pondérée = 46.3%
#       Cette ancre corporate sert de recalage global.
#     IMPORTANT : les chiffres ne sont PAS des PD/LGD sectorielles
#     publiées directement. Ce sont des proxies que nous avons
#     reconstruits puis recalés sur l'ancre corporate.
#
# (B) CROSS-CHECK MARCHE : S&P Global Ratings,
#     "2024 Annual Global Corporate Default and Rating Transition Study"
#     (publiée mars 2025, données à fin 2024).
#     URL : maalot.co.il/Publications/FTS20250331162124.pdf
#     Taux de défaut sectoriels 2024 et moyennes long-terme (LT) :
#     - Energy & Natural Resources : 2024 = 1.1%, LT avg = 3.0%
#     - Transportation             : 2024 = 2.0%, LT avg ≈ 2.5%
#     - Consumer/Service           : 2024 > LT avg (≈2.5%)
#     - High Technology            : 2024 > LT avg (≈2.0%)
#     - Real Estate                : 2024 > LT avg (≈2.5%)
#     - Utilities                  : historiquement très bas (≈0.5-1%)
#     NOTE : univers global (dont beaucoup de HY US). Un portefeuille
#     banque européen IG/crossover a des PD bien plus basses.
#
# (C) CONTEXTE EBA :
#     - EBA Risk Dashboard, Credit Risk Parameters Annex, Q3 2024 :
#       PD pondérées par exposition, classe "Corporate" IRB ≈ 1.0-2.5%.
#       URL : eba.europa.eu/.../Credit%20Risk%20parameters%20annex%20-%20Q3%202024.pdf
#     - EBA, "Report on the 2023 Credit Risk Benchmarking Exercise"
#       (publié avril 2024), Section 2 :
#       LGD pondérées entre 25% et 42% selon collatéralisation (A-IRB).
#     - Basel Foundation IRB (CRR Art. 161 ; BCBS RBC20, 15 Dec 2019) :
#       LGD prescrite = 45% senior unsecured corporate.
#       LGD prescrite = 75% subordinated.
#
# (D) CORRELATION D'ACTIF : formule Bâle IRB, CRE31 (BCBS, 1 Jan 2023) :
#       rho_irb(PD) = 0.12*R + 0.24*(1-R), R = (1-exp(-50*PD))/(1-exp(-50))
#     Documenté en cross-check pour chaque secteur.
#     Les rho_mean principaux restent stylisés.
#
# (E) QUANTILE q = 0.999 :
#     99.9e percentile du cadre IRB Bâle (BCBS CRE31, 2023 ; Gordy 2003).
#     Utilisé dans l'eq. (16) du papier pour Lq(g,x).
#
# MODIFICATIONS PAR RAPPORT A LA VERSION INITIALE
# ===============================================
# 1) Nous conservons exactement la logique documentaire ci-dessus,
#    mais nous abaissons plusieurs lgd_mean pour obtenir un portefeuille
#    mieux équilibré entre pertes de queue et RWA.
# 2) Nous remplaçons l'ancien RW0_exact_proxy direct par une décomposition
#    explicite en deux étapes :
#       K0_proxy        = charge en capital baseline
#       RW0_exact_proxy = 12.5 * K0_proxy
#    afin de convertir correctement la charge en capital en RWA.
# 3) Nous remplaçons rho_mean par un ancrage IRB + rho_shift sectoriel :
#       rho_i = rho_irb(PD_sector) + rho_shift_sector + bruit
#    ce qui garde le cross-check Bâle tout en permettant une stylisation sectorielle.
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
# Hiérarchie cohérente avec ECB WP 2897 (Lo Duca et al., 2025),
# EBA 2025 stress test, et S&P 2024 default study.
SECTORS = [
    "Energy", "Transport", "Manufacturing", "Consumer",
    "Tech", "Utilities", "Defense", "RealEstate",
]

# Poids sectoriels : portefeuille corporate européen diversifié.
# Manufacturing et RealEstate surpondérés (cf. HSBC P3 Table 29 industry breakdown).
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
# [HSBC P3]  = proxy dérivé de HSBC Continental Europe Pillar 3 (31 Dec 2024)
# [fallback] = ancre corporate HSBC (PD=0.43%, LGD=46.3%) faute de ligne sectorielle
# [stylisé]  = hypothèse de modélisation, pas directement sourcée
#
# AJOUT : nous gardons les mêmes commentaires détaillés que dans la version
# initiale, mais les lgd_mean et la façon de traiter rho évoluent pour coller
# au calibrage final retenu après tests portefeuille/capital.
# ============================================================

SECTOR_PROFILE = {

    # ── Energy ──────────────────────────────────────────────
    # pd_mean = 0.445% [HSBC P3] : Table 29, lignes NACE B+D.
    #   Cross-check S&P 2024 : energy default rate = 1.1%, LT avg = 3.0%.
    #   Notre 0.445% << S&P car on modélise un portefeuille banque IG/crossover.
    #   Cross-check EBA Risk Dashboard Q3 2024 : corporate IRB ≈ 1.0-2.5%.
    # lgd_mean = 24.4% [HSBC P3] : Table 29 impairment/encours.
    #   Cross-check EBA Benchmarking 2023 : 25-42% selon collatéral.
    #   Actifs physiques (champs, raffineries) → bas de fourchette. Cohérent.
    #   Réf. réglementaire : Basel F-IRB = 45% senior unsecured (CRR Art. 161).
    # rho_mean = 0.20 [stylisé] ; Bâle IRB CRE31 à PD=0.445% → 0.2161. Proche.
    # m_mean = 2.8 [stylisé] : maturité typique financements énergie mid-term.
    # alpha_mult = 1.15 [stylisé] : surpondération RWA (expositions energy = RW élevés).
    # MODIFICATION : dans la version finale, nous gardons pd_mean et lgd_mean,
    # mais nous remplaçons rho_mean par rho_shift=0.000 autour de l'ancre IRB.
    "Energy": {
        "pd_mean": 0.00445, "lgd_mean": 0.24,
        "rho_shift": 0.000, "m_mean": 2.8, "alpha_mult": 1.15,
    },

    # ── Transport ───────────────────────────────────────────
    # pd_mean = 0.116% [HSBC P3] : Table 29, NACE H.
    #   Cross-check S&P 2024 : transport default rate = 2.0%.
    #   Écart normal : HSBC = IRB PD TTC portefeuille IG ; S&P = univers global + HY.
    # lgd_mean = 75.0% [HSBC P3, cappé] : très haut, reflète probablement des
    #   expositions peu collatéralisées (leasing aérien, shipping non secured).
    #   Réf. : Basel F-IRB = 75% pour subordinated (CRR Art. 161).
    #   EBA Benchmarking 2023 : 25-42% secured, mais transport non secured >> 42%.
    # rho_mean = 0.18 [stylisé] ; Bâle CRE31 → 0.2332. Hypothèse conservatrice.
    # m_mean = 2.7 [stylisé]. alpha_mult = 1.10 [stylisé].
    # MODIFICATION : nous abaissons lgd_mean de 75% à 58% pour mieux équilibrer
    # pertes de queue et RWA dans le portefeuille simulé final.
    # MODIFICATION : rho_mean devient rho_shift=-0.005 autour de l'ancre IRB.
    "Transport": {
        "pd_mean": 0.00116, "lgd_mean": 0.58,
        "rho_shift": -0.005, "m_mean": 2.7, "alpha_mult": 1.10,
    },

    # ── Manufacturing ───────────────────────────────────────
    # pd_mean = 0.187% [HSBC P3] : Table 29, NACE C.
    #   Cross-check S&P 2024 : forest/building + capital goods ≈ 2.5%.
    # lgd_mean = 56.36% [HSBC P3] : machines/stocks = collatéral partiel → ~56%.
    #   EBA Benchmarking 2023 : 25-42% bien collatéralisé, mais mix secured/unsecured.
    # rho_mean = 0.17 [stylisé] ; Bâle CRE31 → 0.2293.
    # m_mean = 2.6, alpha_mult = 1.00 [stylisé] : secteur de référence.
    # MODIFICATION : nous abaissons légèrement lgd_mean à 46%.
    # MODIFICATION : rho_mean devient rho_shift=-0.010 autour de l'ancre IRB.
    "Manufacturing": {
        "pd_mean": 0.00187, "lgd_mean": 0.46,
        "rho_shift": -0.010, "m_mean": 2.6, "alpha_mult": 1.00,
    },

    # ── Consumer ────────────────────────────────────────────
    # pd_mean = 0.421% [HSBC P3] : Table 29, NACE G.
    #   Cross-check S&P 2024 : consumer/service > LT avg (≈2.5%).
    # lgd_mean = 48.26% [HSBC P3] : peu de collatéral physique (marques, goodwill).
    #   Proche de Basel F-IRB 45%. EBA Benchmarking : fourchette haute unsecured.
    # rho_mean = 0.16 [stylisé] ; Bâle CRE31 → 0.2172.
    # m_mean = 2.4 [stylisé] : maturités courtes (revolving, trade finance).
    # MODIFICATION : nous abaissons légèrement lgd_mean à 43%.
    # MODIFICATION : rho_mean devient rho_shift=-0.015 autour de l'ancre IRB.
    "Consumer": {
        "pd_mean": 0.00421, "lgd_mean": 0.43,
        "rho_shift": -0.015, "m_mean": 2.4, "alpha_mult": 1.00,
    },

    # ── Tech ────────────────────────────────────────────────
    # pd_mean = 0.175% [HSBC P3] : Table 29, NACE J.
    #   Cross-check S&P 2024 : high-tech > LT avg (≈2.0%). Grandes techs EU solides.
    # lgd_mean = 69.01% [HSBC P3] : actifs intangibles (IP, software, brevets),
    #   très peu de collatéral physique → très au-dessus de Basel F-IRB 45%.
    # rho_mean = 0.14 [stylisé] ; Bâle CRE31 → 0.2299.
    #   Bas volontairement : forte composante idiosyncratique (innovation, disruption).
    # m_mean = 2.3 [stylisé] : maturités courtes. alpha_mult = 0.80 [stylisé].
    # MODIFICATION : nous abaissons lgd_mean à 52% et remplaçons rho_mean
    # par rho_shift=-0.020 autour de l'ancre IRB.
    "Tech": {
        "pd_mean": 0.00175, "lgd_mean": 0.52,
        "rho_shift": -0.020, "m_mean": 2.3, "alpha_mult": 0.80,
    },

    # ── Utilities ───────────────────────────────────────────
    # pd_mean = 0.150% [HSBC P3] : Table 29, NACE D.
    #   Cross-check S&P : utilities historiquement ≈ 0.5-1%. Secteur régulé. Cohérent.
    # lgd_mean = 67.57% [HSBC P3] : étonnamment haut. Peut refléter du project finance
    #   non secured. EBA Benchmarking 2023 : utilities collatéralisé ≈ 28%.
    #   On garde le proxy HSBC pour fidélité à la source. Le script principal peut override.
    # rho_mean = 0.12 [stylisé] ; Bâle CRE31 → 0.2313.
    #   Bas : cash-flows régulés/contractuels → faible corrélation macro.
    # m_mean = 3.0 [stylisé] : maturités longues (infra). alpha_mult = 0.70 [stylisé].
    # MODIFICATION : nous abaissons lgd_mean à 45% et remplaçons rho_mean
    # par rho_shift=-0.025 autour de l'ancre IRB.
    "Utilities": {
        "pd_mean": 0.00150, "lgd_mean": 0.45,
        "rho_shift": -0.025, "m_mean": 3.0, "alpha_mult": 0.70,
    },

    # ── Defense ─────────────────────────────────────────────
    # pd_mean = 0.430% [fallback ancre corporate HSBC, Table 40 CCR4].
    #   Pas de ligne NACE robuste pour defense dans HSBC P3.
    #   S&P : pas de catégorie dédiée. Contrats étatiques → défaut rare.
    # lgd_mean = 46.3% [fallback ancre corporate HSBC, Table 40 CCR4].
    #   Contrats gouvernementaux → recouvrement intermédiaire.
    # rho_mean = 0.15 [stylisé] ; Bâle CRE31 → 0.2168.
    # m_mean = 2.8 [stylisé] : contrats étatiques long terme.
    # MODIFICATION : nous gardons lgd_mean=46% et remplaçons rho_mean
    # par rho_shift=-0.015 autour de l'ancre IRB.
    "Defense": {
        "pd_mean": 0.00430, "lgd_mean": 0.46,
        "rho_shift": -0.015, "m_mean": 2.8, "alpha_mult": 0.90,
    },

    # ── RealEstate ──────────────────────────────────────────
    # pd_mean = 1.005% [HSBC P3] : Table 29, NACE L.
    #   Cross-check S&P 2024 : real estate > LT avg (≈2.5%).
    #   Cross-check EBA RAR Nov 2024 : chapitre dédié CRE risks, pressions
    #   persistantes en Europe (correction valorisations, taux élevés, vacance bureaux).
    #   PD la plus haute de notre portefeuille → cohérent avec le contexte.
    # lgd_mean = 39.77% [HSBC P3] : collatéral immobilier physique direct.
    #   EBA Benchmarking 2023 : CRE secured ≈ 25-35%. Basel F-IRB = 35% CRE secured.
    #   Notre 40% = légèrement au-dessus (mix secured/partially secured).
    # rho_mean = 0.19 [stylisé] ; Bâle CRE31 → 0.1926. Très proche.
    # m_mean = 3.2 [stylisé] : maturités longues prêts CRE.
    # alpha_mult = 1.20 [stylisé] : surpondération forte (CRE = secteur le plus
    #   sensible aux RWA dans les stress tests EBA 2025, cf. résultats août 2025).
    # MODIFICATION : nous gardons la structure et remplaçons rho_mean par rho_shift=0.000.
    # MODIFICATION : nous arrondissons lgd_mean à 42% dans le calibrage final.
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
# scale=0.04 → ±4pp d'écart-type, modéré.
# Bornes : lower=0.10 (coûts minimaux de recouvrement), upper=0.90.
#
# MODIFICATION : nous réduisons légèrement la dispersion à 0.035 pour garder
# une LGD moyenne portefeuille plus stable après les ajustements sectoriels.
# ============================================================

lgd_noise = RNG.normal(loc=0.0, scale=0.035, size=len(df))
df["LGD0"] = df["lgd_mean_sector"] + lgd_noise
df["LGD0"] = df["LGD0"].clip(lower=0.10, upper=0.90)


# ============================================================
# 5. SIMULATION DE rho_i
# ============================================================
# rho_i = corrélation d'actif, eq. (11) du papier :
#   Y_i = sqrt(rho_i)*Z + sqrt(1-rho_i)*epsilon_i
# Bruit additif normal, scale=0.015 (±1.5pp). Bornes : [0.05, 0.35].
# La formule Bâle IRB (CRE31) donne max ≈ 0.24 ; nos bornes sont plus larges
# pour flexibilité (le modèle de corrélation stylisé peut différer du réglementaire).
#
# MODIFICATION : nous n'utilisons plus directement rho_mean_sector.
# Nous ancrons désormais rho sur la formule IRB corporate, puis nous ajoutons
# un rho_shift sectoriel et un bruit plus faible (scale=0.010).
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
# MODIFICATION MAJEURE :
# Dans la version finale, on explicite d'abord K0_proxy comme proxy de charge
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
# AJOUT : facteur réglementaire de conversion charge en capital → RWA.
df["RW0_exact_proxy"] = 12.5 * df["K0_proxy"]


# ============================================================
# 8. CONSTRUCTION DE alpha_rwa_i
# ============================================================
# Eq. (21) du papier : RWA(g,x) = RWA0 + sum_i alpha_i*(PD_i(g,x) - PD0_i).
# alpha_i = pente locale des RWA par rapport à PD_i.
# On calibre : alpha_i ∝ EAD_i * RW0_i * alpha_mult_k / max(PD0_i, 0.005).
# Le facteur 0.60 est un coefficient d'échelle [stylisé].
# La division par PD0 (avec plancher 0.005) capte la non-linéarité IRB
# (pente dRW/dPD plus forte pour PD basses).
#
# MODIFICATION : comme RW0 est désormais exprimé en unité RWA via 12.5,
# nous abaissons le coefficient d'échelle de 0.60 à 0.35 pour garder
# alpha_rwa dans un ordre de grandeur raisonnable.
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
