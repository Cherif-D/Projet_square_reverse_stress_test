from __future__ import annotations

# ============================================================
# SIMULATION SOLIDE DE inputs/capital.csv
# ============================================================
#
# BUT
# ---
# Construire un capital.csv CONNECTE à exposures.csv, en dérivant
# RWA_0 du portefeuille simulé (pas fixé "dans le vide").
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 7 janvier 2026)
# -------------------------------------------------------------
# - CET1_0, RWA_0 → Section 3.1, eq. (2) : R0 = CET1_0 / RWA_0
# - R_omega       → eq. (4) : R* = R0*(1-Delta), Delta=0.03, soit 300 bps
#                   Conforme à l'annonce BCE du 12 décembre 2025 (ECB, 2025b) :
#                   "identify the most relevant geopolitical risk events that
#                   could lead to at least a 300-basis point depletion in
#                   [their] Common Equity Tier 1 (CET1) capital."
# - Lq0_abs       → eq. (16) évaluée en s=0 (baseline) :
#                   Lq(0) = sum_i EAD_i * LGD0_i * Phi((Phi^{-1}(PD0_i) + sqrt(rho_i)*Phi^{-1}(q))/sqrt(1-rho_i))
# - q = 0.999     → 99.9e percentile IRB (BCBS CRE31, 2023 ; Gordy, 2003)
#
# CORRECTION DE BASELINE (préparée ici, implémentée dans le script principal)
# ---------------------------------------------------------------------------
# Le script principal devra utiliser :
#   delta_Lq(s) = Lq(s) - Lq(0)
#   CET1(s) = CET1_0 - delta_Lq(s) + delta_non_credit    → eq. (17) du papier
#   RWA(s)  = RWA_0  + sum_i alpha_i*(PD_i(s) - PD0_i)   → eq. (21) du papier
#   R(s)    = CET1(s) / RWA(s)                            → eq. (22) du papier
# Cela garantit R(0) = R0 exactement.
#
# CONTEXTE PRUDENTIEL
# -------------------
# - R0 = 14% : cohérent avec le ratio CET1 moyen des banques SSM à fin 2024
#   (~15.9%, source : EBA, "2025 EU-wide Stress Test Results", août 2025,
#   Section "Starting point", et ECB, "2025 stress test of euro area banks",
#   Chart 1). On prend 14% pour représenter une banque individuelle
#   légèrement en-dessous de la moyenne du système.
# - R_omega = 11% : R0 - 300 bps = 14% - 3% = 11%.
#   La condition de rupture est R(s) <= R_omega (eq. 23 du papier).
# - CET1 capital = composante de plus haute qualité du capital réglementaire,
#   composée principalement de fonds propres ordinaires et de bénéfices mis
#   en réserve (note 4 du papier).
#
# MODIFICATIONS PAR RAPPORT A LA VERSION INITIALE
# ===============================================
# 1) Nous conservons exactement les commentaires détaillés de la version
#    initiale.
# 2) Nous n'estimons plus RW0_proxy localement à partir de PD/LGD/M dans ce
#    script : nous lisons désormais RW0_exact_proxy déjà construit dans
#    simulate_exposures.py après conversion correcte en RWA (facteur 12.5).
# 3) Cela évite d'avoir deux définitions concurrentes de RW0 et garantit que
#    capital.csv est parfaitement cohérent avec exposures.csv.
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

EXPOSURES_PATH = INPUTS / "exposures.csv"

# ── R0 = 14% ──
# Source contextuelle : EBA 2025 stress test starting point = 15.9% (agrégé SSM).
# On choisit 14% pour une banque individuelle légèrement en-dessous.
R0_TARGET = 0.14

# ── Déplétion 300 bps ──
# Source : ECB press release 12 December 2025 (ECB, 2025b) :
# "identify the most relevant geopolitical risk events that could lead to
# at least a 300-basis point depletion in CET1 capital ratio."
#
# On retient la dépletion en termes ABSOLUS (300 bps = 0.03 en points de ratio).
# R_omega = R0 - 0.03 = 0.14 - 0.03 = 0.11
#
# Note : l'eq. (4) du papier peut s'écrire R* = R0 - Delta avec Delta = 0.03,
# ou de façon équivalente R* = R0*(1 - Delta/R0). On n'utilise pas la forme
# multiplicative R* = R0*(1-Delta) car elle ne correspond pas aux 300 bps absolus
# de l'ECB lorsque Delta = 0.03.
DEPLETION = 0.03
R_OMEGA = R0_TARGET - DEPLETION   # = 0.11  (dépletion absolue de 300 bps)

# ── q = 0.999 ──
# 99.9e percentile IRB (BCBS CRE31, 2023). Eq. (16) du papier.
Q = 0.999

# ── delta_non_credit = 0.0 ──
# Eq. (17) du papier : delta_P&L_non_credit(g,x).
# Dans une analyse de portefeuille de crédit pur, on fixe ce terme à 0.
# Le papier note qu'on pourrait y inclure trading losses, fee income, opex.
DELTA_NON_CREDIT = 0.0


# ============================================================
# 1. CHARGER exposures.csv
# ============================================================

if not EXPOSURES_PATH.exists():
    raise FileNotFoundError(
        f"{EXPOSURES_PATH} introuvable. Lance d'abord simulate_exposures.py"
    )

exp_df = pd.read_csv(EXPOSURES_PATH)

required_cols = {"EAD", "PD0", "LGD0", "rho", "M", "RW0_exact_proxy"}
missing = required_cols - set(exp_df.columns)
if missing:
    raise ValueError(f"Colonnes manquantes dans exposures.csv : {missing}")


# ============================================================
# 2. RECONSTRUIRE UN PROXY DE RWA_0
# ============================================================
# Eq. (20) du papier : RWA(g,x) = sum_i EAD_i * RW_i(g,x).
# Au baseline (s=0) : RWA_0 = sum_i EAD_i * RW0_i.
# On utilise désormais directement RW0_exact_proxy produit dans
# simulate_exposures.py.
# AJOUT : comme RW0_exact_proxy a déjà été converti en densité RWA,
# on évite ici toute re-définition locale du proxy.
# ============================================================

RWA_0 = float(np.sum(exp_df["EAD"].to_numpy() * exp_df["RW0_exact_proxy"].to_numpy()))


# ============================================================
# 3. CALCULER Lq(0) ABSOLU
# ============================================================
# Eq. (16) du papier évaluée en s=0 :
#   Lq(0) = sum_i EAD_i * LGD0_i * Phi((Phi^{-1}(PD0_i) + sqrt(rho_i)*Phi^{-1}(q))/sqrt(1-rho_i))
# Ce n'est PAS la correction finale (qui est delta_Lq = Lq(s) - Lq(0)).
# On le stocke pour diagnostic et pour préparer le script principal.
# ============================================================

tail_pd0 = norm.cdf(
    (norm.ppf(exp_df["PD0"].to_numpy()) + np.sqrt(exp_df["rho"].to_numpy()) * norm.ppf(Q))
    / np.sqrt(1.0 - exp_df["rho"].to_numpy())
)

LQ0_ABS = float(
    np.sum(
        exp_df["EAD"].to_numpy()
        * exp_df["LGD0"].to_numpy()
        * tail_pd0
    )
)


# ============================================================
# 4. CALCULER CET1_0
# ============================================================
# Eq. (2) du papier : R0 = CET1_0 / RWA_0.
# On fixe R0 = R0_TARGET et on en déduit CET1_0 = R0 * RWA_0.
# Cela garantit la cohérence portefeuille ↔ capital.
# ============================================================

CET1_0 = R0_TARGET * RWA_0


# ============================================================
# 5. SORTIE FINALE
# ============================================================
# On ajoute des colonnes de diagnostic :
# - Lq0_abs : perte de queue baseline (diagnostic)
# - baseline_convention : rappel que le script principal doit utiliser delta_Lq
# ============================================================

capital_df = pd.DataFrame([{
    "CET1_0": CET1_0,
    "RWA_0": RWA_0,
    "R0": R0_TARGET,
    "R_omega": R_OMEGA,
    "q": Q,
    "delta_non_credit": DELTA_NON_CREDIT,
    "Lq0_abs": LQ0_ABS,
    "baseline_convention": "use_delta_Lq_in_main_script",
}])

out_path = INPUTS / "capital.csv"
capital_df.to_csv(out_path, index=False)

print("Fichier créé :", out_path)
print(capital_df)
print("\nDiagnostic cohérence portefeuille/capital :")
print(f"  RWA_0 dérivé du portefeuille  : {RWA_0:.6f}  [eq. (20) baseline]")
print(f"  CET1_0 déduit de R0 cible     : {CET1_0:.6f}  [eq. (2) : R0*RWA_0]")
print(f"  Lq(0) absolu (diagnostic)     : {LQ0_ABS:.6f}  [eq. (16) en s=0]")
print(f"  R0 (ratio initial)            : {R0_TARGET:.4f}  [contexte EBA ~15.9%]")
print(f"  R_omega (seuil de rupture)    : {R_OMEGA:.4f}  [ECB 2025b, -300bps]")
print(f"  q (quantile IRB)              : {Q}      [BCBS CRE31, Gordy 2003]")
print(f"  delta_non_credit              : {DELTA_NON_CREDIT}        [eq. (17), crédit pur]")
print("\nIMPORTANT : le script principal devra utiliser DeltaLq = Lq(s) - Lq(0).")
