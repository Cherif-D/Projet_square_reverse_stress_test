from __future__ import annotations

# ============================================================
# SIMULATION SOLIDE DE inputs/sector_params.csv
# ============================================================
#
# BUT
# ---
# Produire les coefficients sectoriels reliant le scénario s=(g,x)
# aux PD/LGD stressées, conformément aux eq. (7)-(8) et (9)-(10)
# du papier (Hurlin, Lajaunie, Pull, 7 janvier 2026).
#
# FORME UTILISEE DANS LE SCRIPT PRINCIPAL
# ----------------------------------------
# PD — eq. (7) du papier :
#   logit(PD_i(g,x)) = logit(PD0_i) + beta_k(i)^T * x + delta_k(i) * g
# On note ici :
#   delta_g   = delta_k(i) dans l'eq. (7) : sensibilité géopolitique de la PD
#   b_shock_* = composantes de beta_k(i) : sensibilités macro de la PD
#
# LGD — eq. (10) du papier :
#   LGD_i(g,x) = LGD0_i + gamma_k(i)^T * x + eta_k(i) * g
# On note ici :
#   eta_g     = eta_k(i) dans l'eq. (10) : sensibilité géopolitique de la LGD
#   c_shock_* = composantes de gamma_k(i) : sensibilités macro de la LGD
#
# Les chocs x sont standardisés dans le script principal (z-scores),
# donc chaque coefficient s'interprète comme l'effet pour 1 écart-type.
#
# LIEN AVEC L'ANNEXE A.2 DU PAPIER
# ----------------------------------
# La version sectorielle (Annexe A.2, eq. 57-58) est directement :
#   logit(PD_k(g,x)) = logit(PD0_k) + b_k^T * x + d_k * g
#   LGD_k(g,x)       = LGD0_k      + c_k^T * x + e_k * g
# Nos coefficients sont les b_k, d_k, c_k, e_k de l'Annexe A.2.
#
# ============================================================
#
# SOURCES ET JUSTIFICATION DES HIERARCHIES SECTORIELLES
# =====================================================
#
# (A) SENSIBILITE GEOPOLITIQUE (delta_g, eta_g) :
#
#     Source principale : ECB Working Paper No. 2897,
#     Lo Duca, Moccero, Parlapiano (2025) :
#     "Geopolitical risk and the economy: a sectoral perspective."
#
#     Ce papier établit empiriquement que :
#     - Energy, Transport et Manufacturing sont les secteurs les plus
#       sensibles au risque géopolitique (commerce mondial, supply-chain,
#       sanctions, prix commodities).
#     - Utilities est le plus résilient (domestique, régulé).
#     - Consumer et Real Estate ont une sensibilité intermédiaire
#       (confiance, conditions de crédit).
#
#     Le papier de Hurlin et al. (Section 2.3) cite Flament et al. (2026)
#     qui incorporent le risque géopolitique dans des stress tests
#     prospectifs via un cadre VAR–Merton, confirmant cette hiérarchie.
#
# (B) CONFIRMATION EBA 2025 STRESS TEST (résultats août 2025) :
#
#     Source : EBA, "2025 EU-wide Stress Test – Results", août 2025.
#     URL : eba.europa.eu/.../2025%20EU-wide%20stress%20test%20-%20Results.pdf
#
#     Éléments pertinents :
#     - Le scénario adverse 2025 suppose une aggravation des tensions
#       géopolitiques avec un choc GDP cumulé de -6.3% sur 3 ans.
#     - L'EBA note (Figure 29) que "banks which make a greater use of
#       models for PD and LGD parameters tend to project higher loss
#       rates for the vulnerable sectors."
#     - Les "secteurs vulnérables" sont ceux avec les plus grandes
#       déviations de GVA cumulée sous adverse = industrie, énergie, commerce.
#     - Capital depletion moyenne = 370 bps sous adverse.
#
# (C) SCENARIO ESRB 2025 :
#
#     Source : ESRB, "Macro-financial scenario for the 2025 EU-wide
#     banking sector stress test" (janvier 2025).
#     Le scénario inclut un breakdown GVA par 16 secteurs d'activité
#     économique, confirmant l'hétérogénéité sectorielle.
#
# (D) CROSS-CHECK S&P 2024 :
#
#     Source : S&P Global Ratings, "2024 Annual Global Corporate Default
#     and Rating Transition Study" (mars 2025).
#     Les taux de défaut sectoriels 2024 confirment la hiérarchie :
#     - Leisure/media, consumer/service : taux les plus hauts
#     - Energy, transport : en baisse mais LT avg élevées
#     - Utilities : historiquement très bas
#
# IMPORTANT
# ---------
# Les coefficients ci-dessous ne sont PAS des coefficients BCE/EBA
# "publiés" prêts à l'emploi pour nos variables exactes.
# Il n'existe pas de publication officielle donnant directement
# "b_shock_gdp = -0.32 pour Consumer" sous cette forme.
#
# Ce que nous avons fait :
# - structure et hiérarchie = empirically grounded (sources A-D ci-dessus)
# - valeurs numériques = calibrations stylisées cohérentes avec ces hiérarchies
# - dans un exercice de production, on calibrerait ces coefficients par
#   régression sur panel (logit PD ~ macro drivers) avec données internes banque
#   (cf. Section 6.2 du papier, étape 2 de la feuille de route).
#
# MODIFICATIONS PAR RAPPORT A LA VERSION INITIALE
# ===============================================
# 1) Nous gardons exactement les commentaires détaillés de la version initiale.
# 2) Nous ajoutons des colonnes de traçabilité dans le CSV final :
#       param_origin, calibration_status, note
#    pour rappeler explicitement que les coefficients restent stylisés.
# 3) Les valeurs numériques du cœur des coefficients ne changent pas ;
#    nous ajoutons surtout de la transparence documentaire.
# ============================================================

from pathlib import Path
import pandas as pd


# ============================================================
# 0. CONFIGURATION
# ============================================================

ROOT = Path(".")
INPUTS = ROOT / "inputs"
INPUTS.mkdir(parents=True, exist_ok=True)

SECTORS = [
    "Energy", "Transport", "Manufacturing", "Consumer",
    "Tech", "Utilities", "Defense", "RealEstate",
]

# Variables macro-financières du vecteur x.
# Correspondent aux composantes de x dans l'eq. (5) du papier :
#   s = (g, x)^T ∈ S, x ∈ R^{d-1}
# Ici d-1 = 7 variables macro :
#   shock_gdp    : croissance du PIB réel (activité économique)
#   shock_vix    : volatilité implicite (stress financier)
#   shock_sp500  : rendement actions (soutien marché)
#   shock_wti    : prix du pétrole (commodities)
#   shock_t10Y2Y : pente des taux (10Y-2Y, conditions de financement)
#   shock_unrate : taux de chômage (marché du travail)
#   shock_epu    : Economic Policy Uncertainty (incertitude politique)
X_COLS = [
    "shock_gdp", "shock_vix", "shock_sp500", "shock_wti",
    "shock_t10Y2Y", "shock_unrate", "shock_epu",
]


# ============================================================
# 1. CONSTRUCTION DES PARAMETRES
# ============================================================

rows = []

for sector in SECTORS:
    row = {"sector": sector}

    # ========================================================
    # TRACEABILITE AJOUTEE DANS LA VERSION FINALE
    # ========================================================
    # Nous ajoutons ces colonnes pour documenter explicitement
    # le caractère stylisé des paramètres dans le CSV final.
    row["param_origin"] = "stylized"
    row["calibration_status"] = "not_empirically_estimated"
    row["note"] = "sector hierarchy economically grounded, numeric values stylized"

    # ========================================================
    # A. EFFETS GEOPOLITIQUES DIRECTS (delta_g, eta_g)
    # ========================================================
    #
    # delta_g = delta_k(i) dans l'eq. (7) du papier.
    #   Interprétation (eq. 7) : "des valeurs positives de delta_k(i)
    #   reflètent l'idée qu'une hausse du risque géopolitique augmente
    #   la probabilité de défaut dans les secteurs davantage exposés au
    #   commerce mondial, aux perturbations des chaînes d'approvisionnement
    #   ou aux sanctions." (Section 4.1 du papier)
    #
    # eta_g = eta_k(i) dans l'eq. (10) du papier.
    #   Interprétation (eq. 10) : "le terme eta_k(i)*G capte l'effet du
    #   risque géopolitique sur les recouvrements via des canaux tels que
    #   des saisies d'actifs, des perturbations dans l'exécution
    #   transfrontalière, des pertes de valeur des collatéraux ou des
    #   restrictions sur la vente d'actifs gagés." (Section 4.1 du papier)
    #
    # Hiérarchie sectorielle (source : ECB WP 2897, Lo Duca et al. 2025) :
    #
    # TIER 1 (delta_g=0.65, eta_g=0.10) : Energy, Transport, Manufacturing
    #   Les plus exposés au géopolitique. Le papier ECB WP 2897 montre que
    #   ces secteurs subissent les impacts les plus forts des chocs
    #   géopolitiques via le commerce mondial et les chaînes d'approvisionnement.
    #   Le stress test EBA 2025 confirme que les "secteurs vulnérables" au
    #   sens de la GVA cumulée sous adverse sont liés à l'industrie et
    #   l'énergie. Le delta_g=0.65 signifie qu'un choc géopolitique de
    #   1 écart-type augmente le log-odds de défaut de 0.65 pour ces secteurs.
    #
    # TIER 2 (delta_g=0.40, eta_g=0.08) : Consumer, RealEstate
    #   Sensibilité intermédiaire. Le géopolitique affecte ces secteurs
    #   indirectement via la confiance des consommateurs et les conditions
    #   de crédit (Nguyen and Thuy, 2023 ; cité dans Section 2.3 du papier).
    #
    # TIER 3 (delta_g=0.25-0.28, eta_g=0.05) : Defense, Tech
    #   Defense : pas neutre au géopolitique, mais contrats étatiques
    #   = contreparties souveraines. On reste modéré faute d'estimation
    #   publique robuste dans le format de nos variables.
    #   Tech : export controls et tensions Chine/US, mais grandes techs
    #   EU relativement isolées. ECB WP 2897 montre une sensibilité modérée.
    #
    # TIER 4 (delta_g=0.18, eta_g=0.04) : Utilities
    #   Le plus résilient. Secteur régulé, domestique, cash-flows contractuels.
    #   ECB WP 2897 confirme que utilities est le secteur le moins impacté.
    #
    # ========================================================
    if sector in ["Energy", "Transport", "Manufacturing"]:
        row["delta_g"] = 0.65    # TIER 1 : forte sensibilité géopolitique
        row["eta_g"] = 0.10      # TIER 1 : fort impact sur recouvrements
    elif sector in ["Consumer", "RealEstate"]:
        row["delta_g"] = 0.40    # TIER 2 : sensibilité intermédiaire
        row["eta_g"] = 0.08      # TIER 2
    elif sector == "Defense":
        row["delta_g"] = 0.28    # TIER 3 : modéré, contrats étatiques
        row["eta_g"] = 0.05      # TIER 3
    elif sector == "Tech":
        row["delta_g"] = 0.25    # TIER 3 : export controls, mais tech EU solide
        row["eta_g"] = 0.05      # TIER 3
    else:  # Utilities
        row["delta_g"] = 0.18    # TIER 4 : le plus résilient
        row["eta_g"] = 0.04      # TIER 4

    # ========================================================
    # B. EFFETS MACRO SUR LA PD (composantes de beta_k dans eq. 7)
    # ========================================================
    #
    # Chaque coefficient s'interprète pour 1 écart-type du choc macro
    # (les x sont standardisés dans le script principal).
    #
    # Logique économique de chaque variable :
    # - GDP ↑ → meilleure activité → PD ↓ (coefficient négatif)
    # - VIX ↑ → stress financier → PD ↑ (coefficient positif)
    # - SP500 ↑ → soutien des marchés → PD ↓ (coefficient négatif)
    # - WTI ↑ → favorable à Energy, défavorable à Transport
    # - t10Y2Y ↑ (pentification) → effet sur RealEstate (financement)
    # - unrate ↑ → chômage hausse → PD ↑
    # - epu ↑ → incertitude politique → PD ↑
    #
    # Hiérarchie sectorielle pour chaque variable :
    # Les secteurs cycliques (Consumer, Manufacturing, Transport,
    # RealEstate) sont les plus sensibles au GDP et au chômage.
    # Cela est cohérent avec le stress test EBA 2025 où ces secteurs
    # montrent les loss rates les plus élevées sous adverse.
    #
    # ========================================================

    # ── GDP ──
    # Secteurs les plus cycliques → coefficient le plus fort (négatif).
    # EBA 2025 stress test : le choc GDP cumulé de -6.3% impacte
    # surtout les secteurs liés à la demande intérieure et au commerce.
    if sector in ["Consumer", "Manufacturing", "Transport", "RealEstate"]:
        row["b_shock_gdp"] = -0.32   # forte sensibilité cyclique
    elif sector in ["Tech", "Energy"]:
        row["b_shock_gdp"] = -0.18   # sensibilité modérée
    elif sector == "Utilities":
        row["b_shock_gdp"] = -0.12   # faible (cash-flows régulés)
    else:  # Defense
        row["b_shock_gdp"] = -0.16   # modéré (contrats étatiques, peu cyclique)

    # ── VIX ──
    # Secteurs les plus exposés au stress financier (financement de marché,
    # conditions de crédit) → coefficient le plus fort (positif).
    # Le papier (Section 2.3) cite Nguyen and Thuy (2023) : le risque
    # géopolitique augmente les coûts d'emprunt et resserre les conditions.
    if sector in ["Transport", "Consumer", "Tech", "RealEstate"]:
        row["b_shock_vix"] = 0.26    # forte sensibilité financière
    elif sector in ["Manufacturing", "Energy"]:
        row["b_shock_vix"] = 0.16    # modéré
    elif sector == "Utilities":
        row["b_shock_vix"] = 0.10    # faible (faible dépendance marché)
    else:  # Defense
        row["b_shock_vix"] = 0.12    # modéré

    # ── SP500 ──
    # Soutien des marchés actions → PD ↓. Plus fort pour Tech et Consumer
    # (dépendance aux valorisations et au sentiment de marché).
    if sector in ["Tech", "Consumer"]:
        row["b_shock_sp500"] = -0.12
    elif sector in ["RealEstate", "Transport"]:
        row["b_shock_sp500"] = -0.08
    else:
        row["b_shock_sp500"] = -0.05

    # ── WTI (prix du pétrole) ──
    # Effet sectoriel spécifique :
    # - Energy : WTI ↑ → revenus ↑ → PD ↓ (coefficient négatif)
    # - Transport : WTI ↑ → coûts ↑ → PD ↑ (coefficient positif, le plus fort)
    # - Manufacturing : WTI ↑ → coûts inputs ↑ → PD ↑ légèrement
    # - Autres : effet faible
    if sector == "Energy":
        row["b_shock_wti"] = -0.10   # bénéficiaire (revenues pétrolières)
    elif sector == "Transport":
        row["b_shock_wti"] = 0.18    # défavorable (coûts carburant)
    elif sector == "Manufacturing":
        row["b_shock_wti"] = 0.05    # léger (coûts inputs)
    else:
        row["b_shock_wti"] = 0.02    # marginal

    # ── t10Y2Y (pente des taux) ──
    # Pentification favorable en général (soutien activité),
    # mais surtout visible pour RealEstate (financement long terme,
    # coût des crédits immobiliers).
    if sector == "RealEstate":
        row["b_shock_t10Y2Y"] = -0.14  # fort : coût du financement CRE
    elif sector in ["Utilities", "Consumer"]:
        row["b_shock_t10Y2Y"] = -0.05  # modéré
    else:
        row["b_shock_t10Y2Y"] = -0.03  # faible

    # ── Unemployment ──
    # Chômage ↑ → PD ↑. Plus fort pour Consumer, RealEstate, Transport
    # (secteurs dépendant de la demande ménages et de l'emploi).
    # EBA 2025 stress test : le chômage augmente de 5.8pp sous adverse.
    if sector in ["Consumer", "RealEstate", "Transport"]:
        row["b_shock_unrate"] = 0.28   # forte sensibilité emploi
    elif sector in ["Manufacturing", "Tech"]:
        row["b_shock_unrate"] = 0.16   # modéré
    elif sector == "Utilities":
        row["b_shock_unrate"] = 0.08   # faible (demande inélastique)
    else:  # Energy / Defense
        row["b_shock_unrate"] = 0.12   # modéré

    # ── EPU (Economic Policy Uncertainty) ──
    # Incertitude politique ↑ → PD ↑. Plus fort pour les secteurs
    # les plus exposés aux décisions politiques (réglementation énergie,
    # politique commerciale manufacturing, contrats defense).
    if sector in ["Energy", "Manufacturing", "Transport", "Defense"]:
        row["b_shock_epu"] = 0.14    # forte sensibilité policy
    elif sector in ["Consumer", "RealEstate"]:
        row["b_shock_epu"] = 0.10    # modéré
    else:  # Tech, Utilities
        row["b_shock_epu"] = 0.08    # faible

    # ========================================================
    # C. EFFETS MACRO SUR LA LGD (composantes de gamma_k dans eq. 10)
    # ========================================================
    #
    # Le papier (Section 4.1, eq. 10) note que la LGD est affectée
    # par les chocs macro-financiers via "des canaux tels que des
    # saisies d'actifs, des perturbations dans l'exécution transfrontalière,
    # des pertes de valeur des collatéraux".
    #
    # En pratique, la LGD réagit souvent MOINS VITE que la PD aux
    # chocs macro. Les coefficients c_shock_* sont donc typiquement
    # 3x à 5x plus faibles que les b_shock_* correspondants.
    #
    # ========================================================

    # ── GDP → LGD ──
    # GDP ↑ → valeurs de récupération ↑ → LGD ↓.
    if sector in ["Consumer", "Manufacturing", "Transport", "RealEstate"]:
        row["c_shock_gdp"] = -0.05   # cycliques : LGD réactive
    else:
        row["c_shock_gdp"] = -0.02   # autres : LGD peu réactive

    # ── VIX → LGD ──
    # Stress financier → valeurs de liquidation ↓ → LGD ↑.
    if sector in ["Transport", "Consumer", "Tech", "RealEstate"]:
        row["c_shock_vix"] = 0.03
    else:
        row["c_shock_vix"] = 0.02

    # ── SP500 → LGD ──
    # Faible effet direct sur les recouvrements.
    row["c_shock_sp500"] = -0.01

    # ── WTI → LGD ──
    # Energy : WTI ↑ → valeur des actifs gagés ↑ → LGD ↓.
    # Transport : WTI ↑ → dégradation rentabilité → valeur d'entreprise ↓ → LGD ↑.
    if sector == "Energy":
        row["c_shock_wti"] = -0.01
    elif sector == "Transport":
        row["c_shock_wti"] = 0.03
    elif sector == "Manufacturing":
        row["c_shock_wti"] = 0.01
    else:
        row["c_shock_wti"] = 0.00

    # ── t10Y2Y → LGD ──
    # Effet uniquement sur RealEstate (valeur collatéral immobilier
    # sensible aux conditions de financement long terme).
    if sector == "RealEstate":
        row["c_shock_t10Y2Y"] = 0.01
    else:
        row["c_shock_t10Y2Y"] = 0.00

    # ── Unemployment → LGD ──
    # Chômage ↑ → conditions de liquidation dégradées → LGD ↑.
    if sector in ["Consumer", "RealEstate", "Transport"]:
        row["c_shock_unrate"] = 0.03
    else:
        row["c_shock_unrate"] = 0.02

    # ── EPU → LGD ──
    # Incertitude → processus de recouvrement allongés → LGD ↑.
    if sector in ["Energy", "Manufacturing", "Defense", "Transport"]:
        row["c_shock_epu"] = 0.02
    else:
        row["c_shock_epu"] = 0.01

    rows.append(row)

params_df = pd.DataFrame(rows)

out_path = INPUTS / "sector_params.csv"
params_df.to_csv(out_path, index=False)

print("Fichier créé :", out_path)
print(params_df)
print("\n" + "="*60)
print("SOURCES DOCUMENTEES POUR LA HIERARCHIE SECTORIELLE :")
print("="*60)
print("  [1] ECB WP 2897 (Lo Duca, Moccero, Parlapiano, 2025) :")
print("      → hiérarchie de sensibilité géopolitique par secteur")
print("  [2] EBA 2025 EU-wide Stress Test Results (août 2025) :")
print("      → secteurs vulnérables, loss rates, capital depletion 370bps")
print("  [3] ESRB macro-financial scenario 2025 (janvier 2025) :")
print("      → breakdown GVA par 16 secteurs, tensions géopolitiques")
print("  [4] S&P 2024 Annual Default Study (mars 2025) :")
print("      → taux de défaut sectoriels (cross-check qualitatif)")
print("  [5] Hurlin, Lajaunie, Pull (2026) :")
print("      → eq. (7), (8), (9), (10) ; Section 4.1 interprétation des coefficients")
print("  [6] Nguyen and Thuy (2023), cité Section 2.3 du papier :")
print("      → géopolitique augmente coûts d'emprunt et conditions non-prix")
print("\nNOTE AJOUTEE : le CSV contient aussi param_origin, calibration_status et note pour tracer explicitement le caractère stylisé des paramètres.")
