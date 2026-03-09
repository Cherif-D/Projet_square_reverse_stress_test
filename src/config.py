from __future__ import annotations

# ============================================================
# CONFIGURATION GLOBALE — REVERSE STRESS TEST GEOPOLITIQUE
# ============================================================
#
# Ce module regroupe toutes les constantes et hypothèses de mise en œuvre.
# Il correspond à la section 0 de MVP.py.
#
# USE_LEDOIT_WOLF :
#   on stabilise Sigma avec Ledoit-Wolf, utile si certaines
#   variables sont corrélées ou si l'échantillon n'est pas énorme.
#
# ENFORCE_STRESS_ONLY_FLOOR :
#   si True, on force PD_stress >= PD0 et LGD_stress >= LGD0.
#   Pour l'instant, on le laisse à False pour ne pas rigidifier
#   inutilement l'optimisation.
#
# G_COL :
#   nom du facteur géopolitique g dans la table scénario standardisée.
# ============================================================

import numpy as np

SEED = 42
RNG  = np.random.default_rng(SEED)

USE_LEDOIT_WOLF         = True
ENFORCE_STRESS_ONLY_FLOOR = False
G_COL                   = "shock_GPRD"

# ------------------------------------------------------------
# Paramètres des ensembles plausibles
# ------------------------------------------------------------
ETA_LOCAL     = 1.00       # rayon local en y-space : ||y - y*||^2 <= ETA_LOCAL
PHI_NEAR      = 1.00       # near-optimal : d²(s) <= d²(s*) + PHI_NEAR
POOL_SIZE     = 4000
SHORTLIST_SIZE = 8

# ------------------------------------------------------------
# Transformations macro
# ------------------------------------------------------------
#
# On part de macro.csv (données tuteur harmonisées).
# On construit des chocs sur les variables principales.
#
# Convention :
# - diff     : pour indices / taux / séries déjà compatibles
# - logdiff  : pour variables de niveau strictement positives
# ------------------------------------------------------------
TRANSFORMS = {
    "GPRD"   : "diff",
    "gdp"    : "logdiff",
    "vix"    : "diff",
    "sp500"  : "logdiff",
    "wti"    : "logdiff",
    "t10Y2Y" : "diff",
    "unrate" : "diff",
    "epu"    : "diff",
}
