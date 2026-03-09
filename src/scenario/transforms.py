from __future__ import annotations

# ============================================================
# TRANSFORMATIONS DES SERIES MACRO
# ============================================================
#
# Ce module contient les transformations appliquées aux variables
# de macro.csv pour construire les chocs standardisés.
#
# Convention :
# - diff     : pour indices / taux / séries déjà compatibles
# - logdiff  : pour variables de niveau strictement positives
# ============================================================

import numpy as np
import pandas as pd


def safe_log_diff(s: pd.Series) -> pd.Series:
    """
    Log-diff robuste :
    - conversion numérique
    - mise à NaN des valeurs non positives
    - puis différence logarithmique
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)
    return np.log(s).diff()


def safe_diff(s: pd.Series) -> pd.Series:
    """
    Différence simple robuste :
    - conversion numérique
    - puis différence première
    """
    s = pd.to_numeric(s, errors="coerce")
    return s.diff()
