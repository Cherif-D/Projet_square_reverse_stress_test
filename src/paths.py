from __future__ import annotations

# ============================================================
# CHEMINS DU DEPOT
# ============================================================
#
# Ce module centralise tous les chemins utilisés dans l'implémentation.
#
# On détecte automatiquement la racine du dépôt.
# src/paths.py est dans src/, donc la racine est deux niveaux au-dessus.
# ============================================================

from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]   # src/ -> racine du dépôt

DATA    = ROOT / "data"
RAW     = DATA / "raw"
INTERIM = DATA / "interim"
INPUTS  = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
TABLES  = OUTPUTS / "tables"
REPORTS = OUTPUTS / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
