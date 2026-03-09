from __future__ import annotations

# ============================================================
# VERIFICATION DU CONTEXTE DONNEES TUTEUR
# ============================================================
#
# Ce module documente ce qu'on utilise du côté données tuteur.
#
# Ce bloc n'est pas nécessaire mathématiquement, mais il est utile :
# - pour le debug
# - pour garder une trace de la présence de data/raw
# - pour montrer que le script est bien branché sur ton jeu de données réel
# ============================================================

import json

from src.paths import INTERIM, RAW, REPORTS


def inspect_tutor_data_context() -> None:
    """
    On documente ce qu'on utilise du côté données tuteur.

    Ce bloc n'est pas nécessaire mathématiquement, mais il est utile :
    - pour le debug
    - pour garder une trace de la présence de data/raw
    - pour montrer que le script est bien branché sur ton jeu de données réel
    """
    context = {
        "interim_exists": (INTERIM / "macro.csv").exists(),
        "raw_exists"    : RAW.exists(),
        "raw_files"     : sorted([p.name for p in RAW.glob("*")]) if RAW.exists() else [],
    }

    with open(REPORTS / "tutor_data_context.json", "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)
