# Ce package centralise les écritures de fichiers dans outputs/reports/
# et outputs/tables/. Deux artefacts numériques sont écrits ailleurs :
#   - outputs/tables/Sigma.csv : produit par src.scenario.covariance.save_sigma
#   - outputs/tables/scenario_standardized.csv : écrit par src.scenario.build_scenario
# Cette répartition reflète le fait que ces deux fichiers appartiennent
# logiquement à l'étape « construction du scénario », pas au reporting final.

from src.reporting.save_reports import save_calibration_governance_note, save_summary
from src.reporting.save_tables import (
    save_baseline_exposure_metrics,
    save_candidate_pool,
    save_design_point,
    save_exposure_stress_at_design_point,
    save_sector_diagnostics,
    save_shortlist,
)

__all__ = [
    "save_summary",
    "save_calibration_governance_note",
    "save_baseline_exposure_metrics",
    "save_candidate_pool",
    "save_design_point",
    "save_exposure_stress_at_design_point",
    "save_sector_diagnostics",
    "save_shortlist",
]
