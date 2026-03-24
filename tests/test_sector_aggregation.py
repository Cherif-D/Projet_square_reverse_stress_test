from __future__ import annotations

import unittest

import pandas as pd

from src.portfolio.sector_aggregation import build_sector_diagnostics


class SectorAggregationTests(unittest.TestCase):
    def test_sector_table_reconciles_to_total_delta_lq(self) -> None:
        baseline = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "sector": ["Energy", "Energy", "Tech"],
                "EAD": [10.0, 20.0, 30.0],
                "PD_stress": [0.01, 0.02, 0.03],
                "LGD_stress": [0.4, 0.45, 0.5],
                "tail_PD": [0.05, 0.06, 0.07],
                "loss_q_i": [0.2, 0.6, 1.0],
            }
        )
        stressed = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "sector": ["Energy", "Energy", "Tech"],
                "EAD": [10.0, 20.0, 30.0],
                "PD_stress": [0.011, 0.025, 0.034],
                "LGD_stress": [0.41, 0.48, 0.52],
                "tail_PD": [0.055, 0.07, 0.08],
                "loss_q_i": [0.23, 0.73, 1.24],
                "delta_loss_q_i": [0.03, 0.13, 0.24],
            }
        )

        diag = build_sector_diagnostics(baseline, stressed)

        self.assertIn("PD_sector_baseline", diag.columns)
        self.assertIn("PD_sector_stress", diag.columns)
        self.assertIn("Delta_PD_sector_bp", diag.columns)
        self.assertIn("LGD_sector_baseline", diag.columns)
        self.assertIn("LGD_sector_stress", diag.columns)
        self.assertIn("Delta_LGD_sector_pp", diag.columns)
        self.assertAlmostEqual(diag["DeltaLossQ_sector"].sum(), stressed["delta_loss_q_i"].sum())


if __name__ == "__main__":
    unittest.main()
