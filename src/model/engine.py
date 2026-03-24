from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.config import ENFORCE_STRESS_ONLY_FLOOR, G_COL
from src.model.capital_model import compute_cet1, compute_ratio, compute_rwa
from src.model.stress_mappings import inverse_smooth_unit_interval, smooth_unit_interval
from src.model.vasicek import tail_default_prob, validate_probability_scalar


@dataclass
class ReverseStressEngine:
    """Single quantitative engine used by optimization, reporting and figures."""

    feature_cols: list[str]
    exposures: pd.DataFrame
    capital: pd.Series
    sector_params: pd.DataFrame
    g_col: str = G_COL
    enforce_stress_only_floor: bool = ENFORCE_STRESS_ONLY_FLOOR

    base_df: pd.DataFrame = field(init=False)
    feature_index: dict[str, int] = field(init=False)
    x_cols: list[str] = field(init=False)
    x_idx: list[int] = field(init=False)
    g_idx: int = field(init=False)

    ead: np.ndarray = field(init=False)
    pd0: np.ndarray = field(init=False)
    lgd0: np.ndarray = field(init=False)
    rho: np.ndarray = field(init=False)
    alpha_rwa: np.ndarray = field(init=False)
    delta_g: np.ndarray = field(init=False)
    eta_g: np.ndarray = field(init=False)
    pd_beta: np.ndarray = field(init=False)
    lgd_beta: np.ndarray = field(init=False)
    pd0_logit: np.ndarray = field(init=False)
    lgd0_anchor: np.ndarray = field(init=False)
    baseline_tail_pd: np.ndarray = field(init=False)
    baseline_loss_q_i: np.ndarray = field(init=False)

    q: float = field(init=False)
    CET1_0: float = field(init=False)
    RWA_0: float = field(init=False)
    R_omega: float = field(init=False)
    delta_non_credit: float = field(init=False)
    Lq_baseline: float = field(init=False)
    baseline_out: dict = field(init=False)

    def __post_init__(self) -> None:
        if self.g_col not in self.feature_cols:
            raise ValueError(f"Le facteur geopolitique {self.g_col!r} est absent de feature_cols")

        self.feature_index = {col: idx for idx, col in enumerate(self.feature_cols)}
        self.g_idx = self.feature_index[self.g_col]
        self.x_cols = [col for col in self.feature_cols if col != self.g_col]
        self.x_idx = [self.feature_index[col] for col in self.x_cols]

        self.q = validate_probability_scalar(float(self.capital["q"]), "q")
        self.CET1_0 = float(self.capital["CET1_0"])
        self.RWA_0 = float(self.capital["RWA_0"])
        self.R_omega = float(self.capital["R_omega"])
        self.delta_non_credit = float(self.capital["delta_non_credit"])

        params = self.sector_params.set_index("sector")
        missing_sectors = sorted(set(self.exposures["sector"]) - set(params.index))
        if missing_sectors:
            raise KeyError(
                "Secteurs absents de sector_params.csv : " + ", ".join(missing_sectors)
            )

        self.base_df = self.exposures.copy().merge(
            self.sector_params,
            on="sector",
            how="left",
            validate="many_to_one",
        )

        self.ead = self.base_df["EAD"].to_numpy(dtype=float)
        self.pd0 = self.base_df["PD0"].to_numpy(dtype=float)
        self.lgd0 = self.base_df["LGD0"].to_numpy(dtype=float)
        self.rho = self.base_df["rho"].to_numpy(dtype=float)
        self.alpha_rwa = self.base_df["alpha_rwa"].to_numpy(dtype=float)
        self.delta_g = self.base_df["delta_g"].to_numpy(dtype=float)
        self.eta_g = self.base_df["eta_g"].to_numpy(dtype=float)
        self.pd_beta = self._build_beta_matrix("b")
        self.lgd_beta = self._build_beta_matrix("c")
        self.pd0_logit = logit(np.clip(self.pd0, 1e-12, 1.0 - 1e-12))
        self.lgd0_anchor = np.vectorize(inverse_smooth_unit_interval, otypes=[float])(self.lgd0)

        self.baseline_tail_pd = tail_default_prob(self.pd0, self.rho, self.q)
        self.baseline_loss_q_i = self.ead * self.lgd0 * self.baseline_tail_pd
        self.Lq_baseline = float(self.baseline_loss_q_i.sum())
        self.baseline_out = self.evaluate(np.zeros(len(self.feature_cols)), include_exposures=True)

    def _build_beta_matrix(self, prefix: str) -> np.ndarray:
        columns = []
        n_rows = len(self.base_df)

        for col in self.x_cols:
            key = f"{prefix}_{col}"
            if key in self.base_df.columns:
                arr = pd.to_numeric(self.base_df[key], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            else:
                arr = np.zeros(n_rows, dtype=float)
            columns.append(arr)

        if not columns:
            return np.zeros((n_rows, 0), dtype=float)
        return np.column_stack(columns)

    def _coerce_s_vec(self, s_vec: np.ndarray | list[float] | pd.Series) -> np.ndarray:
        arr = np.asarray(s_vec, dtype=float).reshape(-1)
        if arr.size != len(self.feature_cols):
            raise ValueError(
                f"Dimension scenario incorrecte : attendu {len(self.feature_cols)}, recu {arr.size}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Le vecteur scenario contient des valeurs non finies")
        return arr

    def scenario_from_mapping(
        self,
        overrides: dict[str, float],
        anchor: np.ndarray | list[float] | pd.Series | None = None,
    ) -> np.ndarray:
        if anchor is None:
            s_vec = np.zeros(len(self.feature_cols), dtype=float)
        else:
            s_vec = self._coerce_s_vec(anchor).copy()

        for key, value in overrides.items():
            if key not in self.feature_index:
                raise KeyError(f"Composante scenario inconnue : {key}")
            s_vec[self.feature_index[key]] = float(value)
        return s_vec

    def _stressed_arrays(self, s_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g_val = float(s_arr[self.g_idx])
        x_vec = s_arr[self.x_idx] if self.x_idx else np.zeros(0, dtype=float)

        z_pd = self.pd0_logit + self.delta_g * g_val
        if self.x_idx:
            z_pd = z_pd + self.pd_beta @ x_vec

        pd_stress = np.clip(expit(z_pd), 1e-12, 1.0 - 1e-12)
        if self.enforce_stress_only_floor:
            pd_stress = np.maximum(pd_stress, self.pd0)

        raw_lgd = self.lgd0_anchor + self.eta_g * g_val
        if self.x_idx:
            raw_lgd = raw_lgd + self.lgd_beta @ x_vec

        lgd_stress = smooth_unit_interval(raw_lgd)
        if self.enforce_stress_only_floor:
            lgd_stress = np.maximum(lgd_stress, self.lgd0)

        return pd_stress, lgd_stress

    def stressed_exposures(self, s_vec: np.ndarray | list[float] | pd.Series) -> pd.DataFrame:
        s_arr = self._coerce_s_vec(s_vec)
        pd_stress, lgd_stress = self._stressed_arrays(s_arr)

        out = self.base_df.copy()
        out["PD_stress"] = pd_stress
        out["LGD_stress"] = lgd_stress
        return out

    def evaluate(
        self,
        s_vec: np.ndarray | list[float] | pd.Series,
        include_exposures: bool = True,
    ) -> dict:
        s_arr = self._coerce_s_vec(s_vec)
        pd_stress, lgd_stress = self._stressed_arrays(s_arr)

        tail_pd = tail_default_prob(pd_stress, self.rho, self.q)
        loss_q_i = self.ead * lgd_stress * tail_pd
        delta_loss_q_i = loss_q_i - self.baseline_loss_q_i

        Lq_abs = float(loss_q_i.sum())
        delta_Lq = float(Lq_abs - self.Lq_baseline)
        CET1 = compute_cet1(self.CET1_0, delta_Lq, self.delta_non_credit)
        RWA = compute_rwa(self.RWA_0, self.alpha_rwa, pd_stress, self.pd0)
        R = compute_ratio(CET1, RWA)
        break_slack = float(self.R_omega - R)

        result = {
            "R": R,
            "Lq_abs": Lq_abs,
            "delta_Lq": delta_Lq,
            "RWA": RWA,
            "CET1": CET1,
            "Lq_baseline": self.Lq_baseline,
            "break_slack": break_slack,
            "breaks_capital": bool(R <= self.R_omega + 1e-12),
        }

        if include_exposures:
            stressed = self.base_df.copy()
            stressed["PD_stress"] = pd_stress
            stressed["LGD_stress"] = lgd_stress
            stressed["tail_PD"] = tail_pd
            stressed["loss_q_i"] = loss_q_i
            stressed["loss_q_i_baseline"] = self.baseline_loss_q_i
            stressed["delta_loss_q_i"] = delta_loss_q_i
            result["stressed"] = stressed

        return result

    def ratio_from_mapping(
        self,
        overrides: dict[str, float],
        anchor: np.ndarray | list[float] | pd.Series | None = None,
    ) -> float:
        s_vec = self.scenario_from_mapping(overrides, anchor=anchor)
        return float(self.evaluate(s_vec, include_exposures=False)["R"])


def build_reverse_stress_engine(
    feature_cols: list[str],
    exposures: pd.DataFrame,
    capital: pd.Series,
    sector_params: pd.DataFrame,
) -> ReverseStressEngine:
    return ReverseStressEngine(
        feature_cols=feature_cols,
        exposures=exposures,
        capital=capital,
        sector_params=sector_params,
    )
