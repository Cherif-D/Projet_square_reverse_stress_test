from __future__ import annotations

# ============================================================
# SCRIPT PRINCIPAL FINAL — REVERSE STRESS TEST GEOPOLITIQUE
# ============================================================
#
# BUT
# ---
# On assemble ici tout le pipeline du reverse stress test, en utilisant :
#
# - data/interim/macro.csv        → bloc scénario (g, x)
# - inputs/exposures.csv          → portefeuille exposition-level
# - inputs/capital.csv            → capital initial / seuil de rupture
# - inputs/sector_params.csv      → coefficients de transmission scénario -> PD/LGD
#
# On produit ensuite :
#
# - outputs/tables/Sigma.csv
# - outputs/tables/design_point.csv
# - outputs/tables/exposure_stress_at_design_point.csv
# - outputs/tables/sector_diagnostics_at_design_point.csv
# - outputs/tables/candidate_pool.csv
# - outputs/tables/scenario_shortlist.csv
# - outputs/reports/summary.json
# - outputs/reports/summary.md
#
# LOGIQUE METHODOLOGIQUE
# ----------------------
# On suit la chaîne du papier :
#
#   s = (g, x)
#     -> PD_i(g,x), LGD_i(g,x)
#     -> L_q(g,x)
#     -> CET1(g,x), RWA(g,x)
#     -> R(s)
#     -> design point s_omega
#     -> ensembles plausibles + shortlist gouvernance-ready
#
# LIEN AVEC LE PAPIER (Hurlin, Lajaunie, Pull, 7 janvier 2026)
# ------------------------------------------------------------
# - eq. (5)  : scénario s = (g, x)^T
# - eq. (7)  : logit(PD_i(g,x)) = logit(PD0_i) + beta_k(i)^T x + delta_k(i) g
# - eq. (10) : LGD_i(g,x) = LGD0_i + gamma_k(i)^T x + eta_k(i) g
# - eq. (16) : L_q(g,x) ≈ sum_i EAD_i * LGD_i(g,x) * tail_PD_i(g,x)
# - eq. (17) : CET1(g,x) = CET1_0 - pertes + delta_non_credit
# - eq. (21) : RWA(g,x) = RWA_0 + sum_i alpha_i * (PD_i(g,x) - PD0_i)
# - eq. (22) : R(s) = CET1(g,x) / RWA(g,x)
# - eq. (24) : design point = scénario cassant le plus plausible
# - eq. (38) : near-optimal set N_phi
# - eq. (49) : réduction maximin / farthest-point
#
# CORRECTION MAJEURE DE BASELINE
# ------------------------------
# On corrige ici explicitement le point qu'on avait identifié :
#
#   si on utilise L_q(s) en niveau absolu dans CET1(s),
#   alors au point initial s=0 le ratio ne vaut plus R0.
#
# On impose donc :
#
#   Lq_baseline = Lq(s=0)
#   delta_Lq(s) = Lq(s) - Lq_baseline
#   CET1(s)     = CET1_0 - delta_Lq(s) + delta_non_credit
#
# Cela garantit bien :
#
#   R(0) = CET1_0 / RWA_0 = R0
#
# CE SCRIPT TIENT COMPTE DES DONNEES DU TUTEUR
# --------------------------------------------
# On utilise directement data/interim/macro.csv comme source principale,
# car c'est la table déjà harmonisée issue des données de ton tuteur.
#
# On garde aussi une vérification de présence de data/raw/ pour la suite :
# - recalibrage éventuel,
# - robustesse,
# - reconstruction alternative de macro.csv si besoin plus tard.
#
# IMPORTANT
# ---------
# - Les inputs portefeuille/capital/sensibilités sont maintenant cohérents.
# - Les coefficients sectoriels restent stylisés/documentés.
# - Le bloc scénario vient de tes données tuteur via macro.csv.
#
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import chi2, norm
from sklearn.covariance import LedoitWolf


# ============================================================
# 0. CHEMINS ET CONFIGURATION
# ============================================================

# On détecte automatiquement la racine du projet.
# Cas standard :
#   fichier placé dans Simulations/
#   exécution depuis la racine du projet
SCRIPT_PATH = Path(__file__).resolve()

if SCRIPT_PATH.parent.name.lower() == "simulations":
    ROOT = SCRIPT_PATH.parents[1]
else:
    ROOT = SCRIPT_PATH.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
REPORTS = OUTPUTS / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# ------------------------------------------------------------
# Hypothèses de mise en œuvre
# ------------------------------------------------------------
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
# ------------------------------------------------------------
USE_LEDOIT_WOLF = True
ENFORCE_STRESS_ONLY_FLOOR = False
G_COL = "shock_GPRD"

# ------------------------------------------------------------
# Paramètres des ensembles plausibles
# ------------------------------------------------------------
ETA_LOCAL = 1.00       # rayon local en y-space : ||y - y*||^2 <= ETA_LOCAL
PHI_NEAR = 1.00        # near-optimal : d²(s) <= d²(s*) + PHI_NEAR
POOL_SIZE = 4000
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
    "GPRD": "diff",
    "gdp": "logdiff",
    "vix": "diff",
    "sp500": "logdiff",
    "wti": "logdiff",
    "t10Y2Y": "diff",
    "unrate": "diff",
    "epu": "diff",
}


# ============================================================
# 1. OUTILS NUMERIQUES
# ============================================================

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


def sigmoid(z):
    """
    Sigmoïde standard.
    On l'utilise pour des transformations lisses.
    """
    return 1.0 / (1.0 + np.exp(-z))


def smooth_unit_interval(raw_value: float, center: float = 0.5, scale: float = 0.15) -> float:
    """
    Projection lisse dans (0,1).

    Pourquoi ?
    Parce que pour la LGD, on veut éviter un simple clip brutal
    qui casse les dérivées et peut gêner l'optimisation.

    La transformation renvoie une valeur dans (0.02, 0.98),
    ce qui évite les problèmes numériques sur les bords.
    """
    return float(0.02 + 0.96 * sigmoid((raw_value - center) / scale))


def inverse_smooth_unit_interval(u: float, center: float = 0.5, scale: float = 0.15) -> float:
    """
    Inverse de smooth_unit_interval.

    Pourquoi cette fonction est importante ?
    Parce qu'on veut que, à choc nul :
        LGD_i(s=0) = LGD0_i
    exactement.

    Si on appliquait directement smooth_unit_interval(LGD0_i),
    on ne retrouverait pas LGD0_i à la baseline.

    Donc :
    - on transforme d'abord LGD0_i en "latent anchor"
    - puis on ajoute les chocs en espace latent
    - puis on re-projette dans (0,1)
    """
    u = float(np.clip(u, 0.020001, 0.979999))
    z = (u - 0.02) / 0.96
    return float(center + scale * logit(z))


def tail_default_prob(pd_vec: np.ndarray, rho_vec: np.ndarray, q: float) -> np.ndarray:
    """
    Probabilité de défaut de queue dans l'approximation ASRF / Vasicek.

    C'est le terme :
        Phi((Phi^{-1}(PD) + sqrt(rho) * Phi^{-1}(q)) / sqrt(1-rho))

    Il apparaît dans :
    - eq. (16) du papier pour L_q(g,x)
    - eq. (59) de l'annexe sectorielle
    """
    zq = norm.ppf(q)
    arg = (norm.ppf(pd_vec) + np.sqrt(rho_vec) * zq) / np.sqrt(1.0 - rho_vec)
    return norm.cdf(arg)


def farthest_point_selection(Y: np.ndarray, P: int, start_idx: int = 0) -> list[int]:
    """
    Réduction maximin / farthest-point.

    Logique :
    - on part du scénario le plus plausible
    - puis on ajoute itérativement le point le plus éloigné
      des scénarios déjà sélectionnés en y-space

    Cela correspond à l'esprit de l'eq. (49) du papier :
    produire une petite shortlist non redondante.
    """
    selected = [start_idx]
    dmin = np.linalg.norm(Y - Y[start_idx], axis=1)

    for _ in range(1, min(P, len(Y))):
        j = int(np.argmax(dmin))
        selected.append(j)
        dnew = np.linalg.norm(Y - Y[j], axis=1)
        dmin = np.minimum(dmin, dnew)

    return selected


# ============================================================
# 2. VERIFICATION DU CONTEXTE DONNEES TUTEUR
# ============================================================

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
        "raw_exists": RAW.exists(),
        "raw_files": sorted([p.name for p in RAW.glob("*")]) if RAW.exists() else [],
    }

    with open(REPORTS / "tutor_data_context.json", "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)


# ============================================================
# 3. ETAPE 1 — CONSTRUIRE LE SCENARIO s = (g, x)
# ============================================================

def build_scenario_reference():
    """
    On charge macro.csv, on construit les chocs, puis on estime Sigma.

    LIEN AVEC LE PAPIER
    -------------------
    - eq. (5) : s = (g, x)^T
    - Section 6.2, étape 1 :
      construire les drivers géopolitiques / macro-financiers
      puis la covariance de référence Sigma.
    """

    macro_path = INTERIM / "macro.csv"
    if not macro_path.exists():
        raise FileNotFoundError(
            f"{macro_path} introuvable. "
            "Le script principal s'attend à trouver la table harmonisée du tuteur dans data/interim/macro.csv."
        )

    macro = pd.read_csv(macro_path)
    if "Date" not in macro.columns:
        raise ValueError("macro.csv doit contenir une colonne Date")

    macro["Date"] = pd.to_datetime(macro["Date"])
    macro["quarter"] = macro["Date"].dt.to_period("Q")

    scenario = pd.DataFrame({"quarter": macro["quarter"]})

    for col, kind in TRANSFORMS.items():
        if col not in macro.columns:
            raise ValueError(f"Colonne manquante dans macro.csv : {col}")

        if kind == "logdiff":
            scenario[f"shock_{col}"] = safe_log_diff(macro[col])
        elif kind == "diff":
            scenario[f"shock_{col}"] = safe_diff(macro[col])
        else:
            raise ValueError(f"Transformation inconnue pour {col}: {kind}")

    scenario = scenario.dropna().reset_index(drop=True)
    feature_cols = [c for c in scenario.columns if c.startswith("shock_")]

    # --------------------------------------------------------
    # Standardisation
    # --------------------------------------------------------
    # On standardise toutes les composantes du scénario.
    #
    # Pourquoi ?
    # - pour rendre les coefficients de sector_params interprétables
    #   comme effets pour 1 écart-type de choc ;
    # - pour éviter qu'une variable domine juste à cause de son unité.
    # --------------------------------------------------------
    mu = scenario[feature_cols].mean()
    sd = scenario[feature_cols].std(ddof=1).replace(0.0, 1.0)

    z = scenario.copy()
    z[feature_cols] = (scenario[feature_cols] - mu) / sd

    # --------------------------------------------------------
    # Recentrage sur l'état courant
    # --------------------------------------------------------
    # On prend le dernier trimestre observé comme baseline.
    # Après recentrage :
    #   s = 0 correspond à l'état de départ de la banque.
    # --------------------------------------------------------
    s0 = z.iloc[-1][feature_cols]
    z[feature_cols] = z[feature_cols].sub(s0, axis=1)

    X = z[feature_cols].values

    # --------------------------------------------------------
    # Estimation de Sigma
    # --------------------------------------------------------
    if USE_LEDOIT_WOLF:
        lw = LedoitWolf()
        lw.fit(X)
        Sigma = lw.covariance_
    else:
        Sigma = np.cov(X, rowvar=False)

    Sigma = Sigma + 1e-12 * np.eye(len(feature_cols))
    Sigma_inv = np.linalg.inv(Sigma)
    L = np.linalg.cholesky(Sigma)

    eigvals = np.linalg.eigvalsh(Sigma)
    cond_number = float(eigvals.max() / eigvals.min())

    pd.DataFrame(Sigma, index=feature_cols, columns=feature_cols).to_csv(TABLES / "Sigma.csv")
    z.to_csv(TABLES / "scenario_standardized.csv", index=False)

    sigma_report = {
        "feature_cols": feature_cols,
        "min_eigenvalue": float(eigvals.min()),
        "max_eigenvalue": float(eigvals.max()),
        "condition_number": cond_number,
        "is_spd": bool(np.all(eigvals > 0)),
        "n_observations": int(len(z)),
    }

    with open(REPORTS / "sigma_report.json", "w", encoding="utf-8") as f:
        json.dump(sigma_report, f, indent=2, ensure_ascii=False)

    return z, feature_cols, Sigma, Sigma_inv, L


# ============================================================
# 4. ETAPE 2 — CHARGER LES INPUTS CREDIT / CAPITAL
# ============================================================

def load_inputs():
    """
    On charge :
    - exposures.csv
    - capital.csv
    - sector_params.csv

    Ces trois fichiers sont la couche d'inputs qu'on a construite avant.
    """

    exp_path = INPUTS / "exposures.csv"
    cap_path = INPUTS / "capital.csv"
    sec_path = INPUTS / "sector_params.csv"

    if not exp_path.exists():
        raise FileNotFoundError(f"{exp_path} introuvable")
    if not cap_path.exists():
        raise FileNotFoundError(f"{cap_path} introuvable")
    if not sec_path.exists():
        raise FileNotFoundError(f"{sec_path} introuvable")

    exposures = pd.read_csv(exp_path)
    capital = pd.read_csv(cap_path).iloc[0]
    sector_params = pd.read_csv(sec_path)

    required_exp = {"id", "sector", "EAD", "PD0", "LGD0", "rho", "M", "alpha_rwa"}
    required_cap = {"CET1_0", "RWA_0", "R0", "R_omega", "q", "delta_non_credit", "Lq0_abs"}
    required_sec = {"sector", "delta_g", "eta_g"}

    missing_exp = required_exp - set(exposures.columns)
    missing_cap = required_cap - set(capital.index)
    missing_sec = required_sec - set(sector_params.columns)

    if missing_exp:
        raise ValueError(f"Colonnes manquantes dans exposures.csv : {missing_exp}")
    if missing_cap:
        raise ValueError(f"Colonnes manquantes dans capital.csv : {missing_cap}")
    if missing_sec:
        raise ValueError(f"Colonnes manquantes dans sector_params.csv : {missing_sec}")

    return exposures, capital, sector_params


# ============================================================
# 5. ETAPE 2 BIS — MAPPINGS PD_i(g,x) ET LGD_i(g,x)
# ============================================================

def build_stressed_exposures_fn(feature_cols, exposures, sector_params):
    """
    On construit PD_i(g,x) et LGD_i(g,x).

    LIEN AVEC LE PAPIER
    -------------------
    - eq. (7) / (8) : PD stressée en logit
    - eq. (10)      : LGD stressée en affine
    - Annexe A.2    : version sectorielle (b_k, d_k, c_k, e_k)

    Ici :
    - chaque exposition i hérite des sensibilités de son secteur k(i)
    - on reste donc dans une version parcimonieuse et robuste
    """

    params_idx = sector_params.set_index("sector")
    g_col = G_COL
    x_cols = [c for c in feature_cols if c != g_col]

    def stressed_exposures(s_vec: np.ndarray) -> pd.DataFrame:
        s = pd.Series(s_vec, index=feature_cols)

        out = exposures.copy()
        pd_vals = []
        lgd_vals = []

        for _, row in out.iterrows():
            sec = row["sector"]
            if sec not in params_idx.index:
                raise KeyError(f"Secteur absent de sector_params.csv : {sec}")

            p = params_idx.loc[sec]

            # ------------------------------------------------
            # PD stressée
            # ------------------------------------------------
            # eq. (7) :
            #   logit(PD_i(g,x)) = logit(PD0_i) + beta_k(i)^T x + delta_k(i) g
            # ------------------------------------------------
            pd0 = float(np.clip(row["PD0"], 1e-12, 1 - 1e-12))
            z_pd = logit(pd0)

            # effet géopolitique direct
            z_pd += float(p["delta_g"]) * float(s[g_col])

            # effets macro-financiers
            for c in x_cols:
                key = f"b_{c}"
                if key in p.index and pd.notna(p[key]):
                    z_pd += float(p[key]) * float(s[c])

            pd_stress = float(np.clip(expit(z_pd), 1e-12, 1 - 1e-12))

            if ENFORCE_STRESS_ONLY_FLOOR:
                pd_stress = max(pd_stress, float(row["PD0"]))

            # ------------------------------------------------
            # LGD stressée
            # ------------------------------------------------
            # eq. (10) :
            #   LGD_i(g,x) = LGD0_i + gamma_k(i)^T x + eta_k(i) g
            #
            # Pour préserver exactement LGD_i(0)=LGD0_i, on travaille
            # en espace latent puis on re-projette dans (0,1).
            # ------------------------------------------------
            lgd0 = float(row["LGD0"])
            lgd_anchor_raw = inverse_smooth_unit_interval(lgd0)

            raw_lgd = lgd_anchor_raw + float(p["eta_g"]) * float(s[g_col])

            for c in x_cols:
                key = f"c_{c}"
                if key in p.index and pd.notna(p[key]):
                    raw_lgd += float(p[key]) * float(s[c])

            lgd_stress = smooth_unit_interval(raw_lgd)

            if ENFORCE_STRESS_ONLY_FLOOR:
                lgd_stress = max(lgd_stress, float(row["LGD0"]))

            pd_vals.append(pd_stress)
            lgd_vals.append(lgd_stress)

        out["PD_stress"] = pd_vals
        out["LGD_stress"] = lgd_vals
        return out

    return stressed_exposures


# ============================================================
# 6. ETAPE 3 — L_q, CET1, RWA, R(s)
# ============================================================

def build_loss_rwa_ratio_fn(exposures, capital, stressed_exposures_fn):
    """
    On construit les fonctions :
    - Lq_abs(s)
    - delta_Lq(s)
    - CET1(s)
    - RWA(s)
    - R(s)

    LIEN AVEC LE PAPIER
    -------------------
    - eq. (16) : L_q(g,x)
    - eq. (17) : CET1(g,x)
    - eq. (21) : RWA(g,x)
    - eq. (22) : R(s)

    CORRECTION DE BASELINE
    ----------------------
    On travaille ici avec :
        delta_Lq(s) = Lq_abs(s) - Lq_baseline

    de façon à garantir :
        R(0) = R0
    """

    q = float(capital["q"])
    CET1_0 = float(capital["CET1_0"])
    RWA_0 = float(capital["RWA_0"])
    delta_non_credit = float(capital["delta_non_credit"])

    PD0_base = exposures["PD0"].to_numpy(dtype=float)
    alpha_rwa = exposures["alpha_rwa"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Baseline s = 0
    # --------------------------------------------------------
    s_zero = np.zeros(1)  # placeholder overwritten below in caller? no, we rebuild with len via closure outside? we need len(feature cols) unavailable
    # On ne peut pas utiliser s_zero ici sans connaître la dimension du scénario.
    # Donc on calcule Lq_baseline dans une fonction dédiée à partir des inputs baseline.

    baseline_tail_pd = tail_default_prob(
        exposures["PD0"].to_numpy(dtype=float),
        exposures["rho"].to_numpy(dtype=float),
        q,
    )

    Lq_baseline = float(
        np.sum(
            exposures["EAD"].to_numpy(dtype=float)
            * exposures["LGD0"].to_numpy(dtype=float)
            * baseline_tail_pd
        )
    )

    def loss_rwa_ratio(s_vec: np.ndarray):
        stressed = stressed_exposures_fn(s_vec)

        EAD = stressed["EAD"].to_numpy(dtype=float)
        PD = stressed["PD_stress"].to_numpy(dtype=float)
        LGD = stressed["LGD_stress"].to_numpy(dtype=float)
        rho = stressed["rho"].to_numpy(dtype=float)

        # ----------------------------------------------------
        # L_q(s) absolu — eq. (16)
        # ----------------------------------------------------
        tail_pd = tail_default_prob(PD, rho, q)
        stressed["tail_PD"] = tail_pd

        stressed["loss_q_i"] = EAD * LGD * tail_pd
        Lq_abs = float(stressed["loss_q_i"].sum())

        # ----------------------------------------------------
        # Correction baseline : delta_Lq(s)
        # ----------------------------------------------------
        delta_Lq = float(Lq_abs - Lq_baseline)

        # ----------------------------------------------------
        # CET1(s) corrigé baseline
        # ----------------------------------------------------
        CET1 = float(CET1_0 - delta_Lq + delta_non_credit)

        # ----------------------------------------------------
        # RWA(s) — eq. (21)
        # ----------------------------------------------------
        RWA = float(RWA_0 + np.sum(alpha_rwa * (PD - PD0_base)))
        RWA = max(RWA, 1e-12)

        # ----------------------------------------------------
        # Ratio — eq. (22)
        # ----------------------------------------------------
        R = CET1 / RWA

        # Diagnostics enrichis
        stressed["delta_loss_q_i"] = stressed["loss_q_i"] - (
            stressed["EAD"].to_numpy(dtype=float)
            * exposures["LGD0"].to_numpy(dtype=float)
            * baseline_tail_pd
        )

        return {
            "R": R,
            "Lq_abs": Lq_abs,
            "delta_Lq": delta_Lq,
            "RWA": RWA,
            "CET1": CET1,
            "Lq_baseline": Lq_baseline,
            "stressed": stressed,
        }

    return loss_rwa_ratio


# ============================================================
# 7. ETAPE 4 — DESIGN POINT EN ESPACE BLANCHI
# ============================================================

def build_starting_points(feature_cols, L):
    """
    On construit un multi-start robuste :
    - random starts en y-space
    - ladder sur l'intensité géopolitique

    Pourquoi ?
    Parce que l'optimisation inverse peut être sensible au point de départ.
    """
    g_idx = feature_cols.index(G_COL)
    starts = []

    # starts aléatoires
    for _ in range(20):
        y0 = RNG.normal(0.0, 0.5, len(feature_cols))
        y0[g_idx] = abs(y0[g_idx]) + 0.25
        starts.append(y0)

    # ladder sur g
    for level in [0.25, 0.50, 1.00, 1.50, 2.00, 3.00]:
        s0 = np.zeros(len(feature_cols))
        s0[g_idx] = level
        y0 = np.linalg.solve(L, s0)
        starts.append(y0)

    return starts


def solve_design_point(feature_cols, Sigma_inv, L, capital, loss_rwa_ratio_fn):
    """
    On résout le design point.

    Sous hypothèse gaussienne :
        maximiser la plausibilité
    revient à
        minimiser la distance de Mahalanobis

    En espace blanchi :
        s = L y
    la distance devient :
        d²(s) = ||y||²

    On résout donc :
        min_y  1/2 ||y||²
        s.c.   R(Ly) <= R_omega
               g(Ly) >= 0
    """
    g_idx = feature_cols.index(G_COL)
    R_omega = float(capital["R_omega"])

    def s_from_y(y):
        return L @ y

    def objective_y(y):
        return 0.5 * float(np.dot(y, y))

    def cst_break(y):
        s = s_from_y(y)
        out = loss_rwa_ratio_fn(s)
        return R_omega - out["R"]

    def cst_g_positive(y):
        s = s_from_y(y)
        return s[g_idx]

    constraints = [
        {"type": "ineq", "fun": cst_break},
        {"type": "ineq", "fun": cst_g_positive},
    ]

    starts = build_starting_points(feature_cols, L)
    sols = []

    for y0 in starts:
        res = minimize(
            objective_y,
            y0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-9, "disp": False},
        )

        if not res.success:
            continue

        y_star = res.x
        s_star = s_from_y(y_star)
        out = loss_rwa_ratio_fn(s_star)

        if out["R"] <= R_omega + 1e-6 and s_star[g_idx] >= -1e-8:
            d2 = float(s_star.T @ Sigma_inv @ s_star)
            pval = float(1.0 - chi2.cdf(d2, df=len(feature_cols)))

            sols.append({
                "y_star": y_star,
                "s_star": s_star,
                "R": out["R"],
                "Lq_abs": out["Lq_abs"],
                "delta_Lq": out["delta_Lq"],
                "RWA": out["RWA"],
                "CET1": out["CET1"],
                "Lq_baseline": out["Lq_baseline"],
                "d2": d2,
                "pval": pval,
                "stressed": out["stressed"],
                "success": True,
            })

    if not sols:
        raise RuntimeError("Aucune solution faisable trouvée. Vérifie les inputs ou la sévérité des paramètres.")

    best = min(sols, key=lambda z: z["d2"])
    return best, sols


# ============================================================
# 8. ETAPE 5 — S_eta ET N_phi
# ============================================================

def generate_candidate_pool(best, feature_cols, L, Sigma_inv, capital, loss_rwa_ratio_fn,
                            eta_local=ETA_LOCAL, phi_near=PHI_NEAR, n=POOL_SIZE):
    """
    On construit un pool de scénarios admissibles :

    - S_eta : voisinage local autour du design point
    - N_phi : near-optimal set avec d² <= d²* + phi

    Cela correspond à l'esprit de la Section 5 du papier :
    ne pas s'arrêter au point unique, mais explorer un ensemble plausible.
    """
    y_star = best["y_star"]
    d2_star = best["d2"]
    R_omega = float(capital["R_omega"])
    g_idx = feature_cols.index(G_COL)

    rows = []

    for _ in range(n):
        if RNG.uniform() < 0.5:
            y = y_star + RNG.normal(0.0, 0.35, len(feature_cols))
        else:
            y = RNG.normal(0.0, 1.0, len(feature_cols))
            y[g_idx] = abs(y[g_idx]) + 0.1

        s = L @ y
        d2 = float(s.T @ Sigma_inv @ s)
        local_dist2 = float(np.dot(y - y_star, y - y_star))

        out = loss_rwa_ratio_fn(s)

        if s[g_idx] < 0:
            continue
        if out["R"] > R_omega:
            continue

        in_local = local_dist2 <= eta_local
        in_near = d2 <= d2_star + phi_near

        if in_local or in_near:
            rows.append({
                "set_type": "local" if in_local else "near",
                "d2": d2,
                "local_dist2": local_dist2,
                "R": out["R"],
                "Lq_abs": out["Lq_abs"],
                "delta_Lq": out["delta_Lq"],
                "RWA": out["RWA"],
                "CET1": out["CET1"],
                **{c: float(s[i]) for i, c in enumerate(feature_cols)},
            })

    pool = pd.DataFrame(rows)
    return pool


# ============================================================
# 9. ETAPE 6 — SHORTLIST MAXIMIN
# ============================================================

def build_shortlist(pool: pd.DataFrame, feature_cols: list[str], L: np.ndarray, P: int = SHORTLIST_SIZE):
    """
    On réduit le pool à une shortlist non redondante.

    On travaille en y-space car les distances y sont directement
    cohérentes avec la géométrie de Mahalanobis.
    """
    if pool.empty:
        return pool.copy()

    Y = np.linalg.solve(L, pool[feature_cols].values.T).T
    start_idx = int(np.argmin(pool["d2"].values))
    idx = farthest_point_selection(Y, P=min(P, len(pool)), start_idx=start_idx)
    shortlist = pool.iloc[idx].copy()
    return shortlist


# ============================================================
# 10. DIAGNOSTICS SECTORIELS AU DESIGN POINT
# ============================================================

def build_sector_diagnostics(stressed_exposures_df: pd.DataFrame):
    """
    On agrège les résultats par secteur.

    Intuition :
    - le script principal optimise au niveau exposition-level ;
    - mais pour le reporting, on veut aussi une lecture sectorielle.

    On calcule :
    - EAD sectorielle
    - PD moyenne pondérée EAD
    - LGD moyenne pondérée EAD
    - Tail PD moyenne pondérée
    - perte de queue sectorielle
    """
    df = stressed_exposures_df.copy()

    grp = df.groupby("sector", as_index=False).apply(
        lambda g: pd.Series({
            "EAD_sector": g["EAD"].sum(),
            "PD_sector_stress": np.average(g["PD_stress"], weights=g["EAD"]),
            "LGD_sector_stress": np.average(g["LGD_stress"], weights=g["EAD"]),
            "TailPD_sector": np.average(g["tail_PD"], weights=g["EAD"]),
            "LossQ_sector": g["loss_q_i"].sum(),
            "DeltaLossQ_sector": g["delta_loss_q_i"].sum(),
        })
    ).reset_index(drop=True)

    return grp


# ============================================================
# 11. MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Contexte données tuteur
    # --------------------------------------------------------
    inspect_tutor_data_context()

    # --------------------------------------------------------
    # Étape 1 : scénario / Sigma
    # --------------------------------------------------------
    scenario_std, feature_cols, Sigma, Sigma_inv, L = build_scenario_reference()

    # --------------------------------------------------------
    # Étape 2 : inputs crédit / capital
    # --------------------------------------------------------
    exposures, capital, sector_params = load_inputs()

    # --------------------------------------------------------
    # Étape 3 : fonctions PD/LGD
    # --------------------------------------------------------
    stressed_exposures_fn = build_stressed_exposures_fn(feature_cols, exposures, sector_params)

    # --------------------------------------------------------
    # Étape 4 : fonctions Lq / CET1 / RWA / ratio
    # --------------------------------------------------------
    loss_rwa_ratio_fn = build_loss_rwa_ratio_fn(exposures, capital, stressed_exposures_fn)

    # --------------------------------------------------------
    # Baseline s = 0
    # --------------------------------------------------------
    s_zero = np.zeros(len(feature_cols))
    baseline_out = loss_rwa_ratio_fn(s_zero)

    baseline_stressed = baseline_out["stressed"].copy()
    baseline_stressed.to_csv(TABLES / "baseline_exposure_metrics.csv", index=False)

    # --------------------------------------------------------
    # Étape 5 : design point
    # --------------------------------------------------------
    best, all_optima = solve_design_point(feature_cols, Sigma_inv, L, capital, loss_rwa_ratio_fn)

    design_point = pd.Series(best["s_star"], index=feature_cols)
    design_point.to_csv(TABLES / "design_point.csv")

    best["stressed"].to_csv(TABLES / "exposure_stress_at_design_point.csv", index=False)

    sector_diag = build_sector_diagnostics(best["stressed"])
    sector_diag.to_csv(TABLES / "sector_diagnostics_at_design_point.csv", index=False)

    # --------------------------------------------------------
    # Étape 6 : S_eta / N_phi
    # --------------------------------------------------------
    pool = generate_candidate_pool(
        best=best,
        feature_cols=feature_cols,
        L=L,
        Sigma_inv=Sigma_inv,
        capital=capital,
        loss_rwa_ratio_fn=loss_rwa_ratio_fn,
        eta_local=ETA_LOCAL,
        phi_near=PHI_NEAR,
        n=POOL_SIZE,
    )
    pool.to_csv(TABLES / "candidate_pool.csv", index=False)

    # --------------------------------------------------------
    # Étape 7 : shortlist
    # --------------------------------------------------------
    shortlist = build_shortlist(pool, feature_cols, L, P=SHORTLIST_SIZE)
    shortlist.to_csv(TABLES / "scenario_shortlist.csv", index=False)

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------
    top_drivers = design_point.abs().sort_values(ascending=False).head(8).to_dict()

    summary = {
        "feature_cols": feature_cols,
        "baseline_ratio_model": float(baseline_out["R"]),
        "baseline_Lq_abs": float(baseline_out["Lq_abs"]),
        "baseline_delta_Lq": float(baseline_out["delta_Lq"]),
        "baseline_RWA_model": float(baseline_out["RWA"]),
        "baseline_CET1_model": float(baseline_out["CET1"]),
        "baseline_Lq_reference": float(baseline_out["Lq_baseline"]),
        "R_omega": float(capital["R_omega"]),
        "R0_input": float(capital["R0"]),
        "q": float(capital["q"]),
        "design_point_ratio": float(best["R"]),
        "design_point_Lq_abs": float(best["Lq_abs"]),
        "design_point_delta_Lq": float(best["delta_Lq"]),
        "design_point_RWA": float(best["RWA"]),
        "design_point_CET1": float(best["CET1"]),
        "design_point_d2": float(best["d2"]),
        "design_point_pvalue": float(best["pval"]),
        "n_local_optima_found": int(len(all_optima)),
        "n_candidate_pool": int(len(pool)),
        "n_shortlist": int(len(shortlist)),
        "top_drivers_abs": {k: float(v) for k, v in top_drivers.items()},
        "eta_local": ETA_LOCAL,
        "phi_near": PHI_NEAR,
        "baseline_check_R_equals_R0": float(baseline_out["R"]),
    }

    with open(REPORTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(REPORTS / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Reverse Stress Test — script principal final\n\n")
        f.write("## Baseline\n")
        f.write(f"- Ratio baseline modèle : {baseline_out['R']:.6f}\n")
        f.write(f"- Ratio initial input R0 : {float(capital['R0']):.6f}\n")
        f.write(f"- Lq absolu baseline : {baseline_out['Lq_abs']:.6f}\n")
        f.write(f"- DeltaLq baseline : {baseline_out['delta_Lq']:.6f}\n")
        f.write(f"- RWA baseline : {baseline_out['RWA']:.6f}\n")
        f.write(f"- CET1 baseline : {baseline_out['CET1']:.6f}\n")
        f.write(f"- Lq référence (s=0) : {baseline_out['Lq_baseline']:.6f}\n\n")

        f.write("## Design point\n")
        f.write(f"- Seuil de rupture R_omega : {float(capital['R_omega']):.6f}\n")
        f.write(f"- Ratio design point : {best['R']:.6f}\n")
        f.write(f"- Lq absolu design point : {best['Lq_abs']:.6f}\n")
        f.write(f"- DeltaLq design point : {best['delta_Lq']:.6f}\n")
        f.write(f"- RWA design point : {best['RWA']:.6f}\n")
        f.write(f"- CET1 design point : {best['CET1']:.6f}\n")
        f.write(f"- d² Mahalanobis : {best['d2']:.6f}\n")
        f.write(f"- p-value plausibilité : {best['pval']:.8f}\n\n")

        f.write("## Ensembles de scénarios\n")
        f.write(f"- Nombre d'optima faisables trouvés : {len(all_optima)}\n")
        f.write(f"- Taille du pool candidat : {len(pool)}\n")
        f.write(f"- Taille de la shortlist : {len(shortlist)}\n\n")

        f.write("## Top drivers du design point\n")
        for k, v in design_point.abs().sort_values(ascending=False).head(8).items():
            f.write(f"- {k}: {float(v):.4f}\n")

    print("OK - script principal final terminé.")
    print("Sorties disponibles dans :", OUTPUTS)
    print(f"Baseline check : R(0) = {baseline_out['R']:.6f} vs R0 input = {float(capital['R0']):.6f}")


if __name__ == "__main__":
    main()