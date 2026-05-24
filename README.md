# Reverse Stress Test Géopolitique — implémentation Python

Implémentation Python du cadre de **reverse stress test (RST) géopolitique**
appliqué à un portefeuille de crédit corporate, dans l'esprit du papier :

> Hurlin, C., Lajaunie, Q., and Pull, Y. (2026).
> *Reverse Stress Testing Geopolitical Risk in Corporate Credit Portfolios :
> A Formal and Operational Framework.*

Le dépôt accompagne la note rédigée dans le cadre du Master 1 ESA — programme
Square Management : *Stress Test Inversé du Risque Géopolitique dans les
Portefeuilles de Crédit aux Entreprises* (Mars 2026, version révisée Mai 2026).

**Statut.** Prototype académique reproductible, **non model-approved**. Les
sorties sont labellisées `prototype / academic` tant qu'une calibration
empirique sur données bancaires réelles n'est pas en place. Cet état est
explicite dans le rapport de gouvernance généré à chaque run
(`outputs/reports/calibration_backtesting_note.md`).

---

## Table des matières

1. [Aperçu de la méthode](#1-aperçu-de-la-méthode)
2. [Architecture du dépôt](#2-architecture-du-dépôt)
3. [Prérequis et installation](#3-prérequis-et-installation)
4. [Reproduction en une commande](#4-reproduction-en-une-commande)
5. [Substituer les données simulées par des données réelles](#5-substituer-les-données-simulées-par-des-données-réelles)
6. [Sorties produites](#6-sorties-produites)
7. [Lancement des tests](#7-lancement-des-tests)
8. [Formules clés](#8-formules-clés)
9. [Paramètres centralisés](#9-paramètres-centralisés)
10. [Références](#10-références)

---

## 1. Aperçu de la méthode

Un stress test forward part d'un scénario adverse imposé et calcule l'impact
sur le capital. Le reverse stress test inverse la logique : on part d'un état
de rupture (ratio CET1 sous un seuil critique) et on cherche le scénario
géopolitique et macro-financier **le plus plausible** qui l'atteint.

Formellement, le *design point* `s*` est la solution du programme

$$s^\star \in \arg\min_{s \in \mathcal{S}_{\mathrm{red}},\ g \geq 0} \tfrac{1}{2}\, d^2(s),$$

où `d²(s) = sᵀ Σ⁻¹ s` est la distance de Mahalanobis sous la loi gaussienne
de référence et `S_red = {s : R(s) <= R_omega}` la zone où le ratio de
capital franchit le seuil. Le pipeline propage chaque scénario `s` vers les
PD et LGD sectorielles, vers la perte de queue Vasicek au quantile
prudentiel et vers le ratio `R(s) = CET1(s) / RWA(s)`. La contrainte
théorique `g > 0` du papier est écrite `g >= 0` dans le code pour rester
compatible avec le solveur SLSQP.

Le run de référence (paramètres par défaut, `SEED = 42`) :

- ratio baseline `R_0 = 14 %`, seuil de rupture `R_omega = 11 %` ;
- design point sur la frontière, slack `≈ 1.06 × 10⁻¹⁴` ;
- distance de Mahalanobis `d²(s*) = 0,076508`, p-valeur `≈ 0,99999991` ;
- pool de 3 516 scénarios candidats, shortlist finale de 8 scénarios diversifiés.

---

## 2. Architecture du dépôt

Le dépôt s'organise en quatre couches fonctionnelles :

```
Reverse_stress_test/
│
├── Simulations/                ← Couche 1 : génération des inputs (étape 0)
│   ├── simulate_exposures.py   → inputs/exposures.csv      (500 emprunteurs)
│   ├── simulate_capital.py     → inputs/capital.csv        (paramètres de capital)
│   └── simulate_sector_params.py → inputs/sector_params.csv (coefficients sectoriels)
│
├── inputs/                     ← Trois CSV substituables (cf. § 5)
│   ├── exposures.csv
│   ├── capital.csv
│   └── sector_params.csv
│
├── data/
│   ├── interim/macro.csv       ← Table macro harmonisée (8 variables, 128 obs.)
│   └── raw/                    ← Sources macro brutes
│
├── src/                        ← Couche 2 : moteur analytique
│   ├── config.py               ← Constantes globales (SEED, ETA_LOCAL, PHI_NEAR, …)
│   ├── paths.py                ← Chemins centralisés
│   ├── run_rst.py              ← Orchestrateur du pipeline RST
│   ├── scenario/               ← Standardisation, covariance Ledoit-Wolf, Cholesky
│   ├── portfolio/              ← Chargement et agrégation sectorielle
│   ├── model/                  ← Vasicek ASRF, mappings PD/LGD, capital, ratio
│   ├── optimization/           ← SLSQP multi-start, pool, shortlist farthest-point
│   ├── reporting/              ← Export tables et rapports
│   └── visualization/          ← Figures du papier (fig1 à fig5) + profils sectoriels
│
├── outputs/                    ← Couche 3 : sorties générées par run_project.py
│   ├── tables/                 ← CSV auditables (design point, pool, shortlist, …)
│   ├── reports/                ← summary.json, summary.md, sigma_report.json, …
│   └── plots/                  ← 13 figures PNG (9 pipeline + 4 héritées, 300 dpi)
│
├── tests/                      ← Couche 4 : tests unitaires ciblés
│   ├── test_vasicek.py
│   └── test_sector_aggregation.py
│
├── run_project.py              ← Point d'entrée unique (Simulations → pipeline → graphes)
└── requirements.txt            ← Dépendances Python (Python 3.12+)
```

---

## 3. Prérequis et installation

- **Python 3.12 ou supérieur.**
- La table macro harmonisée `data/interim/macro.csv` doit être présente avant
  le lancement (elle contient 129 lignes trimestrielles brutes sur huit
  variables : `GPRD`, `gdp`, `vix`, `sp500`, `wti`, `t10Y2Y`, `unrate`, `epu`).
  Après transformations et suppression des premières valeurs manquantes, le
  scénario standardisé utilisé par le pipeline contient 128 observations.
  Cette table n'est pas regénérée par `run_project.py`.

### PowerShell (Windows)

```powershell
cd C:\Master_1_ESA\Projet_square\Reverse_stress_test
py -3.12 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Bash (Linux / macOS / WSL)

```bash
cd Reverse_stress_test
python3.12 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

---

## 4. Reproduction en une commande

```powershell
python run_project.py
```

Cette commande exécute, dans l'ordre, les quatre étapes du pipeline :

| Étape | Fonction                       | Sorties principales |
|-------|--------------------------------|---------------------|
| 0     | `run_simulations()`            | `inputs/exposures.csv`, `inputs/capital.csv`, `inputs/sector_params.csv` |
| 1     | `run_pipeline()` (src/run_rst) | Tables CSV, `summary.json`/`summary.md`, design point `s*` |
| 2     | `run_legacy_visualizations()`  | `frontier_plot.png`, profils et impact sectoriel (versions legacy) |
| 3     | `run_src_visualizations()`     | Huit figures `src/visualization/` (`fig1` à `fig5` + profils sectoriels) |

Pour relancer uniquement le pipeline analytique sans regénérer les inputs :

```powershell
python -m src.run_rst
```

Pour regénérer uniquement les inputs simulés :

```powershell
python Simulations\simulate_exposures.py
python Simulations\simulate_capital.py
python Simulations\simulate_sector_params.py
```

---

## 5. Substituer les données simulées par des données réelles

Le pipeline analytique ne dépend que des trois CSV de `inputs/`. Un
utilisateur peut **remplacer ces trois fichiers par ses propres données**
sans modifier le code de `src/`, à condition de respecter le schéma
ci-dessous.

### 5.1 `inputs/exposures.csv` — portefeuille de crédit (500 lignes ou plus)

| Colonne | Type | Bornes / contraintes | Description |
|---|---|---|---|
| `id` | entier | unique | Identifiant de l'emprunteur |
| `sector` | chaîne | doit appartenir aux secteurs déclarés dans `sector_params.csv` | Secteur économique |
| `EAD` | flottant | > 0 | Exposure At Default (unités monétaires homogènes au CET1/RWA) |
| `PD0` | flottant | strictement dans (0, 1), bornes pratiques [0.0005, 0.20] | Probabilité de défaut baseline |
| `LGD0` | flottant | strictement dans (0, 1), bornes pratiques [0.10, 0.90] | Loss Given Default baseline |
| `rho` | flottant | bornes pratiques [0.08, 0.30] | Corrélation d'actif Vasicek |
| `M` | flottant | bornes pratiques [1.0, 5.0] | Maturité effective en années |
| `alpha_rwa` | flottant | > 0 | Pente locale des RWA par rapport à la PD (eq. 21 du papier) |
| `K0_proxy` | flottant | ≥ 0 | Proxy de charge en capital baseline (diagnostic, peut être recalculé) |
| `RW0_exact_proxy` | flottant | ≥ 0 | Densité RWA baseline = 12,5 × `K0_proxy` |

Exemple — premières lignes du fichier simulé de référence :

```
id,sector,EAD,PD0,LGD0,rho,M,alpha_rwa,K0_proxy,RW0_exact_proxy
201,Consumer,0.18727632396730098,0.00556058945989303,0.38371106033479124,0.19094195638334505,2.184185927231544,4.829302567396034,0.03277512182276463,0.4096890227845579
202,Consumer,0.49053752307111537,0.003959815704424318,0.41298320542553335,0.19728747321457493,2.3292563225708816,12.610901881015646,0.029380951741177757,0.367261896764722
```

### 5.2 `inputs/capital.csv` — paramètres de capital (1 ligne)

| Colonne | Type | Bornes / contraintes | Description |
|---|---|---|---|
| `CET1_0` | flottant | > 0 | Capital CET1 initial |
| `RWA_0` | flottant | > 0 | Actifs pondérés initiaux |
| `R0` | flottant | dans (0, 1) | Ratio baseline `R_0 = CET1_0 / RWA_0` |
| `R_omega` | flottant | strictement inférieur à `R0` | Seuil de rupture |
| `q` | flottant | dans (0, 1), typiquement 0.999 | Quantile prudentiel IRB |
| `delta_non_credit` | flottant | libre, 0.0 par défaut | Ajustement P&L non-crédit |
| `Lq0_abs` | flottant | ≥ 0 | Perte de queue baseline absolue (diagnostic) |
| `baseline_convention` | chaîne | fixée à `use_delta_Lq_in_main_script` | Convention de correction baseline |

Exemple :

```
CET1_0,RWA_0,R0,R_omega,q,delta_non_credit,Lq0_abs,baseline_convention
5.428508523913509,38.77506088509649,0.14,0.11,0.999,0.0,3.2036103423714497,use_delta_Lq_in_main_script
```

### 5.3 `inputs/sector_params.csv` — coefficients sectoriels (1 ligne par secteur)

| Colonne | Type | Description |
|---|---|---|
| `sector` | chaîne | Doit correspondre aux secteurs présents dans `exposures.csv` |
| `param_origin` | chaîne | `stylized` ou `empirical` |
| `calibration_status` | chaîne | Statut documentaire (ex. `not_empirically_estimated`) |
| `note` | chaîne | Note libre sur la calibration |
| `delta_g` | flottant | Sensibilité directe de la PD au choc géopolitique `g` (en logit) |
| `eta_g` | flottant | Sensibilité directe de la LGD au choc géopolitique `g` (en espace latent) |
| `b_shock_gdp`, `b_shock_vix`, `b_shock_sp500`, `b_shock_wti`, `b_shock_t10Y2Y`, `b_shock_unrate`, `b_shock_epu` | flottants | Sensibilités macro de la PD (eq. 7 du papier) |
| `c_shock_gdp`, `c_shock_vix`, `c_shock_sp500`, `c_shock_wti`, `c_shock_t10Y2Y`, `c_shock_unrate`, `c_shock_epu` | flottants | Sensibilités macro de la LGD (eq. 10 du papier) |

Exemple — première ligne du fichier simulé :

```
sector,param_origin,calibration_status,note,delta_g,eta_g,b_shock_gdp,b_shock_vix,b_shock_sp500,b_shock_wti,b_shock_t10Y2Y,b_shock_unrate,b_shock_epu,c_shock_gdp,c_shock_vix,c_shock_sp500,c_shock_wti,c_shock_t10Y2Y,c_shock_unrate,c_shock_epu
Energy,stylized,not_empirically_estimated,"sector hierarchy economically grounded, numeric values stylized",0.65,0.1,-0.18,0.16,-0.05,-0.1,-0.03,0.12,0.14,-0.02,0.02,-0.01,-0.01,0.0,0.02,0.02
```

### 5.4 Avertissements

- **Cohérence sectorielle.** Tout secteur présent dans `exposures.csv` doit
  apparaître dans `sector_params.csv`. Le pipeline lève une erreur de
  chargement sinon.
- **Échelle PD / LGD.** Maintenir `PD0` et `LGD0` strictement dans (0, 1).
  Une PD baseline trop proche de 0 ou de 1 dégrade la stabilité numérique
  de la transmission logit / latent.
- **Signe des coefficients.** Les signes traduisent la logique économique :
  `b_shock_gdp` négatif (PIB en hausse → PD en baisse), `b_shock_unrate` et
  `b_shock_vix` positifs (chômage ou volatilité en hausse → PD en hausse).
  Un signe inverse fausserait l'orientation du design point sans erreur
  explicite — la cohérence est de la responsabilité de l'utilisateur.
- **Seuil de rupture.** `R_omega` doit être strictement inférieur à `R0`,
  faute de quoi la zone de rupture est vide et l'optimisation n'a pas de
  solution faisable.
- **Quantile prudentiel.** `q = 0.999` est le standard IRB ; modifier cette
  valeur sans révision du modèle peut décaler les ordres de grandeur.
- **Table macro.** `data/interim/macro.csv` doit rester présent et
  harmonisé sur les huit variables `GPRD`, `gdp`, `vix`, `sp500`, `wti`,
  `t10Y2Y`, `unrate`, `epu`. Cette table est l'entrée du module
  `src/scenario/`.

---

## 6. Sorties produites

### 6.1 Tables (`outputs/tables/`)

| Fichier | Contenu |
|---|---|
| `Sigma.csv` | Matrice de covariance Sigma (8 × 8), estimée par Ledoit-Wolf |
| `scenario_standardized.csv` | Historique des chocs macro standardisés et recentrés |
| `baseline_exposure_metrics.csv` | Métriques de crédit à la baseline par emprunteur |
| `design_point.csv` | Coordonnées du design point `s*` (huit z-scores) |
| `exposure_stress_at_design_point.csv` | PD, LGD et perte de queue par emprunteur en `s*` |
| `sector_diagnostics_at_design_point.csv` | Agrégats sectoriels (EAD, PD, LGD, ΔLq) en `s*` |
| `candidate_pool.csv` | Pool de scénarios candidats `C_N` après filtrage et déduplication |
| `scenario_shortlist.csv` | Shortlist finale `C_P` (8 scénarios) |

### 6.2 Rapports (`outputs/reports/`)

| Fichier | Contenu |
|---|---|
| `summary.json` | Tous les indicateurs clés (baseline, design point, ensembles) |
| `summary.md` | Version lisible du résumé |
| `sigma_report.json` | Détails de l'estimation de Sigma |
| `calibration_backtesting_note.md` | Limites, calibration minimale et garde-fous de gouvernance |
| `tutor_data_context.json` | Contexte complet des données pour auditabilité |

### 6.3 Graphiques (`outputs/plots/`)

Treize figures PNG à 300 dpi : neuf produites par le pipeline principal
(`src/visualization/` et scripts legacy) et quatre figures héritées du
script `visualize_design_point_multidim.py` (`design_point_candidates_3d`,
`design_point_local_frontier_slices`, `design_point_pairplot`,
`design_point_pca`).

Les neuf figures principales sont : les cinq figures du papier
(`fig1_geometry`, `fig2_near_optimal`, `fig3_roadmap`,
`fig4_scenario_selection`, `fig5_plausibility`), le profil du design
point (`scenario_profile`), les pertes de queue par secteur
(`sector_impact`), les bulles PD / LGD par secteur (`sector_pd_lgd`) et
la frontière legacy (`frontier_plot`).

---

## 7. Lancement des tests

```powershell
python -m unittest discover -s tests -v
```

Deux modules de tests sont actuellement présents :

- `tests/test_vasicek.py` — robustesse numérique et monotonie du module
  Vasicek ASRF.
- `tests/test_sector_aggregation.py` — réconciliation de l'agrégation
  sectorielle (somme des contributions sectorielles vs ΔLq global).

---

## 8. Formules clés

Vecteur de scénario en z-scores :

$$s = (g, x_1, \ldots, x_7) \in \mathbb{R}^8, \qquad g = \mathrm{shock\_GPRD}.$$

Distance de Mahalanobis sous la loi gaussienne de référence :

$$d^2(s) = s^\top \Sigma^{-1} s, \qquad y = L^{-1} s, \qquad \Sigma = L L^\top.$$

Transmission de la PD (eq. 7 du papier, écriture logit) :

$$\mathrm{logit}\bigl(PD_{\text{stress}}(i)\bigr) = \mathrm{logit}\bigl(PD_0(i)\bigr) + \delta_g(i) \cdot g + \sum_j b_j(i) \cdot x_j.$$

Après reprojection logistique, `PD_stress(i)` est strictement dans `(0, 1)`
et `PD_stress(i) = PD_0(i)` exactement en `s = 0`.

Transmission de la LGD (adaptation lisse de l'eq. 10 du papier) :

$$z^{LGD}_i(s) = \phi^{-1}\bigl(LGD_0(i)\bigr) + \eta_g(i) \cdot g + \sum_j c_j(i) \cdot x_j,$$

$$LGD_{\text{stress}}(i) = \phi\bigl(z^{LGD}_i(s)\bigr),$$

où `phi` est une projection lisse bornée dans `(0, 1)`. Le papier écrit une
LGD affine, tronquée si nécessaire ; l'implémentation applique cette logique
en espace latent pour préserver la baseline.

Perte de queue par emprunteur (Vasicek ASRF, eq. 16 du papier) :

$$L_q(i, s) = EAD_i \cdot LGD_{\text{stress}}(i) \cdot \Phi\biggl(\frac{\Phi^{-1}(PD_{\text{stress}}(i)) + \sqrt{\rho_i}\, \Phi^{-1}(q)}{\sqrt{1 - \rho_i}}\biggr).$$

Capital, RWA et ratio sous stress (adaptation baseline de l'eq. 17, puis
eq. 21 et 22 du papier) :

$$\Delta L_q(s) = L_q(s) - L_q(0),$$

$$CET1(s) = CET1_0 - \Delta L_q(s) + \Delta_{\text{non-credit}}, \qquad RWA(s) = RWA_0 + \sum_i \alpha_i \cdot \bigl(PD_{\text{stress}}(i) - PD_0(i)\bigr),$$

$$R(s) = \frac{CET1(s)}{RWA(s)}.$$

Cette adaptation garantit `R(0) = R_0` exactement, car `CET1_0` est déjà le
capital observé au point de départ.

Ensembles plausibles :

$$\mathcal{B}_{\eta}(s^\star) = \bigl\lbrace s : \lVert L^{-1}(s - s^\star) \rVert^2 \leq \eta \bigr\rbrace,$$

$$\mathcal{N}_{\varphi} =
\bigl\lbrace s \in \mathcal{S}_{\mathrm{red}} : g \geq 0,\ d^2(s) \leq d^2(s^\star) + \varphi \bigr\rbrace.$$

Réduction farthest-point pour la shortlist (eq. 49 du papier) :

$$s^{(p)} = \arg\max_{s \in \mathcal{C}_N}\ \min_{p' < p}\ \lVert L^{-1}(s - s^{(p')}) \rVert_2.$$

---

## 9. Paramètres centralisés

Tous regroupés dans [`src/config.py`](src/config.py) :

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| `SEED` | `42` | Graine aléatoire de tous les tirages |
| `USE_LEDOIT_WOLF` | `True` | Stabilisation de Sigma par shrinkage Ledoit-Wolf |
| `ENFORCE_STRESS_ONLY_FLOOR` | `False` | Forcer `PD_stress ≥ PD0` et `LGD_stress ≥ LGD0` |
| `G_COL` | `"shock_GPRD"` | Nom de la colonne du choc géopolitique |
| `ETA_LOCAL` | `1.00` | Rayon de la boule locale autour de `s*` |
| `PHI_NEAR` | `1.00` | Marge near-optimal : `d²(s) ≤ d²(s*) + PHI_NEAR` |
| `POOL_SIZE` | `4000` | Tirages par branche avant filtrage et déduplication |
| `SHORTLIST_SIZE` | `8` | Taille de la shortlist finale `C_P` |

Modifier ces constantes ne nécessite aucune intervention sur le reste du
code ; elles sont propagées automatiquement par le module.

---

## 10. Références

- Hurlin, C., Lajaunie, Q., and Pull, Y. (2026).
  *Reverse Stress Testing Geopolitical Risk in Corporate Credit Portfolios :
  A Formal and Operational Framework.* Working paper. Le PDF est disponible
  localement à `Documents/main_reference-RST.pdf`.

- Note rédigée dans le cadre de cette implémentation :
  *Stress Test Inversé du Risque Géopolitique dans les Portefeuilles de
  Crédit aux Entreprises* (Mars 2026, version révisée Mai 2026).

- Pour la construction du portefeuille simulé et les sources de
  calibration (HSBC Pillar 3, S&P, EBA, Basel CRE31, ECB FSR May 2024) :
  voir [`Simulations/README.md`](Simulations/README.md) et la section
  *Construction du portefeuille simulé* de la note.
