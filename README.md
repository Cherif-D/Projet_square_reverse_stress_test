# Reverse Stress Test Géopolitique

Implémentation Python du **Reverse Stress Test (RST)** géopolitique, basée sur le papier :

> Hurlin, Lajaunie, Pull (2026) — *A Reverse Stress Test Framework with Geopolitical Scenarios*

L'objectif est de trouver le **scénario de choc le plus plausible** qui ferait tomber le ratio de capital d'une banque en dessous d'un seuil réglementaire critique.

---

## Table des matières

1. [Contexte et intuition](#1-contexte-et-intuition)
2. [Prérequis et installation](#2-prérequis-et-installation)
3. [Structure du projet](#3-structure-du-projet)
4. [Données d'entrée](#4-données-dentrée)
5. [Pipeline complet étape par étape](#5-pipeline-complet-étape-par-étape) ← Étape 0 : Simulations → Étape 3 : Graphiques
6. [Formules mathématiques clés](#6-formules-mathématiques-clés)
7. [Sorties produites](#7-sorties-produites)
8. [Graphiques générés](#8-graphiques-générés)
9. [Paramètres de configuration](#9-paramètres-de-configuration)
10. [Lancer le projet](#10-lancer-le-projet)

---

## 1. Contexte et intuition

### Qu'est-ce qu'un Reverse Stress Test ?

Dans un stress test classique, on part d'un scénario macroéconomique adverse (crise, récession...) et on calcule l'impact sur le capital de la banque.

Dans un **Reverse Stress Test**, on inverse la logique :

> *"Quel est le scénario le plus petit (le plus plausible) qui suffit à faire tomber notre ratio de capital en dessous du seuil critique ?"*

### Le scénario géopolitique

Le choc principal est un choc géopolitique $g$ (mesuré par l'indice **GPRD** — Geopolitical Risk Daily), accompagné de chocs macro $x$ (PIB, VIX, chômage...). Ensemble, ils forment le vecteur de scénario :

$$s = (g,\, x_1,\, x_2,\, \ldots,\, x_7) \in \mathbb{R}^8 \quad \text{(z-scores)}$$

### Le design point $s^*$

Le **design point** $s^*$ est la solution du RST : le scénario de rupture le plus plausible. Il est à la fois :
- Sur la **frontière de rupture** : $R(s^*) = \bar{R}$ (ratio CET1/RWA exactement au seuil)
- Le **plus proche de la baseline** $s = 0$ au sens de Mahalanobis (le moins extrême statistiquement)

---

## 2. Prérequis et installation

**Python requis : 3.12+**

```bash
# Cloner ou ouvrir le dossier du projet
cd Reverse_stress_test

# Créer l'environnement virtuel
python -m venv env

# Activer l'environnement (Windows)
env\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 3. Structure du projet

```
Reverse_stress_test/
│
├── run_project.py              ← Point d'entrée unique (lancer ça — 4 étapes)
│
├── Simulations/                ← Étape 0 : génération des inputs
│   ├── simulate_exposures.py   ← Génère inputs/exposures.csv (500 emprunteurs)
│   ├── simulate_capital.py     ← Génère inputs/capital.csv (connecté au portefeuille)
│   └── simulate_sector_params.py ← Génère inputs/sector_params.csv (coefficients stylisés)
│
├── inputs/                     ← Générés automatiquement par Simulations/
│   ├── exposures.csv           ← Portefeuille de crédit (1 ligne = 1 emprunteur)
│   ├── capital.csv             ← Paramètres de capital (R0, R_omega, q...)
│   └── sector_params.csv       ← Sensibilités sectorielles aux chocs
│
├── data/
│   └── raw/
│       └── macro.csv           ← Séries macro historiques (GPRD, PIB, VIX...)
│
├── outputs/                    ← Tout ce que le projet génère
│   ├── plots/                  ← 9 graphiques PNG (300 dpi)
│   ├── tables/                 ← Résultats CSV (design point, pool, shortlist...)
│   └── reports/                ← Résumé JSON + Markdown + contexte
│
├── src/                        ← Code modulaire principal
│   ├── config.py               ← Constantes globales
│   ├── paths.py                ← Chemins centralisés
│   ├── run_rst.py              ← Orchestrateur du pipeline
│   │
│   ├── scenario/               ← Construction du scénario standardisé
│   │   ├── transforms.py       ← Transformations (logdiff, diff)
│   │   ├── covariance.py       ← Estimation Sigma (Ledoit-Wolf)
│   │   └── build_scenario.py   ← Construction complète du scénario
│   │
│   ├── portfolio/              ← Lecture et agrégation du portefeuille
│   │   ├── loaders.py          ← Chargement et validation des inputs
│   │   └── sector_aggregation.py ← Agrégation par secteur
│   │
│   ├── model/                  ← Modèle de pertes et de capital
│   │   ├── stress_mappings.py  ← Fonctions de transmission PD et LGD sous stress
│   │   ├── loss_model.py       ← Calcul de Lq, CET1, RWA, ratio R
│   │   ├── capital_model.py    ← Formules pures CET1, RWA, ratio
│   │   └── baseline.py         ← Diagnostic du contexte initial
│   │
│   ├── optimization/           ← Résolution du RST
│   │   ├── design_point.py     ← Optimisation SLSQP multi-start → s*
│   │   ├── candidate_sets.py   ← Génération du pool de candidats C_N
│   │   └── shortlist.py        ← Sélection farthest-point → shortlist C_P
│   │
│   ├── reporting/              ← Sauvegarde des résultats
│   │   ├── save_tables.py      ← Export des CSV
│   │   └── save_reports.py     ← Export du résumé (JSON + Markdown)
│   │
│   └── visualization/          ← Graphiques
│       ├── helpers.py          ← Fonctions partagées (grille, ratio, distance)
│       ├── fig1_geometry.py    ← Fig 1 papier : géométrie RST
│       ├── fig2_near_optimal.py ← Fig 2 papier : ensemble N_epsilon
│       ├── fig3_roadmap.py     ← Fig 3 papier : roadmap implémentation
│       ├── fig4_scenario_selection.py ← Fig 4 papier : sélection finie
│       ├── fig5_plausibility.py ← Fig 5 papier : Gaussienne vs Student-t
│       ├── scenario_profile.py ← Profil des chocs au design point
│       └── sector_impact.py    ← Impact sectoriel (pertes + PD/LGD)
│
├── MVP.py                      ← Script monolithique original (conservé)
├── visualize_frontier.py       ← Visualisation frontière (original, conservé)
├── visualize_results.py        ← Visualisation résultats (original, conservé)
│
└── requirements.txt            ← Dépendances Python
```

---

## 4. Données d'entrée

> Les fichiers `inputs/` sont **générés automatiquement** par les scripts `Simulations/` à l'étape 0 du pipeline. Il n'est pas nécessaire de les préparer manuellement.

### `inputs/exposures.csv` — Portefeuille de crédit

Généré par `simulate_exposures.py` — 500 emprunteurs répartis sur 8 secteurs, avec des paramètres calibrés sur les formules Bâle IRB (CRE31). Colonnes principales :

| Colonne | Description |
|---------|-------------|
| `id` | Identifiant unique de l'emprunteur |
| `sector` | Secteur économique (Energy, Tech, Consumer...) |
| `EAD` | Exposure At Default — montant exposé (Mds€) |
| `PD0` | Probabilité de défaut initiale $PD_0 \in (0,1)$ |
| `LGD0` | Perte en cas de défaut initiale $LGD_0 \in (0,1)$ |
| `rho` | Corrélation systémique $\rho$ (modèle Vasicek) |
| `M` | Maturité effective |
| `alpha_rwa` | Coefficient de pondération RWA $\alpha$ |

### `inputs/capital.csv` — Paramètres de capital

Généré par `simulate_capital.py` — les valeurs de $\text{CET1}_0$ et $\text{RWA}_0$ sont **dérivées du portefeuille** simulé (pas fixées arbitrairement), ce qui garantit la cohérence interne.

| Colonne | Description |
|---------|-------------|
| `CET1_0` | Capital CET1 initial (Mds€) — déduit de $R_0 \times \text{RWA}_0$ |
| `RWA_0` | Actifs pondérés initiaux (Mds€) — calculé depuis `exposures.csv` |
| `R0` | Ratio $R_0 = \text{CET1}_0 / \text{RWA}_0 = 14\%$ |
| `R_omega` | Seuil de rupture $\bar{R} = 11\%$ |
| `q` | Quantile de queue $q = 99.9\%$ |
| `Lq0_abs` | Perte de queue absolue $L_q(0)$ à la baseline |
| `baseline_convention` | Convention de correction baseline utilisée dans le pipeline |

### `inputs/sector_params.csv` — Sensibilités sectorielles

Généré par `simulate_sector_params.py` — coefficients **stylisés** documentés (sources ECB, EBA, ESRB). Chaque secteur a ses propres coefficients de transmission géopolitique :

| Colonne | Description |
|---------|-------------|
| `delta_g` | Sensibilité directe de $PD$ au choc géopolitique $g$ |
| `eta_g` | Sensibilité directe de $LGD$ au choc géopolitique $g$ |
| `b_shock_*` | Coefficients $b_j$ de transmission des chocs macro $\to PD$ |
| `c_shock_*` | Coefficients $c_j$ de transmission des chocs macro $\to LGD$ |

### `data/raw/macro.csv` — Séries macro historiques

Séries temporelles trimestrielles des 8 variables macroéconomiques :
`GPRD`, `gdp`, `vix`, `sp500`, `wti`, `t10Y2Y`, `unrate`, `epu`

---

## 5. Pipeline complet étape par étape

### Étape 0 — Génération des inputs (Simulations/)

**Fichiers :** `Simulations/simulate_exposures.py`, `simulate_capital.py`, `simulate_sector_params.py`

Ces trois scripts génèrent les données d'entrée du modèle **dans un ordre précis** (car `capital.csv` dépend d'`exposures.csv`) :

| Ordre | Script | Sortie | Contenu |
|-------|--------|--------|---------|
| 1 | `simulate_exposures.py` | `inputs/exposures.csv` | 500 emprunteurs, 8 secteurs, paramètres calibrés Bâle IRB |
| 2 | `simulate_capital.py` | `inputs/capital.csv` | $\text{RWA}_0$ dérivé du portefeuille, $R_0 = 14\%$, $\bar{R} = 11\%$ |
| 3 | `simulate_sector_params.py` | `inputs/sector_params.csv` | Coefficients $\delta_g$, $\eta_g$, $b_j$, $c_j$ par secteur |

Les paramètres sont **stylisés mais économiquement fondés** (hiérarchie sectorielle documentée : ECB WP 2897, EBA Stress Test 2025, ESRB 2025).

---

### Étape 1 — Construction du scénario standardisé

**Fichier :** `src/scenario/build_scenario.py`

On transforme les séries macro brutes en **chocs standardisés** (z-scores) :

1. **Transformation** : $\log$-différence pour les variables de niveau (PIB, SP500...), différence simple pour les indices (VIX, taux...)
2. **Standardisation** : on centre et réduit chaque série — si $\tilde{s}_t$ est la série transformée, le choc est $s_t = (\tilde{s}_t - \mu) / \sigma$
3. **Recentrage** : les valeurs du dernier trimestre connu deviennent la baseline $s = 0$
4. **Estimation de $\Sigma$** : matrice de covariance des chocs, stabilisée par **Ledoit-Wolf shrinkage** pour éviter les problèmes numériques sur de petits échantillons

Résultat : vecteur $s \in \mathbb{R}^8$ avec une matrice $\Sigma \in \mathbb{R}^{8 \times 8}$ décrivant les corrélations entre chocs.

---

### Étape 2 — Modèle de pertes sous stress

**Fichiers :** `src/model/stress_mappings.py`, `src/model/loss_model.py`

Sous un scénario $s = (g, x)$, les paramètres de crédit de chaque emprunteur $i$ sont modifiés.

**PD stressée** (équation 7 du papier) :

$$PD_{\text{stress}}(i) = PD_0(i) \cdot \sigma\!\left(\delta_g(i) \cdot g + \sum_j b_j(i) \cdot x_j\right)$$

où $\sigma(\cdot)$ est la fonction sigmoïde permettant de rester dans $(0, 1)$.

**LGD stressée** (équation 10 du papier) :

$$LGD_{\text{stress}}(i) = \phi\!\left(LGD_0(i),\; \eta_g(i) \cdot g + \sum_j c_j(i) \cdot x_j\right)$$

où $\phi$ est une transformation lisse qui préserve l'intervalle $(0, 1)$.

**Perte de queue par emprunteur** via le **modèle Vasicek / ASRF** au quantile $q = 99.9\%$ :

$$L_q(i,s) = EAD_i \cdot LGD_{\text{stress}}(i) \cdot \Phi\!\left(\frac{\Phi^{-1}\!\left(PD_{\text{stress}}(i)\right) + \sqrt{\rho_i}\,\Phi^{-1}(q)}{\sqrt{1 - \rho_i}}\right)$$

**Ratio de capital** sous scénario $s$ :

$$R(s) = \frac{CET1(s)}{RWA(s)} \qquad \text{avec} \quad RWA(s) = \sum_i \frac{\alpha_i \cdot K_i(s)}{0.08}$$

La **correction baseline** garantit $R(0) = R_0$ exactement :

$$\Delta L_q(s) = L_q(s) - L_q(0)$$

---

### Étape 3 — Résolution du Reverse Stress Test

**Fichier :** `src/optimization/design_point.py`

On résout le problème d'optimisation contraint :

$$\min_{s \in \mathbb{R}^d} \quad \frac{1}{2}\, s^\top \Sigma^{-1} s$$

$$\text{s.t.} \quad R(s) \leq \bar{R} \quad \text{et} \quad g \geq 0$$

**Méthode** : SLSQP (Sequential Least Squares Programming) avec **multi-start** (26 points de départ : aléatoires + grille d'intensités). On garde le meilleur optimum parmi tous les points faisables.

En pratique, on travaille dans l'**espace blanchi** $y = L^{-1}s$ (décomposition de Cholesky $\Sigma = LL^\top$). La distance de Mahalanobis se simplifie alors en norme euclidienne :

$$d^2(s) = s^\top \Sigma^{-1} s = \|y\|^2 = \|L^{-1}s\|^2$$

---

### Étape 4 — Génération du pool de candidats

**Fichier :** `src/optimization/candidate_sets.py`

Au-delà du design point unique, on génère un **pool de scénarios plausibles** à partir de deux ensembles :

- **Boule locale** $\mathcal{B}_\eta(s^*)$ : $\|L^{-1}(s - s^*)\|^2 \leq \eta$ — scénarios proches de $s^*$ en espace blanchi
- **Ensemble near-optimal** $\mathcal{N}_\varphi$ : $d^2(s) \leq d^2(s^*) + \varphi$ — scénarios presque aussi plausibles que $s^*$

Seuls les scénarios dans la **zone de rupture** $\mathcal{S}_{\text{red}} = \{s : R(s) \leq \bar{R}\}$ sont conservés.

---

### Étape 5 — Shortlist farthest-point

**Fichier :** `src/optimization/shortlist.py`

On réduit le pool à **8 scénarios gouvernance-ready** via l'algorithme **farthest-point maximin** (équation 49 du papier) :

$$s^{(p)} = \underset{s \in \mathcal{C}_N}{\arg\max} \; \min_{p' < p} \left\| L^{-1}\!\left(s - s^{(p')}\right) \right\|_2$$

À chaque itération, on choisit le scénario le plus éloigné de tous ceux déjà sélectionnés. Cela garantit **diversité** et **couverture maximale** de la frontière de rupture.

---

### Étape 6 — Rapports et tables

**Fichier :** `src/reporting/`

Sauvegarde de tous les résultats : design point, diagnostics sectoriels, pool, shortlist, résumé global.

---

### Étape 7 — Visualisation

**Fichier :** `src/visualization/`

9 graphiques générés automatiquement dans `outputs/plots/`.

---

## 6. Formules mathématiques clés

| Symbole | Signification |
|---------|--------------|
| $s \in \mathbb{R}^d$ | Vecteur de scénario ($d = 8$ variables en z-scores) |
| $g = s_1$ | Choc géopolitique (`shock_GPRD`) |
| $\Sigma \in \mathbb{R}^{d \times d}$ | Matrice de covariance des chocs, estimée par Ledoit-Wolf |
| $d^2(s) = s^\top \Sigma^{-1} s$ | Distance de Mahalanobis au carré — mesure de plausibilité |
| $y = L^{-1}s$ | Espace blanchi ($L$ = facteur de Cholesky de $\Sigma$) |
| $R(s) = \text{CET1}(s) / \text{RWA}(s)$ | Ratio de capital sous scénario $s$ |
| $\bar{R} = 11\%$ | Seuil de rupture réglementaire |
| $R_0 = 14\%$ | Ratio de capital à la baseline |
| $q = 99.9\%$ | Quantile de queue (modèle Vasicek) |
| $s^*$ | Design point — solution du RST |
| $\mathcal{N}_\varepsilon = \mathcal{S}_{\text{red}} \cap \{d^2(s) \leq d^2(s^*) + \varepsilon\}$ | Ensemble near-optimal |
| $\mathcal{B}_\rho(s^*) = \{\|L^{-1}(s - s^*)\|^2 \leq \rho\}$ | Boule locale autour de $s^*$ |

---

## 7. Sorties produites

### Tables CSV (`outputs/tables/`)

| Fichier | Contenu |
|---------|---------|
| `Sigma.csv` | Matrice de covariance $\Sigma$ (8×8) |
| `scenario_standardized.csv` | Séries macro en z-scores |
| `baseline_exposure_metrics.csv` | Métriques de crédit à la baseline par emprunteur |
| `design_point.csv` | Coordonnées de $s^*$ et métriques associées |
| `exposure_stress_at_design_point.csv` | $PD_{\text{stress}}$, $LGD_{\text{stress}}$, $L_q$ par emprunteur en $s^*$ |
| `sector_diagnostics_at_design_point.csv` | Agrégats sectoriels (EAD, PD, LGD, $L_q$) en $s^*$ |
| `candidate_pool.csv` | Pool de scénarios candidats $\mathcal{C}_N$ |
| `scenario_shortlist.csv` | Shortlist finale $\mathcal{C}_P$ (8 scénarios) |

### Rapports (`outputs/reports/`)

| Fichier | Contenu |
|---------|---------|
| `summary.json` | Tous les indicateurs clés (baseline, design point, ensembles) |
| `summary.md` | Version lisible du résumé |
| `sigma_report.json` | Détails de l'estimation de $\Sigma$ |
| `tutor_data_context.json` | Contexte complet des données pour auditabilité |

### Résultats chiffrés (run de référence)

```
Ratio baseline        R₀  = 14.00%
Seuil de rupture      R̄   = 11.00%
Ratio au design point R*  = 11.00%   ← frontière exacte

Design point s* (z-scores) :
  shock_GPRD   = +0.1845   ← choc géopolitique
  shock_gdp    = -0.1653   ← baisse PIB
  shock_unrate = +0.1312   ← hausse chômage
  shock_vix    = +0.1258   ← hausse volatilité

d²(s*) = 0.0765            ← très petit : scénario très plausible
p-value = 99.99999%         ← non extrême statistiquement

Pool de candidats : 538 scénarios
Shortlist finale  :   8 scénarios
```

---

## 8. Graphiques générés

Tous dans `outputs/plots/` au format PNG 300 dpi.

| Fichier | Description |
|---------|-------------|
| `fig1_geometry.png` | **Fig 1** — Zone $\mathcal{S}_{\text{red}}$, frontière $R(s)=\bar{R}$, ellipses de Mahalanobis, $s^*$, boule $\mathcal{B}_\rho(s^*)$ |
| `fig2_near_optimal.png` | **Fig 2** — Ensemble near-optimal $\mathcal{N}_\varepsilon$ hachuré, niveau $d^2(s^*)+\varepsilon$ |
| `fig3_roadmap.png` | **Fig 3** — Roadmap d'implémentation en 4 étapes (flowchart) |
| `fig4_scenario_selection.png` | **Fig 4** — Pool $\mathcal{C}_N$ → ancres géopolitiques $g_j$ → shortlist farthest-point $\mathcal{C}_P$ |
| `fig5_plausibility.png` | **Fig 5** — Heatmaps $-\log_{10}(p\text{-valeur})$ : Gaussienne vs Student-$t$ ($\nu=6$) |
| `scenario_profile.png` | Profil de $s^*$ — chocs en z-scores par variable |
| `sector_impact.png` | $L_q$ par secteur au design point $s^*$ |
| `sector_pd_lgd.png` | Graphique à bulles $PD_{\text{stress}}$ vs $LGD_{\text{stress}}$ par secteur (taille $\propto$ EAD) |
| `frontier_plot.png` | Visualisation legacy de la frontière (script original) |

---

## 9. Paramètres de configuration

Tous dans `src/config.py` :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `SEED` | `42` | Graine aléatoire (reproductibilité) |
| `USE_LEDOIT_WOLF` | `True` | Stabilisation de $\Sigma$ par Ledoit-Wolf shrinkage |
| `ENFORCE_STRESS_ONLY_FLOOR` | `False` | Forcer $PD_{\text{stress}} \geq PD_0$ et $LGD_{\text{stress}} \geq LGD_0$ |
| `G_COL` | `"shock_GPRD"` | Nom de la colonne du choc géopolitique $g$ |
| `ETA_LOCAL` | `1.0` | Rayon $\eta$ de la boule locale $\mathcal{B}_\eta(s^*)$ |
| `PHI_NEAR` | `1.0` | Marge near-optimal $\varphi$ : $d^2(s) \leq d^2(s^*) + \varphi$ |
| `POOL_SIZE` | `4000` | Nombre de candidats générés dans $\mathcal{C}_N$ |
| `SHORTLIST_SIZE` | `8` | Taille $P$ de la shortlist finale $\mathcal{C}_P$ |

---

## 10. Lancer le projet

### Lancement complet (recommandé)

```bash
# Depuis le dossier du projet, avec l'environnement activé
env\Scripts\python.exe run_project.py
```

Ce script exécute **4 étapes dans l'ordre** :

| Étape | Fonction | Ce qui est produit |
|-------|----------|--------------------|
| **0** | `run_simulations()` | `inputs/exposures.csv`, `capital.csv`, `sector_params.csv` |
| **1** | `run_pipeline()` | Tables CSV, rapports JSON/Markdown, design point $s^*$ |
| **2** | `run_legacy_visualizations()` | `frontier_plot.png` (script original) |
| **3** | `run_src_visualizations()` | 8 figures du papier (`fig1` à `fig5` + profil + secteurs) |

### Lancement du pipeline seul (sans regénérer les inputs)

```bash
env\Scripts\python.exe -m src.run_rst
```

### Regénérer uniquement les inputs

```bash
env\Scripts\python.exe Simulations/simulate_exposures.py
env\Scripts\python.exe Simulations/simulate_capital.py
env\Scripts\python.exe Simulations/simulate_sector_params.py
```

### Lancement d'un graphique seul

```bash
env\Scripts\python.exe -m src.visualization.fig1_geometry
env\Scripts\python.exe -m src.visualization.fig5_plausibility
```

### Lancement du script original (MVP)

```bash
env\Scripts\python.exe MVP.py
```

---

## Référence bibliographique

```
Hurlin, C., Lajaunie, C., & Pull, K. (2026).
A Reverse Stress Test Framework with Geopolitical Scenarios.
Documents/main_reference-RST.pdf
```

---

*Projet réalisé dans le cadre du Master 1 ESA — Programme Square*
