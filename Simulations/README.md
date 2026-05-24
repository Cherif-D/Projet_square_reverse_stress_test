# Simulations

Ce dossier contient les scripts qui génèrent les données d'entrée du
pipeline. Comme on n'a pas accès à un vrai portefeuille bancaire interne,
on simule trois fichiers CSV : les expositions, le capital initial et les
coefficients sectoriels de stress.

Les scripts sont lancés au début de `run_project.py`. Ils peuvent aussi
être exécutés seuls depuis la racine du dépôt.

## Ordre des scripts

Il faut lancer les scripts dans cet ordre :

1. `simulate_exposures.py`
2. `simulate_capital.py`
3. `simulate_sector_params.py`

Le deuxième script lit `inputs/exposures.csv`, donc il dépend du premier.
Le troisième ne lit pas de fichier d'entrée.

## `simulate_exposures.py`

Ce script crée le portefeuille simulé au niveau exposition. Il génère 500
lignes réparties sur huit secteurs : `Energy`, `Transport`,
`Manufacturing`, `Consumer`, `Tech`, `Utilities`, `Defense` et
`RealEstate`.

Il utilise la graine `SEED = 42`, donc le fichier généré est reproductible.
Les principaux paramètres sont écrits dans le script : `N_EXPOSURES = 500`,
`TOTAL_EAD = 100.0`, `Q = 0.999`, les poids sectoriels et les profils
sectoriels. Les EAD sont tirées avec une loi lognormale puis normalisées
pour que l'EAD totale fasse 100. Les PD, LGD, corrélations d'actif et
maturités sont ensuite simulées autour de moyennes sectorielles.

La sortie est `inputs/exposures.csv`. Elle contient les colonnes suivantes :
`id`, `sector`, `EAD`, `PD0`, `LGD0`, `rho`, `M`, `alpha_rwa`,
`K0_proxy`, `RW0_exact_proxy`.

Exemple de lignes générées :

```csv
id,sector,EAD,PD0,LGD0,rho,M,alpha_rwa,K0_proxy,RW0_exact_proxy
201,Consumer,0.18727632396730098,0.00556058945989303,0.38371106033479124,0.19094195638334505,2.184185927231544,4.829302567396034,0.03277512182276463,0.4096890227845579
202,Consumer,0.49053752307111537,0.003959815704424318,0.41298320542553335,0.19728747321457493,2.3292563225708816,12.610901881015646,0.029380951741177757,0.367261896764722
```

Les sources citées dans le script servent surtout à cadrer les ordres de
grandeur. Le fichier HSBC Continental Europe Pillar 3 au 31 décembre 2024
sert d'ancre de structure et d'ordre de grandeur corporate. La S&P Global
Ratings 2024 Annual Global Corporate Default and Rating Transition Study
sert de repère qualitatif sur les secteurs plus cycliques ou plus
résilients. Le Risk Dashboard EBA Q3 2024 et le rapport EBA 2023 Credit Risk
Benchmarking donnent un cadrage sur les paramètres corporate IRB et les LGD.
La corrélation `rho` est ancrée sur Basel IRB CRE31 §31.43. Le quantile
`q = 0.999` reprend le cadre IRB Bâle et Gordy (2003).

Les valeurs sectorielles ne sont pas des extractions directes d'une table
publique. Ce sont des proxys calibrés pour obtenir un portefeuille lisible
et cohérent avec les ordres de grandeur cités.

## `simulate_capital.py`

Ce script construit le fichier capital à partir du portefeuille simulé. Il
lit `inputs/exposures.csv`, calcule `RWA_0`, fixe `R0 = 0.14`, puis en déduit
`CET1_0 = R0 * RWA_0`.

Le seuil `R_omega` vaut `0.11`. On l'obtient avec une déplétion absolue de
300 points de base : `0.14 - 0.03 = 0.11`. C'est la convention retenue dans
le code pour coller à l'annonce ECB du 12 décembre 2025 sur une baisse d'au
moins 300 bps du ratio CET1.

La sortie est `inputs/capital.csv`. Elle contient une ligne et huit colonnes :
`CET1_0`, `RWA_0`, `R0`, `R_omega`, `q`, `delta_non_credit`, `Lq0_abs`,
`baseline_convention`.

Exemple du fichier généré :

```csv
CET1_0,RWA_0,R0,R_omega,q,delta_non_credit,Lq0_abs,baseline_convention
5.428508523913509,38.77506088509649,0.14,0.11000000000000001,0.999,0.0,3.2036103423714497,use_delta_Lq_in_main_script
```

Le script calcule aussi `Lq0_abs`, qui est la perte de queue au point de
départ. Cette valeur est un diagnostic. Dans le pipeline principal, le CET1
est mis à jour avec `delta_Lq(s) = Lq(s) - Lq(0)`. Cela permet de garder
`R(0) = R0` au point de départ.

Les sources citées dans le script sont les ratios CET1 agrégés EBA/ECB
2024-2025 pour cadrer `R0`, l'annonce ECB du 12 décembre 2025 pour la
déplétion de 300 bps, Basel IRB CRE31 et Gordy (2003) pour `q = 0.999`, et
le papier Hurlin, Lajaunie, Pull pour les équations de capital.

## `simulate_sector_params.py`

Ce script génère les coefficients sectoriels qui relient le scénario
`s = (g, x)` aux PD et LGD stressées. Il n'utilise pas de tirage aléatoire.
Toutes les valeurs sont écrites dans le script.

La sortie est `inputs/sector_params.csv`. Le fichier a huit lignes, une par
secteur, et vingt colonnes. Les premières colonnes documentent le statut des
paramètres : `param_origin = stylized`, `calibration_status =
not_empirically_estimated`, et une note indiquant que la hiérarchie est
économiquement motivée mais que les valeurs numériques restent stylisées.

Les colonnes `delta_g` et `eta_g` donnent les sensibilités directes au choc
géopolitique. Les colonnes `b_shock_*` donnent les effets macro sur la PD.
Les colonnes `c_shock_*` donnent les effets macro sur la LGD. Les variables
macro sont `shock_gdp`, `shock_vix`, `shock_sp500`, `shock_wti`,
`shock_t10Y2Y`, `shock_unrate` et `shock_epu`.

Exemple de lignes générées :

```csv
sector,param_origin,calibration_status,note,delta_g,eta_g,b_shock_gdp,b_shock_vix,b_shock_sp500,b_shock_wti,b_shock_t10Y2Y,b_shock_unrate,b_shock_epu,c_shock_gdp,c_shock_vix,c_shock_sp500,c_shock_wti,c_shock_t10Y2Y,c_shock_unrate,c_shock_epu
Energy,stylized,not_empirically_estimated,"sector hierarchy economically grounded, numeric values stylized",0.65,0.1,-0.18,0.16,-0.05,-0.1,-0.03,0.12,0.14,-0.02,0.02,-0.01,-0.01,0.0,0.02,0.02
Transport,stylized,not_empirically_estimated,"sector hierarchy economically grounded, numeric values stylized",0.65,0.1,-0.32,0.26,-0.08,0.18,-0.03,0.28,0.14,-0.05,0.03,-0.01,0.03,0.0,0.03,0.02
```

La hiérarchie géopolitique est appuyée sur l'ECB Financial Stability Review
de mai 2024, spécial feature sur le risque géopolitique et la stabilité
financière. Le script précise aussi que l'ECB Working Paper No. 2897 existe,
mais que ce n'est pas la source sectorielle géopolitique utilisée ici. Les
résultats du stress test EBA 2025 et le scénario ESRB 2025 servent à
confirmer l'hétérogénéité sectorielle. La S&P 2024 Annual Default Study sert
de cross-check qualitatif. Hurlin, Lajaunie, Pull donnent la forme des
transmissions PD/LGD avec les équations 7 à 10.

On a fait attention à garder le statut des coefficients clair. Ils ne sont
pas estimés empiriquement. Dans une vraie application bancaire, il faudrait
les recalibrer avec des données internes, par exemple avec une régression sur
panel pour les PD et une calibration bornée pour les LGD.

## Lancer la génération

Pour relancer seulement les trois fichiers d'entrée :

```bash
python Simulations/simulate_exposures.py
python Simulations/simulate_capital.py
python Simulations/simulate_sector_params.py
```

Pour relancer tout le pipeline, y compris les simulations :

```bash
python run_project.py
```
