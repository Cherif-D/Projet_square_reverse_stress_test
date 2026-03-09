from __future__ import annotations
import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ============================================================
# POINT D'ENTREE DU PROJET
# ============================================================
#
# Ce script orchestre l'ensemble du projet en quatre phases :
#
# 0. Simulations — génération des inputs (Simulations/)
#    - simulate_exposures.py     → inputs/exposures.csv
#    - simulate_capital.py       → inputs/capital.csv
#    - simulate_sector_params.py → inputs/sector_params.csv
#
# 1. Pipeline RST principal (src/run_rst.py)
#    - Construction du scénario (Sigma)
#    - Chargement des inputs crédit / capital
#    - Calcul PD/LGD stressés
#    - Optimisation du design point
#    - Génération du pool candidat et de la shortlist
#    - Sauvegarde de toutes les tables et rapports
#
# 2. Visualisations — legacy (visualize_frontier.py, visualize_results.py)
#    - frontier_plot.png       : frontière de rupture (original)
#    - sector_impact.png       : pertes sectorielles (original)
#
# 3. Visualisations — src/visualization/ (graphiques du papier)
#    - fig1_geometry.png       : Figure 1 — Géométrie RST (S_red, frontière, boule locale)
#    - fig2_near_optimal.png   : Figure 2 — Ensemble N_epsilon near-optimal
#    - fig3_roadmap.png        : Figure 3 — Feuille de route d'implémentation
#    - fig4_scenario_selection.png : Figure 4 — Sélection farthest-point
#    - fig5_plausibility.png   : Figure 5 — Gaussienne vs. Student-t (heatmaps)
#    - scenario_profile.png    : Profil du design point (bar chart corrigé)
#    - sector_impact.png       : Pertes de queue par secteur
#    - sector_pd_lgd.png       : Bulles PD_stress vs LGD_stress par secteur
#
# USAGE
# -----
#   python run_project.py
#
# À exécuter depuis la racine du projet.
# ============================================================

import sys
import importlib
from pathlib import Path

# On s'assure que la racine du projet est dans le path Python.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_simulations():
    """
    Étape 0 — Génération des fichiers inputs via les scripts Simulations/.

    Ordre obligatoire :
      1. simulate_exposures.py     → inputs/exposures.csv
      2. simulate_capital.py       → inputs/capital.csv  (dépend d'exposures.csv)
      3. simulate_sector_params.py → inputs/sector_params.csv

    Note : les scripts sont lancés via subprocess.run() (et non importlib)
    car leur code s'exécute au niveau module. Le subprocess garantit une
    exécution unique et propre, comme si on faisait `python script.py`.
    """
    import subprocess

    print("=" * 60)
    print("ETAPE 0 — Génération des inputs (Simulations/)")
    print("=" * 60)

    sim_dir = ROOT / "Simulations"

    scripts = [
        ("simulate_exposures.py",     "inputs/exposures.csv"),
        ("simulate_capital.py",       "inputs/capital.csv"),
        ("simulate_sector_params.py", "inputs/sector_params.csv"),
    ]

    for script_name, output_hint in scripts:
        script_path = sim_dir / script_name
        print(f"\n--- {script_name}  ->  {output_hint} ---")
        try:
            env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(ROOT),
                check=True,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            print(f"  Erreur {script_name} (code {e.returncode})")
            raise


def run_pipeline():
    """Lance le pipeline RST principal via src/run_rst.py."""
    print()
    print("=" * 60)
    print("ETAPE 1 — Pipeline Reverse Stress Test")
    print("=" * 60)
    from src.run_rst import main
    main()


def run_legacy_visualizations():
    """
    Lance les scripts de visualisation legacy (visualize_frontier.py,
    visualize_results.py) — conservés sans modification.
    """
    print()
    print("=" * 60)
    print("ETAPE 2 — Visualisations legacy (visualize_frontier / visualize_results)")
    print("=" * 60)

    print("\n--- Frontière de rupture (legacy) ---")
    try:
        import visualize_frontier
        importlib.reload(visualize_frontier)
        visualize_frontier.plot_frontier()
    except Exception as e:
        print(f"  Avertissement visualize_frontier : {e}")

    print("\n--- Profil et impact sectoriel (legacy) ---")
    try:
        import visualize_results
        importlib.reload(visualize_results)
        visualize_results.plot_scenario()
        visualize_results.plot_sector_impact()
    except Exception as e:
        print(f"  Avertissement visualize_results : {e}")


def run_src_visualizations():
    """
    Lance tous les graphiques du papier depuis src/visualization/.
    """
    print()
    print("=" * 60)
    print("ETAPE 3 — Graphiques du papier (src/visualization/)")
    print("=" * 60)

    from src.visualization.fig1_geometry import plot_fig1_geometry
    from src.visualization.fig2_near_optimal import plot_fig2_near_optimal
    from src.visualization.fig3_roadmap import plot_fig3_roadmap
    from src.visualization.fig4_scenario_selection import plot_fig4_scenario_selection
    from src.visualization.fig5_plausibility import plot_fig5_plausibility
    from src.visualization.scenario_profile import plot_scenario_profile
    from src.visualization.sector_impact import plot_sector_losses, plot_sector_pd_lgd

    viz_steps = [
        ("\n--- Figure 3 : Feuille de route (flowchart) ---",      plot_fig3_roadmap),
        ("\n--- Figure 1 : Géométrie RST ---",                      plot_fig1_geometry),
        ("\n--- Figure 2 : Ensemble N_epsilon ---",                  plot_fig2_near_optimal),
        ("\n--- Figure 4 : Sélection de scénarios finis ---",        plot_fig4_scenario_selection),
        ("\n--- Figure 5 : Plausibilité Gaussienne vs Student-t ---", plot_fig5_plausibility),
        ("\n--- Profil du scénario de rupture ---",                  plot_scenario_profile),
        ("\n--- Pertes de queue par secteur ---",                    plot_sector_losses),
        ("\n--- PD_stress vs LGD_stress par secteur ---",           plot_sector_pd_lgd),
    ]

    for label, fn in viz_steps:
        print(label)
        try:
            fn()
        except Exception as e:
            print(f"  Avertissement : {e}")


if __name__ == "__main__":
    run_simulations()
    run_pipeline()
    run_legacy_visualizations()
    run_src_visualizations()
    print()
    print("Projet terminé. Consulter outputs/plots/ pour tous les graphiques.")
