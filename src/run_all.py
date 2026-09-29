"""
Orquestador del pipeline completo del Parcial 1.

Ejecuta en orden las 5 fases del protocolo experimental obligatorio:
  Fase 1: Evaluación de la red base (30 épocas).
  Fase 2: Optimización mediante pyGAD (pop>=10, gens>=10, 5 épocas/individuo).
  Fase 3: Selección del mejor individuo (incluida en ga_optimize.run_ga).
  Fase 4: Reentrenamiento del modelo optimizado (30 épocas).
  Fase 5: Evaluación final sobre Test y generación de gráficas/tabla.

Es resumible: si una fase ya tiene sus resultados guardados en disco, se
omite (a menos que se use --force), lo cual es útil porque el proceso completo
puede tardar horas en CPU y podría interrumpirse.
"""
import os
import sys
import json
import argparse

from . import config
from .baseline_model import train_and_evaluate_baseline
from .ga_optimize import run_ga
from .retrain_best import retrain_and_evaluate_best
from .compare_report import generate_all

BASELINE_RESULTS_PATH = os.path.join(config.BASELINE_DIR, "baseline_results.json")
GA_SUMMARY_PATH = os.path.join(config.GA_DIR, "ga_summary.json")
FINAL_RESULTS_PATH = os.path.join(config.FINAL_DIR, "final_results.json")


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo Parcial 1")
    parser.add_argument("--force", action="store_true", help="Reejecutar todas las fases aunque existan resultados")
    parser.add_argument("--baseline-epochs", type=int, default=config.BASELINE_EPOCHS)
    parser.add_argument("--population", type=int, default=config.GA_POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=config.GA_NUM_GENERATIONS)
    parser.add_argument("--individual-epochs", type=int, default=config.GA_INDIVIDUAL_EPOCHS)
    parser.add_argument("--final-epochs", type=int, default=config.FINAL_RETRAIN_EPOCHS)
    args = parser.parse_args()

    print("=" * 70)
    print("FASE 1: Evaluación de la Red Base")
    print("=" * 70)
    if args.force or not os.path.exists(BASELINE_RESULTS_PATH):
        train_and_evaluate_baseline(epochs=args.baseline_epochs)
    else:
        print("Ya existe baseline_results.json; se omite (usa --force para reejecutar).")

    print("=" * 70)
    print("FASE 2 y 3: Optimización mediante pyGAD y selección del mejor individuo")
    print("=" * 70)
    if args.force or not os.path.exists(GA_SUMMARY_PATH):
        run_ga(
            population_size=args.population,
            num_generations=args.generations,
            individual_epochs=args.individual_epochs,
        )
    else:
        print("Ya existe ga_summary.json; se omite (usa --force para reejecutar).")

    print("=" * 70)
    print("FASE 4 y 5: Reentrenamiento del modelo optimizado y evaluación final")
    print("=" * 70)
    if args.force or not os.path.exists(FINAL_RESULTS_PATH):
        retrain_and_evaluate_best(epochs=args.final_epochs)
    else:
        print("Ya existe final_results.json; se omite (usa --force para reejecutar).")

    print("=" * 70)
    print("Generando gráficas y tabla comparativa")
    print("=" * 70)
    generate_all()

    print("\nPipeline completo. Resultados en:", config.RESULTS_DIR)
    print("Gráficas del reporte en:", config.REPORT_DIR)


if __name__ == "__main__":
    main()
