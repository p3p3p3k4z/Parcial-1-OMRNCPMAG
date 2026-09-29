"""
ejecutar_todo.py - Orquestador del Pipeline Completo

Proposito:
Ejecutar de forma ordenada y controlada las 5 fases del experimento.

Glosario:
- Fase 1 (Red Base): Entrena la arquitectura estandar como punto de partida.
- Fase 2 y 3 (Algoritmo Genetico): Busca y selecciona la mejor arquitectura.
- Fase 4 (Reentrenamiento): Entrena la mejor arquitectura desde cero con 
  las mismas epocas que la red base para una comparacion justa.
- Fase 5 (Reporte): Evalua ambos modelos en Test y genera graficas.
- Reanudabilidad: Si el proceso se interrumpe, este script permite continuar 
  desde donde se quedo sin repetir fases ya completadas.
"""
import os
import sys
import argparse

from . import config
from .modelo_base import entrenar_y_evaluar_base
from .algoritmo_genetico import ejecutar_ga
from .reentrenamiento import reentrenar_y_evaluar_mejor
from .comparativa import generate_all

CENTINELA_BASELINE = os.path.join(config.DIRECTORIO_BASE_, "resultados_base.json")
CENTINELA_GA       = os.path.join(config.DIRECTORIO_GA,    "resumen_ga.json")
CENTINELA_FINAL    = os.path.join(config.DIRECTORIO_FINAL, "resultados_optimizado.json")


def _encabezado(titulo, fase_num, fase_total=5):
    """Imprime un separador visual para identificar el inicio de cada fase."""
    print(f"\nFASE {fase_num}/{fase_total}: {titulo}")


def main():
    """Punto de entrada principal: parsea argumentos y ejecuta el pipeline."""
    parser = argparse.ArgumentParser(
        description="Pipeline completo de optimizacion de CNN con AG"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ejecuta todas las fases aunque ya existan resultados en disco"
    )
    parser.add_argument(
        "--epocas-base", type=int, default=config.BASELINE_EPOCAS,
        help=f"Epocas de la red base (default: {config.BASELINE_EPOCAS})"
    )
    parser.add_argument(
        "--poblacion", type=int, default=config.GA_TAMANIO_POBLACION,
        help=f"Tamano de la poblacion del GA (default: {config.GA_TAMANIO_POBLACION}, min: 10)"
    )
    parser.add_argument(
        "--generaciones", type=int, default=config.GA_NUM_GENERACIONES,
        help=f"Generaciones del GA (default: {config.GA_NUM_GENERACIONES}, min: 10)"
    )
    parser.add_argument(
        "--epocas-individuo", type=int, default=config.GA_EPOCAS_INDIVIDUO,
        help=f"Epocas por individuo en el GA (default: {config.GA_EPOCAS_INDIVIDUO})"
    )
    parser.add_argument(
        "--epocas-final", type=int, default=config.EPOCAS_REENTRENAMIENTO_FINAL,
        help=f"Epocas del reentrenamiento final (default: {config.EPOCAS_REENTRENAMIENTO_FINAL})"
    )
    args = parser.parse_args()

    _encabezado("Evaluacion de la Red Base (Baseline)", fase_num=1)
    if args.force or not os.path.exists(CENTINELA_BASELINE):
        entrenar_y_evaluar_base(epocas=args.epocas_base)
    else:
        print("Resultados del baseline ya existen. Omitiendo.")

    _encabezado("Optimizacion con AG + Seleccion del Mejor Individuo", fase_num=2)
    if args.force or not os.path.exists(CENTINELA_GA):
        ejecutar_ga(
            tamanio_poblacion=args.poblacion,
            num_generaciones=args.generaciones,
            epocas_individuo=args.epocas_individuo,
        )
    else:
        print("Resultados del GA ya existen. Omitiendo.")

    _encabezado("Reentrenamiento del Mejor Individuo + Evaluacion en Test", fase_num=4)
    if args.force or not os.path.exists(CENTINELA_FINAL):
        reentrenar_y_evaluar_mejor(epocas=args.epocas_final)
    else:
        print("Resultados del reentrenamiento ya existen. Omitiendo.")

    _encabezado("Generacion del Reporte Final", fase_num=5)
    generate_all()

    print("\nPipeline completo finalizado.")
    print(f"Resultados numericos: {config.DIRECTORIO_RESULTS}")
    print(f"Graficas y reporte: {config.DIRECTORIO_REPORTE}")


if __name__ == "__main__":
    main()
