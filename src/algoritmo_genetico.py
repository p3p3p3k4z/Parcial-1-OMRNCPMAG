"""
algoritmo_genetico.py - Optimizacion Evolutiva con pyGAD

Proposito:
Ejecutar el Algoritmo Genetico para encontrar la mejor arquitectura CNN
del espacio de busqueda definido.

Glosario:
- Poblacion: Conjunto de cromosomas (arquitecturas) evaluados en cada generacion.
- Generacion: Un ciclo completo que incluye evaluar fitness, seleccionar padres,
  cruzar y mutar.
- Seleccion de Padres (SSS - Steady-State Selection): Selecciona a los mejores 
  individuos para reproducirse. Los mejores sobreviven y los peores son 
  reemplazados gradualmente.
- Cruce (Crossover) de un punto: Combina dos padres cortandolos en un punto 
  aleatorio y uniendo la primera mitad de uno con la segunda mitad del otro 
  para crear un hijo con caracteristicas de ambos.
- Mutacion aleatoria: Cambia el valor de algunos genes al azar para mantener 
  la diversidad y explorar nuevas soluciones.
- Elitismo: Los mejores individuos de cada generacion pasan directamente 
  a la siguiente sin modificaciones, garantizando que no se pierda la mejor solucion.
- Inversion de Dependencias (SOLID): La funcion de fitness se inyecta como 
  parametro en lugar de crearse internamente, permitiendo probar el algoritmo 
  con distintas formas de evaluacion.
"""
import os
import json
import time

import numpy as np
import pygad

from . import config
from .datos import cargar_cifar10
from .cromosoma import ESPACIO_GENES, TIPO_GENES, NUM_GENES, cromosoma_a_texto, decodificar_cromosoma
from .aptitud import crear_funcion_fitness, _cache_fitness

RUTA_LOG_GENERACIONES = os.path.join(config.DIRECTORIO_GA, "historial_generaciones.json")
RUTA_LOG_INDIVIDUOS   = os.path.join(config.DIRECTORIO_GA, "historial_individuos.json")
RUTA_RESUMEN_GA       = os.path.join(config.DIRECTORIO_GA, "resumen_ga.json")


def _crear_callback_generacion(tiempo_inicio_ga):
    """
    Crea un callback que pyGAD llama automaticamente al terminar cada generacion
    para guardar el progreso en disco e imprimir el estatus en consola.
    """
    historial_generaciones = []

    def al_terminar_generacion(ga_instancia):
        mejor_cromosoma, mejor_fitness, _ = ga_instancia.best_solution(
            pop_fitness=ga_instancia.last_generation_fitness
        )
        generacion = int(ga_instancia.generations_completed)
        tiempo_transcurrido = time.time() - tiempo_inicio_ga

        registro = {
            "generacion":         generacion,
            "mejor_fitness":      float(mejor_fitness),
            "mejor_cromosoma":    list(map(float, mejor_cromosoma)),
            "descripcion":        cromosoma_a_texto(mejor_cromosoma),
            "tiempo_total_min":   tiempo_transcurrido / 60,
        }
        historial_generaciones.append(registro)

        with open(RUTA_LOG_GENERACIONES, "w", encoding="utf-8") as archivo:
            json.dump(historial_generaciones, archivo, indent=2)

        print(
            f"\nGeneracion {generacion:02d} completada\n"
            f"Mejor fitness: {mejor_fitness:.4f}\n"
            f"Mejor arquitectura: {cromosoma_a_texto(mejor_cromosoma)}\n"
            f"Tiempo total: {tiempo_transcurrido/60:.1f} min\n"
        )

    return al_terminar_generacion


def ejecutar_ga(tamanio_poblacion=None, num_generaciones=None, epocas_individuo=None):
    """
    Orquesta y ejecuta el Algoritmo Genetico completo.

    Retorna el objeto GA con el estado final y un resumen de las metricas.
    """
    tamanio_poblacion = tamanio_poblacion or config.GA_TAMANIO_POBLACION
    num_generaciones  = num_generaciones  or config.GA_NUM_GENERACIONES
    epocas_individuo  = epocas_individuo  or config.GA_EPOCAS_INDIVIDUO

    assert tamanio_poblacion >= 10, (
        f"El enunciado exige poblacion >= 10. Se proporciono: {tamanio_poblacion}"
    )
    assert num_generaciones >= 10, (
        f"El enunciado exige generaciones >= 10. Se proporciono: {num_generaciones}"
    )

    (x_train, y_train), (x_val, y_val), _ = cargar_cifar10()

    funcion_fitness, _ = crear_funcion_fitness(
        x_train, y_train, x_val, y_val,
        epocas=epocas_individuo,
        ruta_log=RUTA_LOG_INDIVIDUOS,
    )

    np.random.seed(config.SEED)
    tiempo_inicio = time.time()

    print("Inicio del Algoritmo Genetico con pyGAD")
    print(f"Poblacion: {tamanio_poblacion} | Generaciones: {num_generaciones}")
    print(f"Epocas por individuo: {epocas_individuo} | Genes: {NUM_GENES}")

    instancia_ga = pygad.GA(
        num_generations=num_generaciones,
        sol_per_pop=tamanio_poblacion,
        num_parents_mating=config.GA_NUM_PADRES,
        fitness_func=funcion_fitness,
        num_genes=NUM_GENES,
        gene_space=ESPACIO_GENES,
        gene_type=TIPO_GENES,
        parent_selection_type=config.GA_SELECCION_PADRES,
        keep_elitism=config.GA_ELITISMO,
        crossover_type=config.GA_TIPO_CRUCE,
        mutation_type=config.GA_TIPO_MUTACION,
        mutation_percent_genes=config.GA_PORCENTAJE_MUTACION,
        random_seed=config.SEED,
        on_generation=_crear_callback_generacion(tiempo_inicio),
        save_best_solutions=True,
        suppress_warnings=True,
    )

    instancia_ga.run()

    tiempo_total = time.time() - tiempo_inicio
    mejor_cromosoma, mejor_fitness, _ = instancia_ga.best_solution()
    mejores_params = decodificar_cromosoma(mejor_cromosoma)

    clave_mejor = tuple(round(float(g), 6) for g in mejor_cromosoma)
    info_mejor = _cache_fitness.get(clave_mejor, {})

    resumen = {
        "tamanio_poblacion":  tamanio_poblacion,
        "num_generaciones":   num_generaciones,
        "epocas_individuo":   epocas_individuo,
        "tiempo_total_min":   tiempo_total / 60,
        "mejor_cromosoma":    list(map(float, mejor_cromosoma)),
        "mejor_descripcion":  cromosoma_a_texto(mejor_cromosoma),
        "mejores_params":     mejores_params,
        "mejor_fitness":      float(mejor_fitness),
        "mejor_val_accuracy": info_mejor.get("val_accuracy"),
        "mejor_num_params":   info_mejor.get("num_parametros"),
    }

    with open(RUTA_RESUMEN_GA, "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2)

    print("\nResultado Final del Algoritmo Genetico")
    print(f"Tiempo total: {tiempo_total/60:.1f} min")
    print(f"Mejor fitness: {mejor_fitness:.4f}")
    print(f"Mejor arquitectura: {cromosoma_a_texto(mejor_cromosoma)}")
    print(f"Val Accuracy: {info_mejor.get('val_accuracy', 'N/A')}")
    print(f"Parametros: {info_mejor.get('num_parametros', 'N/A'):,}")

    return instancia_ga, resumen


if __name__ == "__main__":
    import sys
    pop  = int(sys.argv[1]) if len(sys.argv) > 1 else None
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else None
    epo  = int(sys.argv[3]) if len(sys.argv) > 3 else None
    ejecutar_ga(tamanio_poblacion=pop, num_generaciones=gens, epocas_individuo=epo)
