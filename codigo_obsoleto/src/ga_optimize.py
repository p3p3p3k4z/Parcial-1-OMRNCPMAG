"""
Fase 2 y 3: Optimización evolutiva de la CNN mediante pyGAD y selección del
mejor individuo.
"""
import os
import json
import time
import numpy as np
import pygad

from . import config
from .data_utils import load_cifar10_splits
from .chromosome import GENE_SPACE, GENE_TYPE, NUM_GENES, chromosome_to_str, decode_chromosome
from .fitness import make_fitness_func

GENERATION_LOG_PATH = os.path.join(config.GA_DIR, "generation_best_fitness.json")
ALL_EVALS_PATH = os.path.join(config.GA_DIR, "all_individuals_log.json")
GA_SUMMARY_PATH = os.path.join(config.GA_DIR, "ga_summary.json")

_generation_history = []


def _make_on_generation_callback(ga_start_time):
    def on_generation(ga_instance):
        best_solution, best_fitness, _ = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )
        gen = ga_instance.generations_completed
        elapsed = time.time() - ga_start_time
        record = {
            "generation": int(gen),
            "best_fitness": float(best_fitness),
            "best_chromosome": list(map(float, best_solution)),
            "best_chromosome_str": chromosome_to_str(best_solution),
            "elapsed_seconds": elapsed,
        }
        _generation_history.append(record)
        with open(GENERATION_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(_generation_history, f, indent=2)
        print(
            f"\n>>> Generación {gen} completada | mejor fitness={best_fitness:.4f} "
            f"| tiempo total={elapsed/60:.1f} min\n"
        )
    return on_generation


def run_ga(population_size=None, num_generations=None, individual_epochs=None):
    population_size = population_size or config.GA_POPULATION_SIZE
    num_generations = num_generations or config.GA_NUM_GENERATIONS
    individual_epochs = individual_epochs or config.GA_INDIVIDUAL_EPOCHS

    assert population_size >= 10, "Población mínima exigida por el enunciado: 10"
    assert num_generations >= 10, "Generaciones mínimas exigidas por el enunciado: 10"

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_splits()

    fitness_func, eval_log = make_fitness_func(
        x_train, y_train, x_val, y_val,
        epochs=individual_epochs,
        log_path=ALL_EVALS_PATH,
    )

    np.random.seed(config.SEED)
    start_time = time.time()

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=config.GA_NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=population_size,
        num_genes=NUM_GENES,
        gene_space=GENE_SPACE,
        gene_type=GENE_TYPE,
        parent_selection_type=config.GA_PARENT_SELECTION_TYPE,
        keep_elitism=config.GA_KEEP_ELITISM,
        crossover_type=config.GA_CROSSOVER_TYPE,
        mutation_type=config.GA_MUTATION_TYPE,
        mutation_percent_genes=config.GA_MUTATION_PERCENT_GENES,
        random_seed=config.SEED,
        on_generation=_make_on_generation_callback(start_time),
        save_best_solutions=True,
        suppress_warnings=True,
    )

    print(
        f"Iniciando GA: población={population_size}, generaciones={num_generations}, "
        f"épocas/individuo={individual_epochs}, genes={NUM_GENES}"
    )
    ga_instance.run()

    total_time = time.time() - start_time
    best_solution, best_fitness, best_idx = ga_instance.best_solution()
    best_params = decode_chromosome(best_solution)

    summary = {
        "population_size": population_size,
        "num_generations": num_generations,
        "individual_epochs": individual_epochs,
        "total_time_seconds": total_time,
        "best_chromosome": list(map(float, best_solution)),
        "best_chromosome_str": chromosome_to_str(best_solution),
        "best_params": best_params,
        "best_fitness": float(best_fitness),
        "best_val_accuracy": None,
        "best_trainable_params": None,
    }
    # recuperar val_accuracy / TP del cromosoma ganador desde el cache de fitness
    key = tuple(round(float(v), 6) for v in best_solution)
    from .fitness import _CACHE
    if key in _CACHE:
        summary["best_val_accuracy"] = _CACHE[key]["val_accuracy"]
        summary["best_trainable_params"] = _CACHE[key]["trainable_params"]

    with open(GA_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Resultado del Algoritmo Genético ===")
    print(f"Tiempo total: {total_time/60:.1f} min")
    print(f"Mejor fitness: {best_fitness:.4f}")
    print(f"Mejor cromosoma: {chromosome_to_str(best_solution)}")
    print(f"val_accuracy={summary['best_val_accuracy']}  TP={summary['best_trainable_params']}")

    return ga_instance, summary


if __name__ == "__main__":
    import sys
    pop = int(sys.argv[1]) if len(sys.argv) > 1 else None
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else None
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_ga(population_size=pop, num_generations=gens, individual_epochs=epochs)
