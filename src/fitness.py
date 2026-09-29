"""
Función de fitness multiobjetivo utilizada por pyGAD (Fase 2).

fitness = w1 * val_accuracy + w2 * (1 - clip((TP - TP_min)/(TP_max - TP_min), 0, 1))

- w1 = 0.7 (peso de exactitud) y w2 = 0.3 (peso de eficiencia/parámetros),
  valores sugeridos por el enunciado. Se prioriza más la exactitud porque en
  clasificación de imágenes una caída fuerte de accuracy es más costosa que
  un modelo moderadamente más grande; sin embargo, w2=0.3 sigue siendo lo
  bastante alto para empujar al GA hacia arquitecturas compactas cuando el
  costo en accuracy es marginal.
- TP_min=10,000 / TP_max=1,000,000 se recalibraron empíricamente a partir del
  espacio de búsqueda real (1-3 capas conv, 16-256 filtros, densa 64-512
  unidades), cuyo rango de parámetros entrenables va de ~9,900 a ~29.5
  millones (baseline=315,722). Ver la justificación completa en
  `config.py`.
- Cada individuo se entrena únicamente con (Train) y se evalúa con
  (Validation); el conjunto Test NUNCA se usa aquí (Fase 2 del protocolo).
- Se cachea el fitness por cromosoma (hash exacto de sus genes) para evitar
  reentrenar individuos idénticos que sobreviven por elitismo entre
  generaciones, lo cual reduce significativamente el tiempo total de cómputo
  sin alterar el comportamiento del GA.
"""
import os
import json
import time
import numpy as np
import tensorflow as tf

from . import config
from .chromosome import build_model_from_chromosome, decode_chromosome, chromosome_to_str

_CACHE = {}
_EVAL_LOG = []


def _chrom_key(solution):
    return tuple(round(float(v), 6) for v in solution)


def make_fitness_func(x_train, y_train, x_val, y_val, epochs=None, log_path=None):
    epochs = epochs or config.GA_INDIVIDUAL_EPOCHS

    def fitness_func(ga_instance, solution, solution_idx):
        key = _chrom_key(solution)
        if key in _CACHE:
            cached = _CACHE[key]
            return cached["fitness"]

        t0 = time.time()
        try:
            model, params = build_model_from_chromosome(solution)
            trainable_params = int(
                np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            )
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=params["batch_size"],
                verbose=0,
            )
            val_accuracy = float(np.max(history.history["val_accuracy"]))

            normalized_tp = (trainable_params - config.TP_MIN) / (config.TP_MAX - config.TP_MIN)
            normalized_tp = float(np.clip(normalized_tp, 0.0, 1.0))
            fitness = config.FITNESS_W1 * val_accuracy + config.FITNESS_W2 * (1.0 - normalized_tp)

            tf.keras.backend.clear_session()
        except Exception as exc:  # arquitectura inválida o error de entrenamiento
            print(f"[WARN] Individuo inválido ({exc}); fitness=0")
            val_accuracy = 0.0
            trainable_params = 0
            fitness = 0.0

        elapsed = time.time() - t0
        generation = ga_instance.generations_completed
        record = {
            "generation": int(generation),
            "solution_idx": int(solution_idx),
            "chromosome": list(map(float, solution)),
            "chromosome_str": chromosome_to_str(solution),
            "val_accuracy": val_accuracy,
            "trainable_params": trainable_params,
            "fitness": float(fitness),
            "time_seconds": elapsed,
        }
        _CACHE[key] = record
        _EVAL_LOG.append(record)

        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(_EVAL_LOG, f, indent=2)

        print(
            f"[gen {generation:02d} ind {solution_idx:02d}] "
            f"acc={val_accuracy:.4f} TP={trainable_params:6d} "
            f"fitness={fitness:.4f} ({elapsed:.1f}s)"
        )
        return fitness

    return fitness_func, _EVAL_LOG
