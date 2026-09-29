"""
Fase 4 y 5: Reentrenamiento del modelo optimizado (30 épocas) y evaluación
final sobre el conjunto Test, comparándolo contra el baseline.
"""
import os
import json
import time
import numpy as np
import tensorflow as tf

from . import config
from .data_utils import load_cifar10_splits
from .chromosome import build_model_from_chromosome, chromosome_to_str

GA_SUMMARY_PATH = os.path.join(config.GA_DIR, "ga_summary.json")
FINAL_RESULTS_PATH = os.path.join(config.FINAL_DIR, "final_results.json")


def retrain_and_evaluate_best(epochs=None):
    epochs = epochs or config.FINAL_RETRAIN_EPOCHS

    with open(GA_SUMMARY_PATH, "r", encoding="utf-8") as f:
        ga_summary = json.load(f)
    best_chromosome = ga_summary["best_chromosome"]

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_splits()

    model, params = build_model_from_chromosome(best_chromosome)
    trainable_params = int(
        np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    )

    print("=== Reentrenando el mejor individuo (Fase 4) ===")
    print(chromosome_to_str(best_chromosome))
    print(f"Trainable params: {trainable_params}")

    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=params["batch_size"],
        verbose=1,
    )
    train_time = time.time() - start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    results = {
        "chromosome": best_chromosome,
        "chromosome_str": chromosome_to_str(best_chromosome),
        "params": params,
        "trainable_params": trainable_params,
        "epochs": epochs,
        "batch_size": params["batch_size"],
        "train_time_seconds": train_time,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "ga_selection_val_accuracy": ga_summary.get("best_val_accuracy"),
        "ga_selection_fitness": ga_summary.get("best_fitness"),
    }

    with open(FINAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    model.save(os.path.join(config.FINAL_DIR, "optimized_model.keras"))

    print("=== Modelo Optimizado (Fase 5 - evaluación Test) ===")
    print(f"Trainable params: {trainable_params}")
    print(f"Train time: {train_time:.1f}s")
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")
    return model, results


if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    retrain_and_evaluate_best(epochs=epochs)
