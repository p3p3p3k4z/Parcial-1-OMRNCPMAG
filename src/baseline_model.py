"""
Fase 1: Arquitectura Base (Baseline).

Reproduce EXACTAMENTE la arquitectura presentada en clase (ver enunciado,
Parcial 1) para servir como punto de comparación contra la red optimizada
por el Algoritmo Genético.
"""
import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from . import config
from .data_utils import load_cifar10_splits


def build_baseline_model():
    model = Sequential()
    model.add(
        Conv2D(
            config.BASELINE_FILTER1_SIZE,
            config.BASELINE_FILTER_SHAPE,
            activation="relu",
            input_shape=config.INPUT_SHAPE,
        )
    )
    model.add(MaxPooling2D(config.BASELINE_POOL_SHAPE))
    model.add(
        Conv2D(
            config.BASELINE_FILTER2_SIZE,
            config.BASELINE_FILTER_SHAPE,
            activation="relu",
        )
    )
    model.add(MaxPooling2D(config.BASELINE_POOL_SHAPE))
    model.add(Flatten())
    model.add(Dense(config.BASELINE_FULLY_CONNECT_NUM, activation="relu"))
    model.add(Dense(config.NUM_CLASSES, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.BASELINE_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate_baseline(epochs=None, verbose=1):
    epochs = epochs or config.BASELINE_EPOCHS
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_splits()

    model = build_baseline_model()
    trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))

    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=config.BASELINE_BATCH_SIZE,
        verbose=verbose,
    )
    train_time = time.time() - start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    results = {
        "trainable_params": trainable_params,
        "train_time_seconds": train_time,
        "epochs": epochs,
        "batch_size": config.BASELINE_BATCH_SIZE,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }

    with open(os.path.join(config.BASELINE_DIR, "baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    model.save(os.path.join(config.BASELINE_DIR, "baseline_model.keras"))

    print("=== Baseline ===")
    print(f"Trainable params: {trainable_params}")
    print(f"Train time: {train_time:.1f}s")
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")
    return model, results


if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train_and_evaluate_baseline(epochs=epochs)
