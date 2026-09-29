"""
Carga de CIFAR-10 y partición fija 60% Train / 20% Validation / 20% Test.

El enunciado exige que las particiones se mantengan fijas durante TODO el
experimento (Fase 1 baseline, Fase 2 optimización GA y Fase 4 reentrenamiento
final). Para garantizarlo:
  1. Se combinan las 60,000 imágenes originales de CIFAR-10 (50k train + 10k
     test de Keras) en un único pool.
  2. Se mezclan con una semilla fija (config.SEED).
  3. Se particionan 60/20/20 y los índices se guardan en disco para que
     cualquier fase del proyecto (ejecutada en cualquier momento/proceso)
     reconstruya EXACTAMENTE la misma partición.
"""
import os
import json
import numpy as np
import tensorflow as tf

from . import config


def _split_indices_path():
    return os.path.join(config.RESULTS_DIR, "split_indices.json")


def _compute_and_save_split(n_total: int):
    rng = np.random.default_rng(config.SEED)
    indices = rng.permutation(n_total)

    n_train = int(round(n_total * config.TRAIN_FRAC))
    n_val = int(round(n_total * config.VAL_FRAC))
    # el resto (aprox. 20%) va a test, evitando perder/duplicar muestras por redondeo
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    payload = {
        "seed": config.SEED,
        "n_total": int(n_total),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    with open(_split_indices_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return train_idx, val_idx, test_idx


def _load_split(n_total: int):
    path = _split_indices_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("n_total") == n_total and payload.get("seed") == config.SEED:
            return (
                np.array(payload["train_idx"]),
                np.array(payload["val_idx"]),
                np.array(payload["test_idx"]),
            )
    return _compute_and_save_split(n_total)


def load_cifar10_splits():
    """Devuelve (x_train, y_train), (x_val, y_val), (x_test, y_test).

    Las imágenes se normalizan a [0, 1] (float32) y las etiquetas quedan como
    vectores enteros (n, 1) tal como los entrega keras.datasets.cifar10.
    """
    (x1, y1), (x2, y2) = tf.keras.datasets.cifar10.load_data()
    x_all = np.concatenate([x1, x2], axis=0).astype("float32") / 255.0
    y_all = np.concatenate([y1, y2], axis=0)

    n_total = x_all.shape[0]
    train_idx, val_idx, test_idx = _load_split(n_total)

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    (xtr, ytr), (xva, yva), (xte, yte) = load_cifar10_splits()
    print("Train:", xtr.shape, ytr.shape)
    print("Val:  ", xva.shape, yva.shape)
    print("Test: ", xte.shape, yte.shape)
    total = xtr.shape[0] + xva.shape[0] + xte.shape[0]
    print("Total:", total)
    print(f"% train={xtr.shape[0]/total:.3f} val={xva.shape[0]/total:.3f} test={xte.shape[0]/total:.3f}")
