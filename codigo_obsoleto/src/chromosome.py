"""
Codificación genética del espacio de búsqueda arquitectónico e hiperparamétrico.

Cada individuo (cromosoma) es un vector de longitud fija:

  [ lr, batch_size, optimizer, n_conv_layers,
    filters_1, filters_2, filters_3, filters_4,   <- MAX_CONV_LAYERS genes
    kernel_size, dense_units, dropout ]

Solo se usan los primeros `n_conv_layers` genes de `filters_*`; los restantes
se ignoran al construir el modelo (permite representar profundidad variable
con un vector de longitud fija, requisito de pyGAD).

Restricción arquitectónica (obligatoria en el enunciado):
  Entrada + (Conv2D+MaxPooling2D) x n_conv_layers + Flatten + Dense + Salida
Cada Conv2D genera siempre su MaxPooling2D asociado; nunca se omite.

Se usa padding='valid' en las capas Conv2D generadas por el GA, exactamente
igual que en el baseline (sin argumento `padding`, cuyo valor por defecto en
Keras es 'valid'), para que la comparación de parámetros/arquitectura contra
el baseline sea justa (misma convención de reducción espacial).

Como con padding 'valid' no toda combinación de (kernel_size, n_conv_layers)
es geométricamente válida sobre una entrada de 32x32 (el mapa de
características podría intentar reducirse a un tamaño <= 0), se recorta
automáticamente la profundidad solicitada por el cromosoma al máximo número
de pares Conv2D+MaxPooling2D que sí caben (ver `_max_valid_depth`). Así se
respeta siempre la restricción arquitectónica y nunca se genera un modelo
inválido, sin necesidad de descartar al individuo.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from . import config

# ---------------------------------------------------------------------------
# Espacio de búsqueda (valores discretos permitidos por gen)
# ---------------------------------------------------------------------------
LR_SPACE = [0.01, 0.005, 0.001, 0.0005, 0.0001]
BATCH_SPACE = [32, 64, 128]
OPTIMIZER_SPACE = [0, 1, 2]
NCONV_SPACE = list(range(config.MIN_CONV_LAYERS, config.MAX_CONV_LAYERS + 1))
FILTERS_SPACE = [16, 32, 64, 128, 256]
KERNEL_SPACE = [3, 5]
DENSE_SPACE = [64, 128, 256, 512]
DROPOUT_SPACE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

GENE_NAMES = (
    ["learning_rate", "batch_size", "optimizer", "n_conv_layers"]
    + [f"filters_layer{i+1}" for i in range(config.MAX_CONV_LAYERS)]
    + ["kernel_size", "dense_units", "dropout"]
)

GENE_SPACE = (
    [LR_SPACE, BATCH_SPACE, OPTIMIZER_SPACE, NCONV_SPACE]
    + [FILTERS_SPACE] * config.MAX_CONV_LAYERS
    + [KERNEL_SPACE, DENSE_SPACE, DROPOUT_SPACE]
)

NUM_GENES = len(GENE_SPACE)

# Tipo de cada gen para pyGAD: los hiperparámetros numéricos continuos
# (learning_rate, dropout) son float; el resto son enteros (categorías o
# conteos discretos).
GENE_TYPE = (
    [float, int, int, int]
    + [int] * config.MAX_CONV_LAYERS
    + [int, int, float]
)

_FILTERS_START = 4
_KERNEL_IDX = _FILTERS_START + config.MAX_CONV_LAYERS
_DENSE_IDX = _KERNEL_IDX + 1
_DROPOUT_IDX = _DENSE_IDX + 1


def _max_valid_depth(n_conv_layers, kernel_size, input_size=config.INPUT_SHAPE[0]):
    """Máximo número de pares Conv2D(valid)+MaxPooling2D(2,2) que caben sin
    que el mapa de características colapse a un tamaño <= 0."""
    size = input_size
    usable = 0
    for _ in range(n_conv_layers):
        conv_size = size - kernel_size + 1
        if conv_size < 2:  # no alcanza ni para un MaxPooling2D(2,2) válido
            break
        size = conv_size // 2
        usable += 1
    return max(usable, 1)


def decode_chromosome(vector):
    """Convierte un vector de genes en un diccionario de hiperparámetros."""
    vector = list(vector)
    kernel_size = int(round(vector[_KERNEL_IDX]))
    requested_n_conv_layers = int(round(vector[3]))
    requested_n_conv_layers = max(
        config.MIN_CONV_LAYERS, min(config.MAX_CONV_LAYERS, requested_n_conv_layers)
    )
    n_conv_layers = min(
        requested_n_conv_layers, _max_valid_depth(requested_n_conv_layers, kernel_size)
    )
    filters = [int(round(vector[_FILTERS_START + i])) for i in range(n_conv_layers)]

    return {
        "learning_rate": float(vector[0]),
        "batch_size": int(round(vector[1])),
        "optimizer": int(round(vector[2])),
        "n_conv_layers": n_conv_layers,
        "requested_n_conv_layers": requested_n_conv_layers,
        "filters": filters,
        "kernel_size": kernel_size,
        "dense_units": int(round(vector[_DENSE_IDX])),
        "dropout": float(vector[_DROPOUT_IDX]),
    }


def chromosome_to_str(vector):
    p = decode_chromosome(vector)
    return (
        f"lr={p['learning_rate']}, batch={p['batch_size']}, "
        f"opt={config.OPTIMIZER_MAP[p['optimizer']]}, "
        f"conv_layers={p['n_conv_layers']}, filters={p['filters']}, "
        f"kernel={p['kernel_size']}, dense={p['dense_units']}, dropout={p['dropout']}"
    )


def build_model_from_chromosome(vector):
    """Construye y compila un modelo Keras a partir de un cromosoma."""
    params = decode_chromosome(vector)
    k = params["kernel_size"]

    model = Sequential()
    for i, f in enumerate(params["filters"]):
        if i == 0:
            model.add(
                Conv2D(
                    f, (k, k), activation="relu",
                    input_shape=config.INPUT_SHAPE,
                )
            )
        else:
            model.add(Conv2D(f, (k, k), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    if params["dropout"] > 0:
        model.add(Dropout(params["dropout"]))
    model.add(Dense(params["dense_units"], activation="relu"))
    model.add(Dense(config.NUM_CLASSES, activation="softmax"))

    opt_name = config.OPTIMIZER_MAP[params["optimizer"]]
    if opt_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    elif opt_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=params["learning_rate"])
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, params


def random_chromosome(rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    return [rng.choice(space) for space in GENE_SPACE]


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    chrom = random_chromosome(rng)
    print("Cromosoma aleatorio:", chrom)
    print(chromosome_to_str(chrom))
    model, params = build_model_from_chromosome(chrom)
    model.summary()
    tp = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    print("Trainable params:", tp)
