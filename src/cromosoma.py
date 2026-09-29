"""
cromosoma.py - Codificacion Genetica del Espacio de Busqueda

Proposito:
Definir como se representa una arquitectura de CNN como un vector numerico
(cromosoma), y como se construye la red Keras a partir de ese vector.

Glosario:
- Cromosoma: En Algoritmos Geneticos es un vector numerico que codifica
  una solucion candidata. Cada numero del vector es un "gen".
- Gen: Una posicion en el cromosoma que controla un hiperparametro especifico.
- Mecanismo de enmascaramiento: Como pyGAD requiere cromosomas de longitud
  fija, siempre generamos genes para el maximo numero de capas permitidas (3).
  Sin embargo, si el gen "num_capas_conv" indica un valor menor (ej. 2),
  el constructor ignora los genes de la capa sobrante.
- Padding 'valid': En Keras, reduce las dimensiones espaciales tras cada
  convolucion. La funcion _calcular_profundidad_valida previene que 
  apilar muchas capas colapse la imagen a tamano cero.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from . import config

# Espacio de busqueda: Valores discretos permitidos para cada gen
ESPACIO_LR          = [0.01, 0.005, 0.001, 0.0005, 0.0001]
ESPACIO_BATCH        = [32, 64, 128]
ESPACIO_OPTIMIZADOR  = [0, 1, 2]
ESPACIO_NUM_CONV     = list(range(config.MIN_CAPAS_CONV, config.MAX_CAPAS_CONV + 1))
ESPACIO_FILTROS      = [16, 32, 64, 128, 256]
ESPACIO_KERNEL       = [3, 5]
ESPACIO_NEURONAS     = [64, 128, 256, 512]
ESPACIO_DROPOUT      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

NOMBRES_GENES = (
    ["learning_rate", "batch_size", "optimizador", "num_capas_conv"]
    + [f"filtros_capa_{i+1}" for i in range(config.MAX_CAPAS_CONV)]
    + ["kernel_size", "neuronas_densa", "dropout"]
)

# pyGAD usa esto para saber que valores son validos para cada gen
ESPACIO_GENES = (
    [ESPACIO_LR, ESPACIO_BATCH, ESPACIO_OPTIMIZADOR, ESPACIO_NUM_CONV]
    + [ESPACIO_FILTROS] * config.MAX_CAPAS_CONV
    + [ESPACIO_KERNEL, ESPACIO_NEURONAS, ESPACIO_DROPOUT]
)

TIPO_GENES = (
    [float, int, int, int]
    + [int] * config.MAX_CAPAS_CONV
    + [int, int, float]
)

NUM_GENES = len(ESPACIO_GENES)

_IDX_FILTROS_INICIO = 4
_IDX_KERNEL         = _IDX_FILTROS_INICIO + config.MAX_CAPAS_CONV
_IDX_NEURONAS       = _IDX_KERNEL + 1
_IDX_DROPOUT        = _IDX_NEURONAS + 1


def _calcular_profundidad_valida(num_capas_solicitadas, kernel_size,
                                  tamanio_entrada=config.FORMA_IMAGEN[0]):
    """
    Simula la reduccion de dimensiones tras convoluciones con padding 'valid'.
    Retorna el maximo numero de capas posibles sin que el tensor llegue a tamano 0.
    """
    tamanio_actual = tamanio_entrada
    capas_validas = 0

    for _ in range(num_capas_solicitadas):
        tamanio_tras_conv = tamanio_actual - kernel_size + 1
        # Se requiere un tamano minimo de 2 para el MaxPooling2D de (2,2)
        if tamanio_tras_conv < 2:
            break
        tamanio_actual = tamanio_tras_conv // 2
        capas_validas += 1

    return max(capas_validas, 1)


def decodificar_cromosoma(vector):
    """
    Traduce un vector numerico en un diccionario de hiperparametros legibles.
    Aplica la validacion dimensional para garantizar que el modelo sea valido.
    """
    vector = list(vector)
    kernel_size = int(round(vector[_IDX_KERNEL]))

    capas_solicitadas = int(round(vector[3]))
    capas_solicitadas = max(config.MIN_CAPAS_CONV,
                            min(config.MAX_CAPAS_CONV, capas_solicitadas))

    num_capas_reales = min(
        capas_solicitadas,
        _calcular_profundidad_valida(capas_solicitadas, kernel_size)
    )

    # Extrae solo los genes de filtros para las capas que realmente existiran
    filtros = [
        int(round(vector[_IDX_FILTROS_INICIO + i]))
        for i in range(num_capas_reales)
    ]

    return {
        "learning_rate":       float(vector[0]),
        "batch_size":          int(round(vector[1])),
        "optimizador":         int(round(vector[2])),
        "num_capas_conv":      num_capas_reales,
        "num_capas_solicitadas": capas_solicitadas,
        "filtros":             filtros,
        "kernel_size":         kernel_size,
        "neuronas_densa":      int(round(vector[_IDX_NEURONAS])),
        "dropout":             float(vector[_IDX_DROPOUT]),
    }


def cromosoma_a_texto(vector):
    """Convierte un cromosoma en una cadena para impresion en logs."""
    p = decodificar_cromosoma(vector)
    nombre_opt = config.MAPA_OPTIMIZADOR[p["optimizador"]]
    return (
        f"lr={p['learning_rate']}, batch={p['batch_size']}, "
        f"opt={nombre_opt}, capas={p['num_capas_conv']}, "
        f"filtros={p['filtros']}, kernel={p['kernel_size']}, "
        f"neuronas={p['neuronas_densa']}, dropout={p['dropout']}"
    )


def construir_cnn_desde_cromosoma(vector):
    """
    Construye y compila una CNN a partir del cromosoma decodificado.
    
    Estructura generada:
    Entrada -> N bloques (Conv2D + MaxPooling2D) -> Flatten -> Dropout -> Dense -> Salida
    """
    params = decodificar_cromosoma(vector)
    k = params["kernel_size"]

    modelo = Sequential()

    for indice_capa, num_filtros in enumerate(params["filtros"]):
        if indice_capa == 0:
            modelo.add(Conv2D(
                filters=num_filtros,
                kernel_size=(k, k),
                activation="relu",
                input_shape=config.FORMA_IMAGEN,
            ))
        else:
            modelo.add(Conv2D(
                filters=num_filtros,
                kernel_size=(k, k),
                activation="relu",
            ))

        modelo.add(MaxPooling2D(pool_size=(2, 2)))

    modelo.add(Flatten())

    if params["dropout"] > 0:
        modelo.add(Dropout(rate=params["dropout"]))

    modelo.add(Dense(units=params["neuronas_densa"], activation="relu"))
    modelo.add(Dense(units=config.NUM_CLASES, activation="softmax"))

    nombre_opt = config.MAPA_OPTIMIZADOR[params["optimizador"]]
    lr = params["learning_rate"]

    if nombre_opt == "adam":
        optimizador = tf.keras.optimizers.Adam(learning_rate=lr)
    elif nombre_opt == "rmsprop":
        optimizador = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizador = tf.keras.optimizers.SGD(learning_rate=lr)

    modelo.compile(
        optimizer=optimizador,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return modelo, params


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    cromosoma_prueba = [rng.choice(espacio) for espacio in ESPACIO_GENES]

    print("Verificacion del Cromosoma")
    print(f"Genes en crudo: {cromosoma_prueba}")
    print(f"Decodificado: {cromosoma_a_texto(cromosoma_prueba)}")

    modelo, parametros = construir_cnn_desde_cromosoma(cromosoma_prueba)
    modelo.summary()
