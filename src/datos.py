"""
datos.py - Carga y Particion del Dataset CIFAR-10

Proposito:
Cargar las imagenes y dividirlas en tres subconjuntos fijos que se usaran 
durante todo el experimento.

Glosario:
- Normalizacion (/ 255.0): Los pixeles originales van de 0 a 255. 
  Dividir por 255 los lleva al rango de 0 a 1. Las redes neuronales aprenden 
  mas rapido y de forma mas estable con entradas pequenas.
- Particion estratificada (stratify): Garantiza que cada subconjunto tenga 
  la misma proporcion de clases que el dataset original, evitando sesgos.
- random_state: Hace que la division sea identica en cada ejecucion.
- .ravel(): Convierte un arreglo de multiples dimensiones en un vector plano.
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from . import config


def cargar_cifar10():
    """
    Descarga, normaliza y divide el dataset en Train / Validacion / Test.

    Proporciones:
    - Train: 60%
    - Validacion: 20%
    - Test: 20%

    Retorna tres pares de tuplas con las imagenes y sus respectivas etiquetas.
    """
    (x_keras_train, y_keras_train), (x_keras_test, y_keras_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    x_todo = np.concatenate([x_keras_train, x_keras_test], axis=0)
    y_todo = np.concatenate([y_keras_train, y_keras_test], axis=0)

    x_todo = x_todo.astype("float32") / 255.0

    y_todo = y_todo.ravel()

    # Primera division: Separar Test (20%)
    # El conjunto Test queda "congelado" y solo se usa al final para la 
    # comparacion definitiva de los modelos entrenados.
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_todo, y_todo,
        test_size=config.FRACCION_TEST,
        random_state=config.SEED,
        stratify=y_todo,
    )

    # Segunda division: Del 80% restante, separar Train (75%) y Validacion (25%)
    # Esto resulta matematicamente en 60% Train y 20% Validacion del total.
    fraccion_val_del_temp = config.FRACCION_VAL / (1.0 - config.FRACCION_TEST)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        test_size=fraccion_val_del_temp,
        random_state=config.SEED,
        stratify=y_temp,
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    (xtr, ytr), (xva, yva), (xte, yte) = cargar_cifar10()
    total = xtr.shape[0] + xva.shape[0] + xte.shape[0]

    print("Verificacion de Particion CIFAR-10")
    print(f"Train: {xtr.shape} etiquetas: {ytr.shape} ({xtr.shape[0]/total:.0%})")
    print(f"Validacion: {xva.shape} etiquetas: {yva.shape} ({xva.shape[0]/total:.0%})")
    print(f"Test: {xte.shape} etiquetas: {yte.shape} ({xte.shape[0]/total:.0%})")
    print(f"Total: {total} imagenes")
    print(f"Rango de valores: [{xtr.min():.1f}, {xtr.max():.1f}]")
