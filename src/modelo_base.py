"""
modelo_base.py - Red Neuronal Convolucional de Referencia (Baseline)

Proposito:
Construir, entrenar y evaluar la red neuronal BASE del experimento.
Esta red sirve como punto de comparacion para medir si el Algoritmo
Genetico realmente encontro una arquitectura mejor.

Glosario:
- Baseline (Red de Referencia): Arquitectura estandar. Su rendimiento 
  establece el minimo a superar para el GA.
- Conv2D: Capa convolucional. Aprende a detectar patrones en la imagen 
  aplicando filtros detectores.
- MaxPooling2D: Reduce la imagen a la mitad, haciendo la red mas robusta 
  a pequenas variaciones en la posicion de los patrones.
- Flatten: Convierte el mapa de caracteristicas 3D en un vector 1D, 
  sirviendo de puente hacia las capas densas.
- Dense: Capa totalmente conectada donde cada neurona se conecta con todas 
  las anteriores para aprender relaciones de alto nivel.
- softmax: Funcion que convierte los valores brutos de salida en probabilidades.
"""
import os
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from . import config
from .datos import cargar_cifar10


def construir_red_base():
    """
    Construye y compila la arquitectura CNN base de referencia.
    Solo construye y compila el modelo.

    Retorna el modelo de Keras compilado, listo para entrenar.
    """
    INPUT_SHAPE = config.FORMA_IMAGEN
    FILTER1_SIZE = config.BASELINE_FILTROS_CAPA1
    FILTER2_SIZE = config.BASELINE_FILTROS_CAPA2
    FILTER_SHAPE = config.BASELINE_FORMA_FILTRO
    POOL_SHAPE = config.BASELINE_FORMA_POOLING
    FULLY_CONNECT_NUM = config.BASELINE_NEURONAS_DENSA
    NUM_CLASSES = config.NUM_CLASES

    modelo = Sequential()
    
    modelo.add(
        Conv2D(
            FILTER1_SIZE,
            FILTER_SHAPE,
            activation='relu',
            input_shape=INPUT_SHAPE
        )
    )
    modelo.add(MaxPooling2D(POOL_SHAPE))
    
    modelo.add(
        Conv2D(
            FILTER2_SIZE,
            FILTER_SHAPE,
            activation='relu'
        )
    )
    modelo.add(MaxPooling2D(POOL_SHAPE))
    
    modelo.add(Flatten())
    
    modelo.add(
        Dense(
            FULLY_CONNECT_NUM,
            activation='relu'
        )
    )
    modelo.add(
        Dense(
            NUM_CLASSES,
            activation='softmax'
        )
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.BASELINE_TASA_APRENDIZAJE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return modelo


def entrenar_y_evaluar_base(epocas=None, verbose=1):
    """
    Fase 1: Entrena la red base y la evalua en el conjunto de Test.

    Retorna el modelo entrenado y un diccionario con los resultados.
    """
    epocas = epocas or config.BASELINE_EPOCAS

    print("\n[DEBUG] Cargando y particionando CIFAR-10...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = cargar_cifar10()
    print(f"[DEBUG] Train: {x_train.shape} | Val: {x_val.shape} | Test: {x_test.shape}")
    print(f"[DEBUG] Rango de pixeles: [{x_train.min():.2f}, {x_train.max():.2f}]")

    modelo = construir_red_base()
    num_parametros = modelo.count_params()

    print("\n" + "=" * 55)
    print("FASE 1: Entrenando Red Base (Baseline)")
    print("=" * 55)
    print(f"[DEBUG] Arquitectura: Conv2D({config.BASELINE_FILTROS_CAPA1}) -> Conv2D({config.BASELINE_FILTROS_CAPA2}) -> Dense({config.BASELINE_NEURONAS_DENSA}) -> Softmax(10)")
    print(f"[DEBUG] Parametros entrenables: {num_parametros:,}")
    print(f"[DEBUG] Epocas: {epocas} | Batch size: {config.BASELINE_BATCH_SIZE} | LR: {config.BASELINE_TASA_APRENDIZAJE}")
    print(f"[DEBUG] Optimizador: Adam")
    print("-" * 55)

    tiempo_inicio = time.time()
    historial = modelo.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epocas,
        batch_size=config.BASELINE_BATCH_SIZE,
        verbose=verbose,
    )
    tiempo_entrenamiento = time.time() - tiempo_inicio

    # Evaluacion final en Test (unico uso del conjunto Test para esta red)
    print("\n[DEBUG] Evaluando en Test (primer y unico uso del conjunto de prueba)...")
    perdida_test, accuracy_test = modelo.evaluate(x_test, y_test, verbose=0)

    resultados = {
        "num_parametros":      num_parametros,
        "tiempo_segundos":     tiempo_entrenamiento,
        "epocas":              epocas,
        "batch_size":          config.BASELINE_BATCH_SIZE,
        "perdida_test":        float(perdida_test),
        "accuracy_test":       float(accuracy_test),
        "historial": {
            clave: [float(v) for v in valores]
            for clave, valores in historial.history.items()
        },
    }

    ruta_json = os.path.join(config.DIRECTORIO_BASE_, "resultados_base.json")
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=2)

    modelo.save(os.path.join(config.DIRECTORIO_BASE_, "modelo_base.keras"))

    mejor_val_acc = max(historial.history["val_accuracy"])
    print("\n" + "=" * 55)
    print("RESULTADOS FASE 1")
    print("=" * 55)
    print(f"  Mejor val_accuracy (entrenamiento): {mejor_val_acc:.4f}")
    print(f"  Test Accuracy:                       {accuracy_test:.4f}")
    print(f"  Test Loss:                           {perdida_test:.4f}")
    print(f"  Parametros entrenables:              {num_parametros:,}")
    print(f"  Tiempo total:                        {tiempo_entrenamiento:.1f}s")
    print(f"[DEBUG] Resultados guardados en: {ruta_json}")

    return modelo, resultados


if __name__ == "__main__":
    import sys
    epocas_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    entrenar_y_evaluar_base(epocas=epocas_arg)
