"""
reentrenamiento.py - Fase 4: Reentrenar el Mejor Individuo del GA

Proposito:
Tomar el mejor cromosoma encontrado por el Algoritmo Genetico y entrenar 
la arquitectura que representa con un presupuesto completo de epocas.

Glosario:
- Mejor Individuo: El cromosoma con el fitness mas alto al finalizar el GA.
- Reentrenamiento (Full Training): Durante el GA, se entrenaron muchas redes 
  pocas epocas para evaluar rapidamente. Aqui, entrenamos la mejor red desde 
  cero con todas las epocas (30), para garantizar una comparacion justa 
  con el modelo base.
- Evaluacion en Test (Fase 5): Este es el unico momento donde la red ve los 
  datos de Test, validando su rendimiento en datos completamente nuevos.
"""
import os
import json
import time

import tensorflow as tf

from . import config
from .datos import cargar_cifar10
from .cromosoma import construir_cnn_desde_cromosoma, cromosoma_a_texto

RUTA_RESUMEN_GA      = os.path.join(config.DIRECTORIO_GA, "resumen_ga.json")
RUTA_RESULTADOS_FINAL = os.path.join(config.DIRECTORIO_FINAL, "resultados_optimizado.json")


def reentrenar_y_evaluar_mejor(epocas=None):
    """
    Entrena el mejor individuo con todas las epocas y lo evalua en Test.
    """
    epocas = epocas or config.EPOCAS_REENTRENAMIENTO_FINAL

    if not os.path.exists(RUTA_RESUMEN_GA):
        raise FileNotFoundError(
            f"No se encontro el resumen del GA en: {RUTA_RESUMEN_GA}\n"
            "Ejecuta primero el GA (Fases 2 y 3)"
        )

    with open(RUTA_RESUMEN_GA, "r", encoding="utf-8") as archivo:
        resumen_ga = json.load(archivo)

    mejor_cromosoma = resumen_ga["mejor_cromosoma"]

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = cargar_cifar10()

    tf.keras.backend.clear_session()

    modelo, params = construir_cnn_desde_cromosoma(mejor_cromosoma)
    num_parametros = modelo.count_params()

    print("\n" + "=" * 55)
    print("FASE 4: Reentrenando el Mejor Individuo del GA")
    print("=" * 55)
    print(f"[DEBUG] Fitness GA (5 epocas):  {resumen_ga.get('mejor_fitness', 'N/A'):.4f}")
    print(f"[DEBUG] Val accuracy GA:         {resumen_ga.get('mejor_val_accuracy', 'N/A')}")
    print(f"[DEBUG] Arquitectura: {cromosoma_a_texto(mejor_cromosoma)}")
    print(f"[DEBUG] Parametros entrenables: {num_parametros:,}")
    print(f"[DEBUG] Epocas: {epocas} | Batch: {params['batch_size']}")
    print("-" * 55)

    tiempo_inicio = time.time()
    historial = modelo.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epocas,
        batch_size=params["batch_size"],
        verbose=1,
    )
    tiempo_entrenamiento = time.time() - tiempo_inicio

    perdida_test, accuracy_test = modelo.evaluate(x_test, y_test, verbose=0)

    resultados = {
        "cromosoma":          mejor_cromosoma,
        "descripcion":        cromosoma_a_texto(mejor_cromosoma),
        "hiperparametros":    params,
        "num_parametros":     num_parametros,
        "epocas":             epocas,
        "batch_size":         params["batch_size"],
        "tiempo_segundos":    tiempo_entrenamiento,
        "perdida_test":       float(perdida_test),
        "accuracy_test":      float(accuracy_test),
        "historial": {
            clave: [float(v) for v in valores]
            for clave, valores in historial.history.items()
        },
        "ga_val_accuracy":    resumen_ga.get("mejor_val_accuracy"),
        "ga_fitness":         resumen_ga.get("mejor_fitness"),
    }

    with open(RUTA_RESULTADOS_FINAL, "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=2)

    modelo.save(os.path.join(config.DIRECTORIO_FINAL, "modelo_optimizado.keras"))

    mejor_val_acc_final = max(historial.history["val_accuracy"])
    print("\n" + "=" * 55)
    print("RESULTADOS FASES 4 y 5")
    print("=" * 55)
    print(f"  Mejor val_accuracy (reentrenamiento): {mejor_val_acc_final:.4f}")
    print(f"  Test Accuracy:                         {accuracy_test:.4f}")
    print(f"  Test Loss:                             {perdida_test:.4f}")
    print(f"  Parametros entrenables:                {num_parametros:,}")
    print(f"  Tiempo reentrenamiento:                {tiempo_entrenamiento:.1f}s")
    print(f"[DEBUG] Resultados guardados en: {RUTA_RESULTADOS_FINAL}")

    return modelo, resultados


if __name__ == "__main__":
    import sys
    epocas_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    reentrenar_y_evaluar_mejor(epocas=epocas_arg)
