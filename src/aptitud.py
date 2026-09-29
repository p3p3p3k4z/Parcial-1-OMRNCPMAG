"""
aptitud.py - Funcion de Fitness Multiobjetivo para el Algoritmo Genetico

Proposito:
Evaluar la calidad de una arquitectura CNN codificada en un cromosoma.
El Algoritmo Genetico usa esta funcion para puntuar cada individuo y 
seleccionar a los mejores.

Glosario:
- val_accuracy (Exactitud en Validacion): Porcentaje de imagenes correctas 
  en el conjunto de validacion (0 a 1). No se evalua en Test durante el GA.
- Normalizacion de TP (Parametros Entrenables): Para sumar manzanas con 
  manzanas, convertimos el recuento de parametros a una escala de 0 a 1 
  usando limites esperados (TP_min, TP_max).
- np.clip(valor, 0, 1): Evita que valores fuera del rango esperado 
  distorsionen la formula.
- Cache de fitness: Diccionario que recuerda puntuaciones previas. Evita
  reentrenar individuos que sobreviven por elitismo, ahorrando mucho tiempo.
- Limpieza de memoria (clear_session + gc.collect): Cada evaluacion crea 
  un nuevo modelo en Keras. Si no se libera la memoria tras evaluarlo, 
  el sistema se queda sin RAM y falla prematuramente.
"""
import gc
import json
import time

import numpy as np
import tensorflow as tf

from . import config
from .cromosoma import construir_cnn_desde_cromosoma, cromosoma_a_texto

# Diccionario para evitar reentrenar arquitecturas ya evaluadas
_cache_fitness = {}
_historial_evaluaciones = []


def _clave_cromosoma(cromosoma):
    """
    Convierte el vector de genes a una clave hashable para el cache.
    Redondea para manejar minimas diferencias de punto flotante.
    """
    return tuple(round(float(gen), 6) for gen in cromosoma)


def crear_funcion_fitness(x_train, y_train, x_val, y_val,
                           epocas=None, ruta_log=None):
    """
    Inyecta datos y parametros, devolviendo la funcion que evalua 
    cada individuo (cumpliendo la firma requerida por pyGAD).
    """
    epocas = epocas or config.GA_EPOCAS_INDIVIDUO

    def calcular_fitness(ga_instancia, cromosoma, indice_solucion):
        """
        Entrena el modelo propuesto, evalua su accuracy y su complejidad,
        y retorna una calificacion (fitness) a maximizar.
        """
        clave = _clave_cromosoma(cromosoma)
        if clave in _cache_fitness:
            return _cache_fitness[clave]["fitness"]

        modelo = None
        val_accuracy = 0.0
        num_parametros = 0
        fitness = 0.0
        tiempo_inicio = time.time()

        try:
            modelo, params = construir_cnn_desde_cromosoma(cromosoma)
            num_parametros = modelo.count_params()

            historial = modelo.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epocas,
                batch_size=params["batch_size"],
                verbose=0,
            )

            val_accuracy = float(max(historial.history["val_accuracy"]))

            # Formula de fitness (Maximizar)
            # w1 premia la exactitud, w2 penaliza la cantidad de parametros
            proporcion_complejidad = np.clip(
                (num_parametros - config.TP_MIN) / (config.TP_MAX - config.TP_MIN),
                0.0,
                1.0,
            )
            fitness = (
                config.FITNESS_W1 * val_accuracy
                + config.FITNESS_W2 * (1.0 - proporcion_complejidad)
            )

        except Exception as error:
            # Los modelos invalidos (ej. colapso dimensional) reciben 0 de fitness
            print(f" [AVISO] Individuo gen {ga_instancia.generations_completed} "
                  f"idx {indice_solucion} fallo: {error}")
            fitness = 0.0

        finally:
            # Bloque para prevencion critica de fugas de memoria
            del modelo
            tf.keras.backend.clear_session()
            gc.collect()

        tiempo_transcurrido = time.time() - tiempo_inicio
        generacion_actual = int(ga_instancia.generations_completed)

        registro = {
            "generacion":      generacion_actual,
            "indice":          int(indice_solucion),
            "cromosoma":       list(map(float, cromosoma)),
            "descripcion":     cromosoma_a_texto(cromosoma),
            "val_accuracy":    val_accuracy,
            "num_parametros":  num_parametros,
            "fitness":         float(fitness),
            "tiempo_segundos": tiempo_transcurrido,
        }
        _cache_fitness[clave] = registro
        _historial_evaluaciones.append(registro)

        if ruta_log:
            with open(ruta_log, "w", encoding="utf-8") as archivo:
                json.dump(_historial_evaluaciones, archivo, indent=2)

        print(
            f" [gen {generacion_actual:02d} | ind {indice_solucion:02d}] "
            f"acc={val_accuracy:.4f} TP={num_parametros:7,} "
            f"fitness={fitness:.4f} ({tiempo_transcurrido:.1f}s)"
        )

        return fitness

    return calcular_fitness, _historial_evaluaciones
