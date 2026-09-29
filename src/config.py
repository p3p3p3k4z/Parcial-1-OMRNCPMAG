"""
config.py - Configuracion Global del Proyecto
Parcial 1: Optimizacion de CNN con Algoritmos Geneticos (CIFAR-10)

Proposito:
Centralizar las constantes del experimento para evitar valores magicos 
dispersos por el codigo.

Glosario:
- SEED: Numero fijo que hace los experimentos reproducibles.
- CIFAR-10: Dataset de 60,000 imagenes divididas en 10 clases.
- Baseline: La red de referencia sin optimizar para comparar resultados.
- GA / AG: Algoritmo Genetico, tecnica de optimizacion.
- Fitness: Puntuacion que evalua la calidad de una solucion.
- TP (Parametros Entrenables): Numero de pesos que la red ajusta.
"""
import os

DIRECTORIO_BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRECTORIO_RESULTS = os.path.join(DIRECTORIO_BASE, "results")
DIRECTORIO_BASE_   = os.path.join(DIRECTORIO_RESULTS, "baseline")
DIRECTORIO_GA      = os.path.join(DIRECTORIO_RESULTS, "ga")
DIRECTORIO_FINAL   = os.path.join(DIRECTORIO_RESULTS, "final")
DIRECTORIO_REPORTE = os.path.join(DIRECTORIO_BASE, "report")

for directorio in (
    DIRECTORIO_RESULTS,
    DIRECTORIO_BASE_,
    DIRECTORIO_GA,
    DIRECTORIO_FINAL,
    DIRECTORIO_REPORTE,
):
    os.makedirs(directorio, exist_ok=True)


SEED = 42

FRACCION_TRAIN = 0.60
FRACCION_VAL   = 0.20
FRACCION_TEST  = 0.20

NUM_CLASES   = 10
FORMA_IMAGEN = (32, 32, 3)
NOMBRES_CLASES = [
    "avion", "automovil", "pajaro", "gato", "venado",
    "perro", "rana", "caballo", "barco", "camion",
]

# Arquitectura Baseline (Fase 1)
# Estructura: Entrada -> 2 bloques (Conv2D + MaxPooling) -> Flatten -> Dense -> Salida
BASELINE_FILTROS_CAPA1    = 32
BASELINE_FILTROS_CAPA2    = 64
BASELINE_FORMA_FILTRO     = (3, 3)
BASELINE_FORMA_POOLING    = (2, 2)
BASELINE_NEURONAS_DENSA   = 128
BASELINE_EPOCAS           = 30
BASELINE_BATCH_SIZE       = 64
BASELINE_TASA_APRENDIZAJE = 0.001


# Parametros del Algoritmo Genetico (Fase 2)
# Poblacion: Conjunto de soluciones candidatas evaluadas.
# Generacion: Un ciclo de evaluacion, seleccion, cruce y mutacion.
# Elitismo: Mejores individuos que pasan sin cambios a la siguiente generacion.
GA_TAMANIO_POBLACION   = 12
GA_NUM_GENERACIONES    = 12
GA_NUM_PADRES          = 6
GA_EPOCAS_INDIVIDUO    = 5
GA_ELITISMO            = 2
GA_PORCENTAJE_MUTACION = 25
GA_TIPO_CRUCE          = "single_point"
GA_TIPO_MUTACION       = "random"
GA_SELECCION_PADRES    = "sss"

EPOCAS_REENTRENAMIENTO_FINAL = 30

MAX_CAPAS_CONV = 3
MIN_CAPAS_CONV = 1

# Funcion de Fitness Multiobjetivo
# fitness = w1 * val_accuracy + w2 * (1 - (TP - TP_min) / (TP_max - TP_min))
# Se resta la eficiencia para premiar redes compactas (con pocos parametros).
# TP_min y TP_max normalizan el termino de eficiencia al rango 0 a 1.
FITNESS_W1 = 0.7
FITNESS_W2 = 0.3
TP_MIN = 10_000
TP_MAX = 1_000_000

assert abs(FITNESS_W1 + FITNESS_W2 - 1.0) < 1e-9, (
    f"w1 + w2 debe ser 1.0, pero es {FITNESS_W1 + FITNESS_W2}"
)

# Traduccion de indice genetico al nombre del optimizador en Keras
MAPA_OPTIMIZADOR = {
    0: "adam",
    1: "rmsprop",
    2: "sgd",
}
