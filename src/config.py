"""
Configuración global del proyecto: Optimización Multiobjetivo de CNN mediante
Algoritmos Genéticos (pyGAD) sobre CIFAR-10.

Todas las constantes que definen el protocolo experimental obligatorio del
enunciado (Parcial 1 - Computación Flexible) están centralizadas aquí para
garantizar reproducibilidad entre las distintas fases del experimento.
"""
import os

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline")
GA_DIR = os.path.join(RESULTS_DIR, "ga")
FINAL_DIR = os.path.join(RESULTS_DIR, "final")
REPORT_DIR = os.path.join(BASE_DIR, "report")

for d in (RESULTS_DIR, BASELINE_DIR, GA_DIR, FINAL_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# División del dataset (fija durante todo el experimento)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ---------------------------------------------------------------------------
# Arquitectura Baseline (idéntica a la utilizada en clase)
# ---------------------------------------------------------------------------
BASELINE_FILTER1_SIZE = 32
BASELINE_FILTER2_SIZE = 64
BASELINE_FILTER_SHAPE = (3, 3)
BASELINE_POOL_SHAPE = (2, 2)
BASELINE_FULLY_CONNECT_NUM = 128
BASELINE_EPOCHS = 30
BASELINE_BATCH_SIZE = 64
BASELINE_LEARNING_RATE = 0.001

# ---------------------------------------------------------------------------
# Protocolo experimental del Algoritmo Genético (pyGAD)
# ---------------------------------------------------------------------------
GA_POPULATION_SIZE = 12          # >= 10 (mínimo exigido por el enunciado)
GA_NUM_GENERATIONS = 12          # >= 10 (mínimo exigido por el enunciado)
GA_NUM_PARENTS_MATING = 6
GA_INDIVIDUAL_EPOCHS = 5         # épocas de entrenamiento por individuo
GA_KEEP_ELITISM = 2
GA_MUTATION_PERCENT_GENES = 25
GA_CROSSOVER_TYPE = "single_point"
GA_MUTATION_TYPE = "random"
GA_PARENT_SELECTION_TYPE = "sss"  # steady-state selection

FINAL_RETRAIN_EPOCHS = 30

# ---------------------------------------------------------------------------
# Restricciones arquitectónicas del espacio de búsqueda genético
# ---------------------------------------------------------------------------
MAX_CONV_LAYERS = 3
MIN_CONV_LAYERS = 1

# ---------------------------------------------------------------------------
# Función de fitness multiobjetivo
#
# fitness = w1 * val_accuracy + w2 * (1 - (TP - TP_min) / (TP_max - TP_min))
#
# NOTA METODOLÓGICA 1 (signo de la fórmula): el enunciado escribe el segundo
# término como "1 + (TP - TP_min) / (TP_max - TP_min)", pero su propio
# ejemplo numérico resuelto (val_accuracy=0.91, TP=5230 -> fitness=0.9242)
# solo es consistente con un signo "menos": 0.637 + 0.3*(1 - 4230/99000) =
# 0.9242. Usar "+" penalizaría con un fitness MAYOR a las redes con MÁS
# parámetros, lo cual contradice el objetivo explícito de "minimizar la
# cantidad de parámetros". Se implementa la versión con "-", única
# compatible con (a) el objetivo de minimización declarado y (b) el ejemplo
# numérico provisto en el propio enunciado.
#
# NOTA METODOLÓGICA 2 (TP_min / TP_max): el enunciado sugiere TP_min=1000 y
# TP_max=100000 solo a modo de EJEMPLO para ilustrar el cálculo de la
# fórmula. El propio enunciado exige justificar técnicamente estos valores,
# por lo que se recalibraron empíricamente para el espacio de búsqueda
# realmente usado en este proyecto (arquitecturas Flatten+Dense, 1-3 capas
# Conv2D+MaxPooling2D, 16-256 filtros, kernel 3 o 5, densa 64-512):
#   - Se enumeraron 120 combinaciones representativas del espacio de
#     búsqueda; el número de parámetros entrenables osciló entre ~9,900
#     (1 capa conv, 16 filtros, densa 64) y ~29.5 millones (1 capa conv,
#     256 filtros, densa 512) -- la arquitectura baseline usada en clase
#     tiene 315,722 parámetros entrenables.
#   - Los valores TP_min=1000 / TP_max=100000 del ejemplo del enunciado son
#     demasiado pequeños para este espacio (incluso el baseline los supera
#     ampliamente), por lo que con ellos el término de complejidad se
#     saturaría (clip) en 0 para casi cualquier arquitectura, anulando de
#     facto su capacidad discriminante.
#   - Se eligen en su lugar TP_min=10,000 (cercano a la arquitectura más
#     pequeña del espacio) y TP_max=1,000,000 (por encima del baseline,
#     cubriendo la gran mayoría de arquitecturas "razonables" del espacio
#     de búsqueda, dejando que solo las combinaciones más extremas -1 capa
#     conv con muchos filtros y una capa densa muy grande- saturen la
#     penalización a su valor máximo).
# El valor normalizado se recorta (clip) a [0, 1] para evitar fitness fuera
# de rango en arquitecturas fuera de estos límites.
# ---------------------------------------------------------------------------
FITNESS_W1 = 0.7   # peso de val_accuracy
FITNESS_W2 = 0.3   # peso de la penalización por complejidad (parámetros)
TP_MIN = 10000
TP_MAX = 1000000

assert abs(FITNESS_W1 + FITNESS_W2 - 1.0) < 1e-9, "w1 + w2 debe ser 1"

OPTIMIZER_MAP = {0: "adam", 1: "rmsprop", 2: "sgd"}
