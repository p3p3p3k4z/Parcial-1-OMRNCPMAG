# Optimizacion de CNN con Algoritmos Geneticos (CIFAR-10)

Parcial 1 de **Computacion Flexible**: optimizacion multiobjetivo de una Red Neuronal Convolucional para clasificacion en CIFAR-10 usando **TensorFlow/Keras**, **pyGAD** y **scikit-learn**.

---

## Estructura del Proyecto

```
src/
├── config.py               → Constantes globales y configuracion del experimento
├── datos.py                → Carga y particion de CIFAR-10 (estratificada)
├── cromosoma.py            → Codificacion genetica + constructor de CNN
├── aptitud.py              → Funcion de fitness multiobjetivo
├── modelo_base.py          → Red de referencia (Baseline, Fase 1)
├── algoritmo_genetico.py   → Ejecucion del GA con pyGAD (Fases 2 y 3)
├── reentrenamiento.py      → Reentrenamiento del mejor individuo (Fase 4)
└── ejecutar_todo.py        → Orquestador del pipeline completo (5 fases)

results/
├── baseline/               → resultados_base.json  |  modelo_base.keras
├── ga/                     → resumen_ga.json  |  historial_generaciones.json  |  historial_individuos.json
└── final/                  → resultados_optimizado.json  |  modelo_optimizado.keras

report/
├── generar_reporte.py      → Genera el documento DOCX
├── fig_fitness_evolution.png
├── fig_learning_curves_baseline.png
├── fig_learning_curves_optimized.png
├── fig_accuracy_vs_params.png
└── comparison_table.json

docs/                       → Guias de estudio y glosarios pedagogicos
codigo_obsoleto/            → Versiones antiguas del codigo (no se ejecutan)
```

---

## Formula de Fitness

El GA maximiza la siguiente funcion multiobjetivo:

```
fitness = w1 * val_accuracy + w2 * (1 - (TP - TP_min) / (TP_max - TP_min))
```

- `w1 = 0.7`, `w2 = 0.3` — Premian exactitud (70%) y eficiencia de parametros (30%)
- `TP` = Total de Parametros Entrenables de la red
- `TP_min = 10,000`, `TP_max = 1,000,000`
- El termino de complejidad se limita a `[0, 1]` con `np.clip`

---

## Requisitos

- Python 3.11
- TensorFlow, pyGAD, scikit-learn, matplotlib, python-docx (ver `requirements.txt`)

---

## Instalacion

```bash
python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Como Ejecutar

### Pipeline completo (5 fases en secuencia):

```bash
python -m src.ejecutar_todo
```

### Opciones de linea de comandos:

```bash
# Ver todas las opciones disponibles
python -m src.ejecutar_todo --help

# Forzar re-ejecucion de todas las fases aunque ya existan resultados
python -m src.ejecutar_todo --force

# Ejecutar con parametros personalizados del GA
python -m src.ejecutar_todo --poblacion 15 --generaciones 15 --epocas-individuo 5

# Ejecutar fases individuales
python -m src.modelo_base           # Solo Fase 1 (baseline)
python -m src.algoritmo_genetico    # Solo Fases 2 y 3 (GA)
python -m src.reentrenamiento       # Solo Fase 4 y 5 (reentrenamiento + test)
python -m src.comparativa           # Solo Fase 5 (graficas y tabla)
```

---

## Salida Esperada en Consola

### Inicio y particion de datos:

```
[DEBUG] Cargando y particionando CIFAR-10...
[DEBUG] Train: (36000, 32, 32, 3) | Val: (12000, 32, 32, 3) | Test: (12000, 32, 32, 3)
[DEBUG] Rango de pixeles: [0.00, 1.00]
```

### Fase 1 — Red Base:

```
=======================================================
FASE 1: Entrenando Red Base (Baseline)
=======================================================
[DEBUG] Arquitectura: Conv2D(32) -> Conv2D(64) -> Dense(128) -> Softmax(10)
[DEBUG] Parametros entrenables: 122,570
[DEBUG] Epocas: 30 | Batch size: 64 | LR: 0.001
[DEBUG] Optimizador: Adam
-------------------------------------------------------
Epoch 1/30
563/563 [==============================] - acc: 0.3241 - val_accuracy: 0.4012
...
Epoch 30/30
563/563 [==============================] - acc: 0.7821 - val_accuracy: 0.6934

[DEBUG] Evaluando en Test (primer y unico uso del conjunto de prueba)...

=======================================================
RESULTADOS FASE 1
=======================================================
  Mejor val_accuracy (entrenamiento): 0.6934
  Test Accuracy:                       0.6901
  Test Loss:                           0.9023
  Parametros entrenables:              122,570
  Tiempo total:                        182.3s
[DEBUG] Resultados guardados en: results/baseline/resultados_base.json
```

### Fases 2 y 3 — Algoritmo Genetico:

```
=======================================================
FASE 2: Optimizacion con AG + Seleccion del Mejor Individuo
=======================================================
Inicio del Algoritmo Genetico con pyGAD
Poblacion: 12 | Generaciones: 12
Epocas por individuo: 5 | Genes: 10

 [gen 00 | ind 00] acc=0.4512 TP= 45,320 fitness=0.6294 (38.2s)
 [gen 00 | ind 01] acc=0.3901 TP=  8,204 fitness=0.5730 (31.1s)
 ...

Generacion 01 completada
Mejor fitness: 0.6891
Mejor arquitectura: lr=0.001, batch=64, opt=adam, capas=2, filtros=[64, 128], ...
Tiempo total: 12.3 min
...
Generacion 12 completada
Mejor fitness: 0.7203
...

Resultado Final del Algoritmo Genetico
Tiempo total: 148.5 min
Mejor fitness: 0.7203
Mejor arquitectura: lr=0.001, batch=32, opt=adam, capas=2, filtros=[64, 128], kernel=3, neuronas=256, dropout=0.2
Val Accuracy: 0.7001
Parametros: 345,610
```

### Fase 4 — Reentrenamiento:

```
=======================================================
FASE 4: Reentrenando el Mejor Individuo del GA
=======================================================
[DEBUG] Fitness GA (5 epocas):  0.7203
[DEBUG] Val accuracy GA:         0.7001
[DEBUG] Arquitectura: lr=0.001, batch=32, opt=adam, capas=2, filtros=[64, 128], ...
[DEBUG] Parametros entrenables: 345,610
[DEBUG] Epocas: 30 | Batch: 32
-------------------------------------------------------
Epoch 1/30 ...
...
=======================================================
RESULTADOS FASES 4 y 5
=======================================================
  Mejor val_accuracy (reentrenamiento): 0.7234
  Test Accuracy:                         0.7198
  Test Loss:                             0.8341
  Parametros entrenables:                345,610
  Tiempo reentrenamiento:                241.7s
```

### Fase 5 — Reporte:

```
FASE 5: Generando reporte comparativo
  Generando grafica de evolucion del fitness...
  [OK] Guardado: report/fig_fitness_evolution.png
  Generando curvas de aprendizaje del Baseline...
  [OK] Guardado: report/fig_learning_curves_baseline.png
  ...
  Tabla comparativa:
    Accuracy en Test:
      Baseline:   0.6901
      Optimizada: 0.7198
    Parametros entrenables:
      Baseline:   122570
      Optimizada: 345610
    Tiempo de entrenamiento (s):
      Baseline:   182.3
      Optimizada: 241.7
```

> **Nota:** Los valores numericos de la salida son ejemplos ilustrativos. Los reales dependen del hardware y del cromosoma que encuentre el GA.

---

## Prevencion de Fugas de Memoria

En el bucle del GA se crean y destruyen hasta `poblacion x generaciones` modelos. Cada evaluacion usa:

```python
finally:
    del modelo
    tf.keras.backend.clear_session()   # Libera VRAM/RAM de TensorFlow
    gc.collect()                        # Fuerza recoleccion de basura de Python
```

El bloque `finally` garantiza la limpieza incluso si ocurre un error durante la evaluacion.

---

## Notas de Ejecucion

- En **CPU**: el GA puede tardar varias horas (12 generaciones x 12 individuos x ~35s = ~84 min estimado).
- Si el proceso se **interrumpe**, el orquestador detecta los archivos JSON ya generados y reanuda desde donde quedo.
- Para una prueba rapida: `--poblacion 10 --generaciones 10 --epocas-individuo 3`
