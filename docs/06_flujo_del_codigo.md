# Guia de Flujo del Codigo

Esta guia explica como se conectan todos los modulos del proyecto, desde que ejecutas el comando hasta que obtienes el reporte final.

---

## Vision General: El Pipeline de 5 Fases

```
python -m src.ejecutar_todo
         |
         v
  ejecutar_todo.py (Orquestador)
    |       |       |         |
    v       v       v         v
 Fase 1  Fase 2+3  Fase 4  Fase 5
baseline   GA    reentren. reporte
```

El orquestador (`ejecutar_todo.py`) llama a los demas modulos en orden. Si un archivo de resultados ya existe en disco, omite esa fase y continua desde la siguiente.

---

## Flujo Detallado Fase a Fase

### Fase 1 — Red Base

```
ejecutar_todo.py
    └─ llama a: entrenar_y_evaluar_base()
                    |
                    ├─ cargar_cifar10()          <- datos.py
                    |     Descarga, normaliza y divide CIFAR-10
                    |
                    ├─ construir_red_base()       <- modelo_base.py
                    |     Apila Conv2D, MaxPooling, Flatten, Dense, Softmax
                    |     Compila con Adam + sparse_categorical_crossentropy
                    |
                    ├─ modelo.fit(x_train, x_val) <- Keras (30 epocas)
                    |
                    ├─ modelo.evaluate(x_test)    <- Primera y ultima vez con Test
                    |
                    └─ Guarda: results/baseline/resultados_base.json
                               results/baseline/modelo_base.keras
```

---

### Fases 2 y 3 — Algoritmo Genetico

```
ejecutar_todo.py
    └─ llama a: ejecutar_ga()
                    |
                    ├─ cargar_cifar10()           <- datos.py
                    |
                    ├─ crear_funcion_fitness(...)  <- aptitud.py
                    |     Fabrica la funcion calcular_fitness
                    |     con los datos "atrapados" dentro
                    |
                    └─ pygad.GA(...).run()
                            |
                            | (por cada individuo en cada generacion)
                            v
                        calcular_fitness(ga, cromosoma, idx)
                            |
                            ├─ decodificar_cromosoma()    <- cromosoma.py
                            |     Traduce el vector a hiperparametros
                            |
                            ├─ construir_cnn_desde_cromosoma()  <- cromosoma.py
                            |     Construye y compila el modelo
                            |
                            ├─ modelo.fit(x_train, x_val) <- Keras (5 epocas)
                            |
                            ├─ Calcula fitness multiobjetivo
                            |
                            └─ Limpieza de memoria (clear_session, gc)

                    (al terminar cada generacion)
                    └─ al_terminar_generacion(ga)
                            └─ Guarda: results/ga/historial_generaciones.json

                    (al terminar todas las generaciones)
                    └─ Extrae mejor cromosoma
                       Guarda: results/ga/resumen_ga.json
                               results/ga/historial_individuos.json
```

---

### Fase 4 — Reentrenamiento

```
ejecutar_todo.py
    └─ llama a: reentrenar_y_evaluar_mejor()
                    |
                    ├─ Lee: results/ga/resumen_ga.json
                    |     Obtiene el mejor cromosoma encontrado
                    |
                    ├─ cargar_cifar10()               <- datos.py
                    |
                    ├─ construir_cnn_desde_cromosoma() <- cromosoma.py
                    |     Reconstruye la misma arquitectura desde cero
                    |     (pesos aleatorios, entrenamiento limpio)
                    |
                    ├─ modelo.fit(x_train, x_val)     <- Keras (30 epocas)
                    |
                    ├─ modelo.evaluate(x_test)        <- Evaluacion final
                    |
                    └─ Guarda: results/final/resultados_optimizado.json
                               results/final/modelo_optimizado.keras
```

---

### Fase 5 — Reporte Comparativo

```
ejecutar_todo.py
    └─ llama a: generate_all()
                    |
                    ├─ Lee: results/baseline/resultados_base.json
                    ├─ Lee: results/final/resultados_optimizado.json
                    ├─ Lee: results/ga/historial_generaciones.json
                    ├─ Lee: results/ga/historial_individuos.json
                    |
                    ├─ plot_fitness_evolution()
                    |     Grafica: mejor fitness por generacion del GA
                    |
                    ├─ plot_learning_curves(baseline)
                    |     Grafica: accuracy/loss del Baseline por epoca
                    |
                    ├─ plot_learning_curves(optimizado)
                    |     Grafica: accuracy/loss del modelo Optimizado por epoca
                    |
                    ├─ plot_accuracy_vs_params()
                    |     Scatter: todos los individuos GA + Baseline + Optimizado
                    |
                    └─ build_comparison_table()
                          Tabla: accuracy, parametros, tiempo de ambos modelos
                          Guarda: report/comparison_table.json
```

---

## Dependencias entre Modulos

```
config.py
    ← datos.py
    ← cromosoma.py
        ← aptitud.py
        ← modelo_base.py
        ← reentrenamiento.py
            ← algoritmo_genetico.py
                ← ejecutar_todo.py
    ← comparativa.py
        ← ejecutar_todo.py
```

`config.py` es el unico modulo del que dependen todos los demas. Ningun modulo depende de otro en circulo (no hay dependencias ciclicas).

---

## Archivos Generados en Disco

Tras una ejecucion completa, el proyecto genera los siguientes archivos:

```
results/
├── baseline/
│   ├── resultados_base.json        <- Metricas de la red base (accuracy, TP, tiempo, historial)
│   └── modelo_base.keras           <- Pesos entrenados del Baseline
│
├── ga/
│   ├── historial_generaciones.json <- Mejor fitness de cada generacion
│   ├── historial_individuos.json   <- Registro de cada evaluacion (todos los individuos)
│   └── resumen_ga.json             <- Mejor cromosoma encontrado y sus metricas
│
└── final/
    ├── resultados_optimizado.json  <- Metricas del modelo optimizado (accuracy, TP, tiempo)
    └── modelo_optimizado.keras     <- Pesos del modelo optimizado

report/
├── fig_fitness_evolution.png       <- Curva de evolucion del fitness
├── fig_learning_curves_baseline.png
├── fig_learning_curves_optimized.png
├── fig_accuracy_vs_params.png      <- Scatter Accuracy vs Parametros
└── comparison_table.json           <- Tabla comparativa final
```

---

## Reanudacion Automatica

Si el pipeline se interrumpe (por apagado, error de GPU, etc.), se puede reanudar simplemente ejecutando de nuevo:

```bash
python -m src.ejecutar_todo
```

El orquestador detecta que archivos JSON ya existen y omite esas fases. Solo ejecuta las que faltan. Para forzar la re-ejecucion de todo desde cero:

```bash
python -m src.ejecutar_todo --force
```
