# Glosario de Parametros del Proyecto

A diferencia de los **hiperparametros** (que nosotros definimos antes de entrenar), los **parametros** son los valores internos que la red neuronal aprende por si sola durante el entrenamiento.

---

## 1. Que son los Parametros Entrenables (TP)?

Los parametros entrenables son los **pesos y sesgos** de la red. Son los numeros que el optimizador ajusta en cada paso de entrenamiento para reducir el error.

- **Peso (weight):** Un numero que escala la influencia de una entrada sobre una neurona.
- **Sesgo (bias):** Un numero adicional que desplaza la activacion de una neurona, permitiendole activarse incluso si todas las entradas son cero.

### Como se cuentan?

Para una capa `Dense` con N neuronas de entrada y M neuronas de salida:
```
Parametros = (N x M) pesos + M sesgos = N*M + M
```

Para una capa `Conv2D` con K filtros de tamano (f, f) y C canales de entrada:
```
Parametros = K * (f * f * C + 1)
             ↑    ↑         ↑
          filtros  pesos    sesgo por filtro
```

---

## 2. Parametros de Configuracion del Experimento

Estos son los parametros de configuracion del dataset y del entorno, que no cambian durante el experimento:

### `SEED = 42`
El numero que inicializa el generador de numeros aleatorios. Garantiza que los resultados sean reproducibles en cualquier maquina.

### `FRACCION_TRAIN = 0.60`
Proporcion del dataset total destinada al entrenamiento. El 60% de 60,000 imagenes = 36,000 imagenes.

### `FRACCION_VAL = 0.20`
Proporcion destinada a la validacion. 20% = 12,000 imagenes.

### `FRACCION_TEST = 0.20`
Proporcion reservada para la evaluacion final. 20% = 12,000 imagenes. **Estas imagenes no se tocan hasta la Fase 5.**

### `NUM_CLASES = 10`
El numero de categorias posibles en CIFAR-10. Determina el numero de neuronas en la capa de salida final.

### `FORMA_IMAGEN = (32, 32, 3)`
Las dimensiones de cada imagen: 32 pixeles de ancho, 32 de alto, 3 canales de color (Rojo, Verde, Azul).

---

## 3. Parametros de la Red Base (Baseline)

Estos son los hiperparametros fijos de la red de referencia, disenada manualmente:

| Constante | Valor | Descripcion |
|---|---|---|
| `BASELINE_FILTROS_CAPA1` | 32 | Filtros en la primera capa Conv2D |
| `BASELINE_FILTROS_CAPA2` | 64 | Filtros en la segunda capa Conv2D |
| `BASELINE_FORMA_FILTRO` | (3, 3) | Tamano del kernel convolucional |
| `BASELINE_FORMA_POOLING` | (2, 2) | Tamano de la ventana de MaxPooling |
| `BASELINE_NEURONAS_DENSA` | 128 | Neuronas en la capa Dense oculta |
| `BASELINE_EPOCAS` | 30 | Ciclos completos de entrenamiento |
| `BASELINE_BATCH_SIZE` | 64 | Imagenes por lote |
| `BASELINE_TASA_APRENDIZAJE` | 0.001 | Velocidad de ajuste de pesos |

**Parametros entrenables estimados del Baseline:**
Con estas configuraciones, la red base tiene aproximadamente **122,570 parametros entrenables** (el numero exacto lo calcula `modelo.count_params()` en tiempo de ejecucion).

---

## 4. Parametros del Cromosoma (Espacio de Busqueda del GA)

El GA busca dentro del siguiente espacio discreto de valores:

| Gen | Nombre | Espacio de Valores |
|---|---|---|
| 0 | `learning_rate` | [0.01, 0.005, 0.001, 0.0005, 0.0001] |
| 1 | `batch_size` | [32, 64, 128] |
| 2 | `optimizador` | [0=Adam, 1=RMSprop, 2=SGD] |
| 3 | `num_capas_conv` | [1, 2, 3] |
| 4 | `filtros_capa_1` | [16, 32, 64, 128, 256] |
| 5 | `filtros_capa_2` | [16, 32, 64, 128, 256] |
| 6 | `filtros_capa_3` | [16, 32, 64, 128, 256] |
| 7 | `kernel_size` | [3, 5] |
| 8 | `neuronas_densa` | [64, 128, 256, 512] |
| 9 | `dropout` | [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] |

> Los genes 5 y 6 (filtros de capa 2 y 3) se ignoran si `num_capas_conv < 2` o `< 3` respectivamente (mecanismo de enmascaramiento).
