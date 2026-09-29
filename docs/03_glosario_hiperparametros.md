# Glosario de Hiperparametros del Proyecto

Los hiperparametros son las configuraciones que se deciden **antes** de entrenar la red. A diferencia de los parametros (pesos), los hiperparametros no los aprende la red sola: los define el experimentador (o el Algoritmo Genetico).

---

## Hiperparametros de la Red Neuronal (CNN)

### `learning_rate` — Tasa de Aprendizaje
**Que es:** La velocidad con la que el optimizador ajusta los pesos de la red en cada paso de entrenamiento.

**Valores en este proyecto:** `[0.01, 0.005, 0.001, 0.0005, 0.0001]`

**Impacto:**
- Valor alto (ej. 0.01): La red aprende rapido pero puede "saltarse" la solucion optima y oscilar.
- Valor bajo (ej. 0.0001): La red converge con precision pero necesita muchas mas epocas.

**Valor tipico recomendado:** `0.001` (el que usa el Baseline).

---

### `batch_size` — Tamano del Lote
**Que es:** Cuantas imagenes se procesan en paralelo antes de actualizar los pesos de la red.

**Valores en este proyecto:** `[32, 64, 128]`

**Impacto:**
- Lote pequeno (32): Actualizaciones frecuentes, mas ruido, generalmente mejor generalizacion pero mas lento.
- Lote grande (128): Actualizaciones mas suaves y estables, pero puede quedarse atrapado en minimos locales.

**Nota:** El tamano del lote tambien afecta cuanta memoria RAM/VRAM se necesita.

---

### `optimizador` — Algoritmo de Optimizacion
**Que es:** El metodo que usa la red para calcular como cambiar sus pesos al cometer un error.

**Valores en este proyecto:** `0=Adam`, `1=RMSprop`, `2=SGD`

**Impacto:**
- **Adam**: Adapta la tasa de aprendizaje por parametro. Es robusto y funciona bien sin muchos ajustes. El mas usado.
- **RMSprop**: Similar a Adam, mantiene un promedio movil del cuadrado de los gradientes. Bueno para problemas con gradientes muy variables.
- **SGD**: El mas simple. Requiere ajustar bien la tasa de aprendizaje. Puede generalizar mejor que Adam en algunos casos con el ajuste correcto.

---

### `num_capas_conv` — Numero de Capas Convolucionales
**Que es:** Cuantos bloques Conv2D + MaxPooling2D tiene la red.

**Valores en este proyecto:** `[1, 2, 3]`

**Impacto:**
- Mas capas → La red puede aprender patrones mas complejos y abstractos (jerarquia de caracteristicas).
- Menos capas → La red es mas simple, mas rapida y menos propensa al sobreajuste.
- **Limitacion practica:** Con imagenes de 32x32 y kernel de 5x5, solo es posible apilar hasta 2 capas antes de que las dimensiones colapsen a 0.

---

### `filtros` — Numero de Filtros por Capa Convolucional
**Que es:** Cuantos detectores de patrones diferentes tiene cada capa Conv2D. Cada filtro produce un "mapa de caracteristicas" distinto.

**Valores en este proyecto:** `[16, 32, 64, 128, 256]` (por capa)

**Impacto:** Mas filtros = mas capacidad para detectar caracteristicas, pero muchos mas parametros entrenables.

---

### `kernel_size` — Tamano del Filtro Convolucional
**Que es:** El tamano de la ventana de analisis de cada filtro. Un kernel de `3` analiza zonas de 3x3 pixeles.

**Valores en este proyecto:** `[3, 5]`

**Impacto:**
- Kernel 3: Detecta patrones locales pequenos (bordes finos, texturas). Mas eficiente.
- Kernel 5: Captura contexto mas amplio en cada paso pero reduce mas las dimensiones de la imagen.

---

### `neuronas_densa` — Neuronas en la Capa Oculta Dense
**Que es:** Cuantas neuronas tiene la capa totalmente conectada que precede a la capa de salida.

**Valores en este proyecto:** `[64, 128, 256, 512]`

**Impacto:** Mas neuronas = mas capacidad para combinar los patrones detectados, pero mas parametros y riesgo de sobreajuste.

---

### `dropout` — Tasa de Dropout
**Que es:** La fraccion de neuronas que se apagan aleatoriamente durante cada paso del entrenamiento.

**Valores en este proyecto:** `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`

**Impacto:**
- `0.0`: Sin dropout, la red tiene maxima capacidad pero puede memorizar los datos.
- `0.2` a `0.4`: Rango tipico que mejora la generalizacion.
- `0.5`: Apaga la mitad de neuronas, muy agresivo. Puede reducir demasiado la capacidad.

---

## Hiperparametros del Algoritmo Genetico

### `GA_TAMANIO_POBLACION`
**Que es:** Cuantas arquitecturas distintas se evaluan en cada generacion.

**Valor:** `12` (minimo exigido por el enunciado: 10)

**Impacto:** Mas poblacion = explora mas el espacio de busqueda, pero tarda mas tiempo por generacion.

---

### `GA_NUM_GENERACIONES`
**Que es:** Cuantos ciclos evolutivos completos realizara el GA.

**Valor:** `12` (minimo exigido: 10)

**Impacto:** Mas generaciones = mas evolucion y mejor solucion potencial, pero tiempo de ejecucion proporcional.

---

### `GA_EPOCAS_INDIVIDUO`
**Que es:** Cuantas epocas se entrena cada arquitectura candidata durante la fase de evaluacion del GA.

**Valor:** `5`

**Por que 5 y no 30?:** Para evaluar 12 individuos x 12 generaciones = 144 entrenamientos. Con 5 epocas se obtiene una estimacion rapida de la calidad de cada arquitectura sin que el proceso tarde dias. El mejor individuo se reentrenara luego con 30 epocas completas.

---

### `GA_ELITISMO`
**Que es:** Los N mejores individuos de una generacion que pasan directamente a la siguiente sin modificarse.

**Valor:** `2`

**Por que se usa:** Garantiza que el GA no "olvide" las mejores soluciones ya encontradas. Sin elitismo, el mejor individuo podria ser eliminado por mutacion o crossover antes de llegar a la siguiente generacion.

---

### `GA_PORCENTAJE_MUTACION`
**Que es:** La probabilidad (en porcentaje) de que un gen de un hijo sufra un cambio aleatorio.

**Valor:** `25` (25% de los genes pueden mutar)

**Por que importa:** La mutacion es la fuente de diversidad genetica. Sin ella, toda la poblacion convergeria rapidamente a una solucion local y dejaria de explorar otras posibilidades.

---

### `FITNESS_W1` y `FITNESS_W2` — Pesos de la Funcion Fitness

**Que son:** Los coeficientes que ponderan cada objetivo en la funcion de fitness multiobjetivo.

**Valores:** `w1 = 0.7`, `w2 = 0.3` (deben sumar 1.0)

**Formula:**
```
fitness = w1 * val_accuracy + w2 * (1 - (TP - TP_min) / (TP_max - TP_min))
```

**Interpretacion:**
- `w1 = 0.7`: El 70% de la calificacion viene de que tan exacta es la red.
- `w2 = 0.3`: El 30% viene de que tan eficiente es (pocas parametros).

---

### `TP_MIN` y `TP_MAX` — Limites de Normalizacion de Parametros

**Que son:** Los valores minimo y maximo esperados de parametros entrenables, usados para normalizar el termino de eficiencia de la funcion fitness al rango [0, 1].

**Valores:** `TP_MIN = 10,000`, `TP_MAX = 1,000,000`

**Por que se normalizan:** Para poder sumar la exactitud (que ya esta en [0, 1]) con el termino de eficiencia (que sin normalizar podria ser un numero enorme como 500,000), ambos terminos deben estar en la misma escala.

**Calibracion:** Estos valores se estimaron en base al espacio de busqueda definido en `cromosoma.py`. Con 1-3 capas de 16-256 filtros y una Dense de 64-512, las redes del espacio de busqueda tienen entre ~10K y ~1M parametros aproximadamente.
