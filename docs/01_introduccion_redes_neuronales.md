# Introduccion a Redes Neuronales Artificiales

Esta guia es el punto de partida del proyecto. Explica desde cero que es una red neuronal, como aprende y por que se usa para clasificar imagenes.

---

## 1. Que es una Red Neuronal?

Una red neuronal artificial esta inspirada en el cerebro humano. El cerebro tiene billones de neuronas que se comunican entre si para procesar informacion. Las redes artificiales imitan esta idea con matematicas simples.

Una neurona artificial recibe varios numeros como entrada, los multiplica por unos valores llamados **pesos** (que son lo que "aprende"), los suma, y decide si pasa esa informacion hacia adelante o no.

La clave de todo el proceso es: **la red se equivoca, mide cuanto se equivoco, y ajusta sus pesos para equivocarse menos la proxima vez.** Este ciclo se repite miles de veces.

---

## 2. Como Aprende la Red?

El proceso de aprendizaje tiene tres actores principales:

### El Optimizador (el "Profesor")
Despues de que la red intenta resolver un problema y se equivoca, el optimizador le dice **como ajustar los pesos** para mejorar. Los optimizadores mas comunes son:

| Optimizador | Descripcion simple |
|---|---|
| **Adam** | El mas popular. Adapta automaticamente el tamano de los pasos segun el historial de errores. |
| **SGD** (Descenso por Gradiente Estocastico) | El clasico. Da pasos fijos hacia la direccion donde el error disminuye. |
| **RMSprop** | Parecido a Adam pero con otra estrategia de adaptacion. Funciona bien en redes recurrentes. |

### La Funcion de Perdida (el "Examen")
Mide que tan equivocada esta la red. En clasificacion de imagenes con muchas clases (como CIFAR-10), se usa **Categorical Cross-Entropy** (o su version esparsa para etiquetas enteras). Entre mas alta la perdida, mas equivocada esta la red.

### La Tasa de Aprendizaje (el "Tamano del Paso")
Controla cuanto cambian los pesos en cada corrida del optimizador.
- Muy alta → La red aprende rapido pero puede volverse inestable y oscilar sin converger.
- Muy baja → La red es estable pero tarda demasiado en aprender.

---

## 3. Tipos de Capas en una CNN

Una Red Neuronal Convolucional (CNN) es una arquitectura especializada para imagenes. Apila diferentes tipos de capas en secuencia.

### Conv2D (Capa Convolucional)
Es el corazon de una CNN. Aplica **filtros** (pequenas matrices de numeros) que se deslizan por la imagen buscando patrones locales: bordes, texturas, curvas.

Parametros clave:
- **filters**: Cuantos patrones distintos busca esta capa. Mas filtros = mas capacidad para detectar cosas, pero mas parametros.
- **kernel_size**: Que tan grande es el "lente" de busqueda. Un kernel de `(3, 3)` analiza zonas de 3x3 pixeles.
- **activation**: La funcion que decide si la neurona "se enciende". Casi siempre se usa `ReLU`.

### MaxPooling2D (Capa de Agrupacion)
Reduce el tamano de la imagen tomando solo el valor maximo de cada zona. Sirve para:
- Reducir la cantidad de datos (y con ello los calculos necesarios).
- Hacer que la red sea robusta a pequenos desplazamientos del objeto en la imagen (si el gato se mueve un pixel a la izquierda, sigue siendo un gato).

Parametro clave:
- **pool_size**: El tamano de la zona a resumir. `(2, 2)` reduce la imagen a la mitad en cada dimension.

### Dropout (Capa de Regularizacion)
Durante el entrenamiento, apaga aleatoriamente un porcentaje de neuronas en cada paso. Esto fuerza al resto a aprender de forma mas independiente y general, reduciendo el **sobreajuste** (cuando la red memoriza los datos en lugar de aprender patrones generales).

Parametro clave:
- **rate**: Fraccion de neuronas que se apagan. `0.2` apaga el 20%.

### Flatten (Capa de Aplanamiento)
Convierte la salida 3D de las capas convolucionales (ancho, alto, canales) en un vector 1D. Es un puente obligatorio antes de las capas Dense.

### Dense (Capa Totalmente Conectada)
Cada neurona de esta capa recibe la salida de **todas** las neuronas de la capa anterior. Es la parte "clasica" de la red, donde se aprenden combinaciones de alto nivel de los patrones detectados por las convoluciones.

Parametros clave:
- **units**: Cuantas neuronas tiene la capa.
- **activation**: La funcion de activacion. La ultima capa usa `softmax` para producir probabilidades.

### Softmax (Funcion de Activacion de Salida)
Convierte los numeros brutos de salida de la ultima capa en **probabilidades** que suman 1.0. La clase con la probabilidad mas alta es la prediccion de la red.

---

## 4. El Dataset CIFAR-10

CIFAR-10 es un dataset estandar de referencia en vision computacional. Contiene:
- **60,000 imagenes** de 32x32 pixeles en color (3 canales: Rojo, Verde, Azul).
- **10 clases** balanceadas (6,000 imagenes por clase):
  `avion`, `automovil`, `pajaro`, `gato`, `venado`, `perro`, `rana`, `caballo`, `barco`, `camion`

### Por que normalizamos las imagenes?
Los pixeles originales tienen valores entre 0 y 255. Dividirlos por 255 los lleva al rango [0, 1]. Las redes neuronales aprenden de forma mas estable con entradas en rangos pequenos porque los pesos internos tambien operan en escalas similares.

### Por que dividimos en Train / Val / Test?

| Conjunto | Proposito | Tamano en este proyecto |
|---|---|---|
| **Train** | La red estudia de estos datos y ajusta sus pesos. | 60% |
| **Validacion** | Se usa para monitorear el progreso durante el entrenamiento y ajustar hiperparametros. La red no aprende directamente de el. | 20% |
| **Test** | Se usa **una sola vez** al final para medir el rendimiento real ante datos completamente nuevos. | 20% |

> Si usaramos el Test durante el entrenamiento, estariamos "contaminando" la evaluacion: la red habria visto esos datos y la medida no seria honesta.
