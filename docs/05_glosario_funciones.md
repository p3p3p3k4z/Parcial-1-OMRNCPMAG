# Glosario de Funciones del Proyecto

Este glosario describe cada funcion del codigo fuente: que recibe, que hace y que devuelve. Esta ordenado por archivo.

---

## `datos.py`

### `cargar_cifar10()`
**Que hace:** Descarga CIFAR-10, normaliza los pixeles al rango [0, 1] y divide el dataset en tres conjuntos usando particion estratificada.

**Recibe:** Nada (usa las constantes de `config.py`).

**Devuelve:**
```python
(x_train, y_train), (x_val, y_val), (x_test, y_test)
```
Tres tuplas, cada una con las imagenes y sus etiquetas correspondientes.

**Por que es estratificada?** La funcion `train_test_split` con `stratify=y` garantiza que la proporcion de imagenes de cada clase sea la misma en los tres conjuntos. Si no se usara, podria haber conjuntos con mas gatos que aviones, lo cual sesga el entrenamiento.

---

## `cromosoma.py`

### `_calcular_profundidad_valida(num_capas_solicitadas, kernel_size, tamanio_entrada)`
**Que hace:** Simula cuanto se reduce una imagen de 32x32 al pasar por N capas de Conv2D + MaxPooling2D con un kernel dado. Detiene el conteo si la imagen colapsaria a menos de 2x2 pixeles (requisito minimo de MaxPooling).

**Recibe:** Numero de capas deseadas, tamano del kernel, tamano inicial de la imagen.

**Devuelve:** El numero maximo de capas que son fisicamente posibles con esos parametros.

**Por que existe?** Sin esta funcion, el GA podria generar un cromosoma con 3 capas y kernel=5, lo cual colapsaria las dimensiones de la imagen a 0 y causaria un error en Keras.

---

### `decodificar_cromosoma(vector)`
**Que hace:** Traduce el vector numerico del cromosoma a un diccionario de hiperparametros legibles. Aplica la validacion de profundidad y el mecanismo de enmascaramiento.

**Recibe:** `vector` — lista de numeros (el cromosoma crudo de pyGAD).

**Devuelve:** Diccionario con claves como `learning_rate`, `num_capas_conv`, `filtros`, `dropout`, etc.

---

### `cromosoma_a_texto(vector)`
**Que hace:** Llama a `decodificar_cromosoma` y convierte el resultado a una cadena de texto compacta para imprimirla en los logs.

**Recibe:** `vector` — el cromosoma crudo.

**Devuelve:** Un `str` como `"lr=0.001, batch=64, opt=adam, capas=2, filtros=[64, 128], ..."`.

---

### `construir_cnn_desde_cromosoma(vector)`
**Que hace:** Decodifica el cromosoma y construye un modelo Keras `Sequential` con la arquitectura especificada. Luego lo compila con el optimizador, la funcion de perdida y las metricas correspondientes.

**Recibe:** `vector` — el cromosoma crudo.

**Devuelve:** Una tupla `(modelo_compilado, params_dict)`. El modelo esta listo para llamar a `.fit()`.

---

## `aptitud.py`

### `_clave_cromosoma(cromosoma)`
**Que hace:** Convierte el vector flotante del cromosoma en una tupla hashable (con redondeo) para usarla como clave de un diccionario (el cache).

**Por que se redondea?** pyGAD puede generar valores como `0.0010000000001` que son matematicamente iguales a `0.001` pero no identicos como claves de diccionario.

---

### `crear_funcion_fitness(x_train, y_train, x_val, y_val, epocas, ruta_log)`
**Que hace:** Es una funcion de orden superior (fabrica de funciones). Recibe los datos de entrenamiento y validacion, y devuelve la funcion `calcular_fitness` configurada con esos datos, lista para ser pasada a pyGAD.

**Por que no se pasan los datos directamente a pyGAD?** La firma que exige pyGAD para la funcion de fitness es `fitness_func(ga_instancia, cromosoma, indice)`. No acepta parametros adicionales. La tecnica de la fabrica de funciones (closure) resuelve esto: los datos quedan "atrapados" dentro de la funcion `calcular_fitness` sin necesidad de pasarlos como argumento.

**Devuelve:** La tupla `(funcion_fitness, historial_evaluaciones)`.

---

### `calcular_fitness(ga_instancia, cromosoma, indice_solucion)` (funcion interna)
**Que hace:**
1. Revisa si el cromosoma ya fue evaluado (cache). Si si, devuelve el fitness guardado.
2. Construye la CNN con `construir_cnn_desde_cromosoma`.
3. Entrena el modelo durante `GA_EPOCAS_INDIVIDUO` epocas.
4. Calcula el fitness con la formula multiobjetivo.
5. Libera la memoria (clear_session + gc).
6. Guarda el resultado en el cache y en el log.

**Recibe:** Los tres argumentos que exige la firma de pyGAD.

**Devuelve:** Un `float` — el valor de fitness del cromosoma.

---

## `modelo_base.py`

### `construir_red_base()`
**Que hace:** Construye y compila la arquitectura CNN fija del Baseline usando las constantes de `config.py`. No recibe argumentos.

**Devuelve:** El modelo Keras compilado, listo para entrenar.

---

### `entrenar_y_evaluar_base(epocas, verbose)`
**Que hace:** Orquesta la Fase 1 completa:
1. Carga los datos.
2. Construye la red.
3. Entrena.
4. Evalua en Test.
5. Guarda resultados en `results/baseline/resultados_base.json`.

**Devuelve:** `(modelo, resultados_dict)`.

---

## `algoritmo_genetico.py`

### `_crear_callback_generacion(tiempo_inicio_ga)`
**Que hace:** Fabrica una funcion de callback que pyGAD llama automaticamente al final de cada generacion para guardar el progreso en disco e imprimir estadisticas en consola.

**Por que es una fabrica?** Necesita capturar `tiempo_inicio_ga` en su alcance para calcular el tiempo transcurrido, pero pyGAD llama al callback con una firma fija `(ga_instancia)`.

---

### `ejecutar_ga(tamanio_poblacion, num_generaciones, epocas_individuo)`
**Que hace:** Orquesta las Fases 2 y 3:
1. Carga datos.
2. Crea la funcion de fitness via `crear_funcion_fitness`.
3. Configura e instancia el objeto `pygad.GA`.
4. Ejecuta `instancia_ga.run()`.
5. Extrae el mejor cromosoma.
6. Guarda el resumen en `results/ga/resumen_ga.json`.

**Devuelve:** `(instancia_ga, resumen_dict)`.

---

## `reentrenamiento.py`

### `reentrenar_y_evaluar_mejor(epocas)`
**Que hace:** Orquesta las Fases 4 y 5:
1. Lee `resumen_ga.json` para obtener el mejor cromosoma.
2. Construye la arquitectura ganadora.
3. La entrena desde cero con epocas completas (30).
4. Evalua en Test.
5. Guarda resultados en `results/final/resultados_optimizado.json`.

**Devuelve:** `(modelo, resultados_dict)`.

---

## `comparativa.py`

### `plot_fitness_evolution()`
Lee `historial_generaciones.json` y genera la grafica de como evoluciono el mejor fitness a lo largo de las generaciones.

### `plot_learning_curves(history, title, out_path)`
Genera dos subgraficas: accuracy y loss de entrenamiento vs validacion por epoca, para cualquier historial de entrenamiento que se le pase.

### `plot_accuracy_vs_params(baseline, final)`
Genera un scatter plot con todos los individuos evaluados por el GA (puntos grises) y marca con estrellas de colores al Baseline (rojo) y al modelo Optimizado (verde).

### `build_comparison_table(baseline, final)`
Construye la tabla comparativa exigida por el enunciado (accuracy, parametros, tiempo) y la guarda en `report/comparison_table.json`.

### `generate_all()`
Funcion orquestadora que llama a todas las funciones anteriores en secuencia. Es el punto de entrada de la Fase 5.

---

## `ejecutar_todo.py`

### `_encabezado(titulo, fase_num, fase_total)`
Imprime un separador visual en consola para marcar el inicio de cada fase. Solo sirve para hacer la salida mas legible.

### `main()`
El punto de entrada del pipeline completo. Parsea los argumentos de linea de comandos (como `--force`, `--poblacion`) y ejecuta cada fase en orden, verificando primero si ya existe el archivo de resultados correspondiente para poder reanudar si el proceso se interrumpio.
