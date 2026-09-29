# Diccionario de Analogias

Esta guia traduce toda la terminologia tecnica del proyecto a situaciones cotidianas. Ideal para afianzar los conceptos antes de un examen o presentacion.

---

## Redes Neuronales

### La Semilla (`SEED = 42`)
Imagina que dos personas juegan al mismo videojuego con el mismo "codigo de mundo" (seed). Ambos veran exactamente el mismo mapa con las mismas montanas y rios. En nuestro experimento, la semilla garantiza que si alguien mas ejecuta el mismo codigo, obtendra exactamente la misma division de datos y los mismos resultados aleatorios. Sin ella, cada ejecucion seria diferente y no podriamos comparar resultados de forma justa.

### Dataset CIFAR-10
Es como una enciclopedia ilustrada con 60,000 fotografias pequenas, organizada en 10 capitulos (aviones, perros, gatos...). La tarea de la red es aprender a identificar a que capitulo pertenece cualquier fotografia nueva que le mostremos.

### Entrenamiento, Validacion y Test
- **Train (Entrenamiento):** Los ejercicios del libro de texto. La red los resuelve y el profesor le da las respuestas para que aprenda.
- **Validacion:** El simulacro de examen semanal. Sirve para medir si la red esta aprendiendo de verdad o solo memorizando los ejercicios.
- **Test:** El examen final de carrera. Se toma una sola vez al finalizar todo el proceso. La red nunca habia visto estas preguntas.

### Normalizacion de Pixeles (`/ 255.0`)
Es como cambiar una regla de "medir en pies del 0 al 837" a "medir en metros del 0 al 1". Las matematicas son mas faciles y estables cuando todos los numeros estan en la misma escala pequeña.

---

## Arquitectura de la CNN

### Conv2D (Filtros Convolucionales)
Imagina que tienes una linterna pequena y exploras un cuarto oscuro. Cada vez que la mueves, iluminas una zona de 3x3 cm y anotas lo que ves. Si tienes 32 linternas diferentes (filtros), cada una busca algo distinto: una busca bordes verticales, otra esquinas, otra texturas rugosas. Al final tienes 32 mapas diferentes de la misma imagen.

### MaxPooling2D
Acabas de tomar 100 fotos de un parque y necesitas resumirlas en 25. Te quedas con las 25 que tienen la escena mas importante (el valor maximo). El MaxPooling hace lo mismo: de cada zona de 2x2 pixeles, conserva solo el valor mas alto, reduciendo la imagen a la mitad y guardando lo mas relevante.

### Flatten (Aplanamiento)
Tienes un cubo de hielo (la imagen en 3D con canales de color). Para meterlo en una pajita (la capa Dense, que solo acepta listas), necesitas derretirlo hasta que se convierta en un hilillo de agua. Flatten hace exactamente eso: aplasta la estructura 3D en una lista 1D larga.

### Capa Dense (Totalmente Conectada)
Es como una reunion de directivos donde todos pueden hablar con todos. Cada neurona de la sala recibe informacion de todas las neuronas de la sala anterior y juntas toman la decision final: "esto es un gato con 87% de probabilidad".

### Dropout (Regularizacion)
Durante el entrenamiento, apagamos aleatoriamente un 20% de las neuronas en cada turno. Es como hacer un examen en grupo donde el profesor elige al azar quien puede responder cada pregunta. Obliga a todos a prepararse bien, porque nunca sabes si te tocara hablar. Resultado: la red aprende patrones generales en lugar de depender de unas pocas neuronas "memorizadoras".

### Softmax (Funcion de Salida)
Imagina que la red es un panel de jueces que le da puntos a cada una de las 10 clases. Softmax convierte esos puntos brutos en porcentajes que suman 100%. Si el panel dice: avion=8, barco=1, resto≈0, softmax convierte eso en: avion=89%, barco=9%, etc.

---

## Proceso de Entrenamiento

### Epocas (`Epochs`)
Leer el mismo libro 30 veces (30 epocas). La primera vez apenas entiendes. A la decima ya identificas los temas principales. A la trigesima te lo sabes casi de memoria (aunque tampoco queremos eso exactamente, pues la red podria memorizar en lugar de entender).

### Batch Size
Tienes que mover 36,000 ladrillos (imagenes) de un edificio a otro. Si intentas cargar todos de golpe, tu espalda colapsa (la RAM se satura). En cambio, los cargas de 64 en 64 con una carretilla. Cada viaje de carretilla es un "batch". Solo ajustas tu postura (los pesos) despues de cada carretilla, no despues de cada ladrillo.

### Tasa de Aprendizaje
Eres un explorador buscando el punto mas bajo de un valle (el minimo error) con los ojos vendados. La tasa de aprendizaje decide que tan grandes son tus pasos.
- Pasos enormes (0.01): Llegas rapido pero puedes saltar sobre el punto exacto y quedarte oscilando.
- Pasos de hormiga (0.0001): Muy preciso, pero tardas una eternidad en llegar.
- Pasos moderados (0.001): El equilibrio que suele funcionar mejor.

### Optimizador (Adam, SGD, RMSprop)
Es el GPS del explorador. SGD camina siempre hacia el punto mas bajo visible. Adam es mas listo: recuerda donde ha caminado y adapta el tamano de sus pasos segun el terreno. Por eso Adam suele funcionar mejor sin necesidad de mucha configuracion.

---

## Algoritmos Geneticos

### El Cromosoma
Es una receta de cocina. Nos dice exactamente como construir una red neuronal: cuanta sal (learning rate), cuanto tiempo en el horno (epocas), que ingredientes (filtros, neuronas). Dos recetas diferentes producen redes diferentes con distintos niveles de calidad.

### El Gen
Es cada ingrediente de la receta. El Gen 0 es "cuantas cucharadas de learning rate", el Gen 3 es "cuantas capas de convoluciones". Cambiar un solo gen puede mejorar o arruinar el plato.

### El Espacio de Busqueda
Es el menu completo de ingredientes disponibles. No puedes usar cualquier cantidad; solo puedes elegir entre las opciones del menu: "learning rate de 0.01, 0.005, 0.001, 0.0005 o 0.0001". El GA busca la mejor combinacion de este menu.

### Enmascaramiento (Genes Inactivos)
Imagina que la receta siempre reserva espacio para 3 capas de toppings en un pastel, pero si decides hacer un pastel sencillo con solo 1 capa, los espacios de las capas 2 y 3 simplemente se dejan en blanco (se ignoran). Los ingredientes siguen en la lista de compras pero no se usan esta vez. En una proxima "generacion" podrian reactivarse.

### Poblacion y Generacion
- **Poblacion:** Una clase de 12 estudiantes (arquitecturas), cada uno con su propio metodo de estudio (cromosoma).
- **Generacion:** Un semestre academico. Al final del semestre, los peores estudiantes repiten y los mejores avanzan y se convierten en mentores de la nueva generacion.

### Elitismo
El campeon de Olimpiadas clasifica directamente a la final sin tener que competir en las eliminatorias. El elitismo protege a los 2 mejores individuos de cada generacion para que no puedan ser "arruinados" por una mutacion accidental. El GA nunca puede retroceder a una solucion peor que la que ya encontro.

### Cruce (Crossover)
Es la reproduccion genetica. Tomamos las primeras 5 genes del Padre A y las ultimas 5 genes del Padre B para crear un hijo. Esperamos que el hijo herede la "inteligencia" de su padre (tal vez su buen learning rate) y la "estructura" de su madre (tal vez sus buenos filtros).

### Mutacion
De vez en cuando, un hijo nace con una caracteristica que ninguno de sus padres tenia (un gen cambia a un valor aleatorio distinto). Esto evita que todos los individuos se parezcan demasiado entre si y que el GA quede atrapado en una solucion "buena pero no la mejor". Es el origen de la diversidad genetica.

### Funcion de Fitness (w1 y w2)
Es el sistema de calificacion del "concurso de arquitecturas". Una red gana puntos por dos cosas:
- **w1 = 70%:** Cuantas imagenes clasifica correctamente (exactitud). Como las notas de los examenes.
- **w2 = 30%:** Que tan eficiente es (pocos parametros). Como la asistencia y participacion.

El mejor "estudiante" no es necesariamente el que saco el mejor examen, sino el que tiene el mejor promedio ponderado de notas mas asistencia.

---

## Conceptos de Evaluacion

### Sobreajuste (Overfitting)
La red "memoriza" los ejercicios del libro en lugar de entender el tema. En el examen final (Test), cuando le dan preguntas nuevas, fracasa porque nunca aprendio a generalizar. Se detecta cuando el accuracy de Train es mucho mas alto que el de Validacion.

### Generalizacion
La red aprendio los conceptos subyacentes, no las respuestas de memoria. Cuando le muestras imagenes que nunca habia visto, las clasifica bien. Un buen modelo tiene accuracy de Train y Validacion similares (y cercanos entre si).

### Trade-off Exactitud vs. Eficiencia
El dilema central del proyecto: una red muy grande puede ser mas exacta pero gasta mas memoria y tiempo. Una red pequena es rapida y eficiente pero puede ser menos exacta. El fitness multiobjetivo busca el equilibrio optimo entre ambos extremos.
