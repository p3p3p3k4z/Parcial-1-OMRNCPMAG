# Introduccion a Algoritmos Geneticos

Esta guia explica desde cero la logica evolutiva detras del Algoritmo Genetico (GA) que usa este proyecto para optimizar la arquitectura de la red neuronal.

---

## 1. La Idea General: Evolucion Computacional

Un Algoritmo Genetico resuelve problemas de optimizacion imitando la seleccion natural de Darwin. En lugar de buscar la solucion matematicamente (lo cual puede ser imposible cuando el espacio de busqueda es enorme), simulamos la evolucion:

1. Creamos una **poblacion** de soluciones candidatas al azar.
2. Evaluamos que tan buena es cada solucion (su **aptitud** o fitness).
3. Las mejores sobreviven y se reproducen.
4. Las peores son eliminadas.
5. Las hijas heredan caracteristicas de sus padres, con pequenas variaciones (mutaciones).
6. Repetimos el proceso durante muchas **generaciones** hasta encontrar una solucion suficientemente buena.

---

## 2. Vocabulario Clave

### Individuo (Solucion Candidata)
Es una solucion propuesta al problema. En este proyecto, un individuo es una arquitectura CNN completa: cuantas capas tiene, que optimizador usa, cuantos filtros, etc.

### Cromosoma
Es la representacion codificada del individuo como un **vector numerico**. En lugar de guardar directamente "usa Adam con lr=0.001 y 2 capas", el GA trabaja con numeros continuos o discretos que luego se "traducen" a una arquitectura.

Ejemplo de cromosoma en este proyecto:
```
[0.001, 64, 0, 2, 64, 128, 0, 3, 256, 0.2]
  lr   batch opt capas fil1 fil2 fil3 k neuronas dropout
```

### Gen
Es cada posicion individual del cromosoma. El Gen 0 controla la tasa de aprendizaje, el Gen 3 controla cuantas capas convolucionales tendra la red, etc.

### Espacio de Busqueda
Es el conjunto de todos los valores validos que puede tomar cada gen. En pyGAD se define con el parametro `gene_space`. Si el espacio es demasiado grande, el GA puede tardar mucho; si es demasiado pequeno, puede no encontrar buenas soluciones.

### Fitness (Aptitud)
Es la calificacion numerica que se le da a un individuo para medir su calidad. El GA siempre intenta **maximizar** el fitness. En este proyecto, el fitness premia la exactitud en validacion y penaliza las redes con demasiados parametros.

---

## 3. El Ciclo Evolutivo

Cada iteracion del GA se llama **Generacion**. En una generacion ocurre lo siguiente:

```
Poblacion Inicial (aleatoria)
        |
        v
[1] Evaluacion de Fitness
    (Se entrena y valida cada individuo)
        |
        v
[2] Seleccion de Padres
    (Los mejores tienen mas probabilidad de reproducirse)
        |
        v
[3] Cruce (Crossover)
    (Se combinan los genes de dos padres para crear hijos)
        |
        v
[4] Mutacion
    (Algunos genes cambian aleatoriamente)
        |
        v
Nueva Generacion
    (Reemplaza a la anterior, conservando a los mejores por Elitismo)
        |
   (Repetir N veces)
        |
        v
Mejor Individuo Encontrado
```

---

## 4. Operadores Geneticos

### Seleccion de Padres (Steady-State Selection - SSS)
Es el mecanismo que elige quienes se reproducen. SSS ordena a los individuos por fitness y selecciona a los mejores directamente. Los peores son reemplazados por los hijos.

Parametro en el codigo: `GA_SELECCION_PADRES = "sss"`

### Cruce de un Punto (Single-Point Crossover)
Se selecciona un punto de corte aleatorio en el cromosoma. El hijo hereda los genes anteriores al corte del Padre A, y los genes posteriores al corte del Padre B.

```
Padre A: [0.001 | 64   | 0 | 2 | 64  | 128 | 0 | 3 | 256 | 0.2]
Padre B: [0.005 | 128  | 1 | 3 | 32  |  64 | 0 | 5 | 512 | 0.0]
                 ^
              Punto de corte (Gen 1)

Hijo:    [0.001 | 128  | 1 | 3 | 32  |  64 | 0 | 5 | 512 | 0.0]
          (A)     (B)  (B) (B) (B)   (B)  (B)(B) (B)  (B)
```

Parametro en el codigo: `GA_TIPO_CRUCE = "single_point"`

### Mutacion Aleatoria
Con cierta probabilidad (`GA_PORCENTAJE_MUTACION = 25`), un gen del hijo cambia a un valor aleatorio dentro de su espacio permitido. Esto introduce diversidad y evita que toda la poblacion converja a la misma solucion sin haber explorado suficiente el espacio.

### Elitismo
Los mejores `GA_ELITISMO = 2` individuos de cada generacion pasan directamente a la siguiente sin ser modificados. Esto garantiza que el mejor fitness encontrado nunca "retroceda".

---

## 5. El Mecanismo de Enmascaramiento

pyGAD requiere que todos los cromosomas tengan exactamente la misma longitud. Pero nuestra red puede tener 1, 2 o 3 capas convolucionales.

La solucion: **siempre generamos genes para el maximo (3 capas)**, pero si el Gen 3 dice "usa 2 capas", los genes de la capa 3 se ignoran al construir el modelo. Esos genes "duermen" en el cromosoma y pueden reactivarse si en alguna generacion futura la mutacion cambia el Gen 3 a 3.

Este mecanismo permite explorar arquitecturas de diferentes profundidades sin cambiar la longitud del vector genetico.

---

## 6. Configuracion del GA en este Proyecto

| Parametro | Valor | Que controla |
|---|---|---|
| `GA_TAMANIO_POBLACION` | 12 | Cuantas arquitecturas se evaluan por generacion |
| `GA_NUM_GENERACIONES` | 12 | Cuantos ciclos evolutivos se realizan |
| `GA_NUM_PADRES` | 6 | Cuantos individuos se seleccionan para reproducirse |
| `GA_EPOCAS_INDIVIDUO` | 5 | Epocas de entrenamiento rapido para evaluar cada individuo |
| `GA_ELITISMO` | 2 | Cuantos campeones pasan intactos a la siguiente generacion |
| `GA_PORCENTAJE_MUTACION` | 25 | Probabilidad de que un gen mute (en porcentaje) |
| `GA_TIPO_CRUCE` | single_point | Como se combinan los genes de dos padres |
| `GA_TIPO_MUTACION` | random | Como se eligen los nuevos valores al mutar |
| `GA_SELECCION_PADRES` | sss | Criterio de seleccion de padres (Steady-State) |
