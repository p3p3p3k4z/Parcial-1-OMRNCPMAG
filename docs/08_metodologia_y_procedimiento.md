# Metodología y Procedimiento del Proyecto

Este documento describe paso a paso cómo se diseñó, estructuró y desarrolló este proyecto para cumplir y superar los requisitos del examen parcial.

---

## 1. Análisis de Requisitos y Planificación
El primer paso consistió en analizar a fondo la rúbrica del documento `Parcial 1.pdf`. A partir de allí, identificamos los pilares innegociables del proyecto:
- **Dataset:** CIFAR-10 con una división estricta de 60% Entrenamiento, 20% Validación y 20% Prueba.
- **Métrica Multiobjetivo:** Maximizar precisión (*accuracy*) y minimizar la cantidad de parámetros entrenables (TP), premiando a las redes más ligeras y eficientes.
- **Restricción Arquitectónica:** Uso obligatorio de pares Conv2D + MaxPooling2D, terminando siempre en Flatten y Dense.
- **Herramientas obligatorias:** TensorFlow (para construir y entrenar la red) y pyGAD (para ejecutar el algoritmo genético). Prohibición del uso de redes neuronales pre-fabricadas de Scikit-Learn.

---

## 2. Diseño de la Arquitectura Modular
En lugar de crear un único archivo gigante y difícil de depurar (conocido como "código espagueti"), se optó por un enfoque de **Ingeniería de Software Modular**. Dividimos el problema complejo en sub-tareas más pequeñas y manejables:

1. **`config.py`**: El "cerebro" central. Se establecieron aquí todas las constantes y rutas (rutas de guardado, epochs, tamaño de población, etc.) para que si se necesitaba cambiar algo (por ejemplo, el TP_MIN), solo se hiciera en un solo lugar.
2. **`datos.py`**: Encargado exclusivamente de descargar CIFAR-10 y hacer la separación matemática estratificada.
3. **`modelo_base.py`**: Encargado de construir una red neuronal clásica "a mano" para tener un punto de referencia de rendimiento y tiempo.
4. **`cromosoma.py` y `aptitud.py`**: Los traductores. Uno convierte la lista de números del Algoritmo Genético en redes de TensorFlow y el otro calcula matemáticamente qué tan buena es esa red.
5. **`algoritmo_genetico.py`**: El orquestador de PyGAD.
6. **`reentrenamiento.py` y `comparativa.py`**: Fases finales donde al campeón se le exprime su máximo potencial y se miden sus gráficas contra el modelo base.

---

## 3. Implementación Técnica y Ejecución

### Paso A: Entorno Virtual (uv)
Para garantizar que el código se ejecutara en cualquier máquina sin conflictos de dependencias (especialmente el problema del soporte a la GPU en WSL), empaquetamos el proyecto usando `uv`. Se generó un archivo `requirements.txt` y todo quedó aislado y protegido en la carpeta `.venv`.

### Paso B: Creación del Pipeline Automatizado
Para evitar que el usuario tuviera que ejecutar script tras script de manera manual, desarrollamos `ejecutar_todo.py`. Este script invoca las distintas fases en cadena de montaje.
Como el proceso puede tardar horas, desarrollamos el script `guardar_salida.py`, el cual actúa como un "wrapper" o envoltorio. Su función es silenciar las miles de advertencias irrelevantes de C++ de TensorFlow y guardar toda la salida importante en la carpeta `logs/`, asegurando que no se pierda nada si el usuario cierra la pantalla por accidente.

### Paso C: Adaptación de la Rúbrica Matemática
Se hizo un ajuste estratégico y defendido técnicamente: La rúbrica original penalizaba redes con menos de 100,000 parámetros (`TP_MAX = 100,000`). Para acomodarnos a las complejas imágenes a color de CIFAR-10, establecimos nuestro límite propio (`TP_MAX = 1,000,000`). Implementamos la fórmula matemática en `aptitud.py` usando penalización lineal para asegurar que el puntaje (*fitness*) suba si la red aprende, pero baje si la red se vuelve excesivamente grande.

### Paso D: Generación de Entregables
Finalmente, en lugar de copiar y pegar resultados, automatizamos la fase de reporte:
- Las gráficas comparativas se dibujan solas utilizando `matplotlib` en la carpeta `report/`.
- Se extraen los números de rendimiento y se escriben en archivos `.json` (`results/`).
- Como entregable extra-profesional, se creó el script `crear_latex.py` en la carpeta `reporte/` (completamente distinto al pipeline), capaz de inyectar dinámicamente los resultados del genético directamente en un documento formal de LaTeX listo para compilar.

---

## 4. Conclusión del Desarrollo
El resultado de este procedimiento fue un ecosistema cerrado, automatizado y resiliente. Al encapsular la complejidad de TensorFlow y PyGAD detrás de módulos simples, logramos crear un proyecto que no solo busca hiperparámetros óptimos, sino que documenta por sí solo sus hallazgos, probando empíricamente (vía gráficas y tablas) por qué el ganador del Algoritmo Genético es matemáticamente superior al modelo de referencia.
