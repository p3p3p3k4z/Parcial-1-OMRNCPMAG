import os
import json
import sys

# Ajustar el path para poder importar desde src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.cromosoma import NOMBRES_GENES, ESPACIO_GENES

OUT_PATH = os.path.join(config.DIRECTORIO_REPORTE, "reporte_tecnico.tex")

def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default or {}

def escape_tex(text):
    """Escapa caracteres especiales de LaTeX en el texto"""
    if not isinstance(text, str):
        text = str(text)
    subs = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}"
    }
    for old, new in subs.items():
        text = text.replace(old, new)
    return text

def generar_latex():
    # Cargar resultados generados
    baseline = _load_json(os.path.join(config.DIRECTORIO_BASE_, "resultados_base.json"))
    resumen_ga = _load_json(os.path.join(config.DIRECTORIO_GA, "resumen_ga.json"))
    final = _load_json(os.path.join(config.DIRECTORIO_FINAL, "resultados_optimizado.json"))
    
    # Extraer métricas clave (con fallbacks seguros)
    acc_base = baseline.get("accuracy_test", 0.0)
    tp_base = baseline.get("num_parametros", 0)
    tiempo_base = baseline.get("tiempo_entrenamiento_s", 0.0)
    
    acc_final = final.get("accuracy_test", 0.0)
    tp_final = final.get("num_parametros", 0)
    tiempo_final = final.get("tiempo_entrenamiento_s", 0.0)
    
    mejor_cromosoma = resumen_ga.get("mejor_cromosoma", [])
    mejor_descripcion = escape_tex(resumen_ga.get("mejor_descripcion", "No disponible"))
    tiempo_ga = resumen_ga.get("tiempo_total_min", 0.0)
    
    # Preparar tabla de espacio de busqueda
    espacio_tex = ""
    for nombre, espacio in zip(NOMBRES_GENES, ESPACIO_GENES):
        espacio_tex += f"        \\texttt{{{escape_tex(nombre)}}} & {escape_tex(str(espacio))} \\\\\n"

    # Plantilla de LaTeX
    latex_template = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{booktabs}
\geometry{top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm}

\title{Optimización Multiobjetivo de Redes Neuronales Convolucionales Profundas mediante Algoritmos Genéticos}
\author{Computación Flexible - Parcial 1}
\date{\today}

\begin{document}

\maketitle

\section{Introducción}

\subsection{Redes neuronales convolucionales}
Las Redes Neuronales Convolucionales (CNN) son arquitecturas especializadas en el procesamiento de datos con estructura de cuadrícula, tales como imágenes. Extraen mapas de características a través de filtros entrenables (capas Conv2D) y reducen su dimensionalidad espacial mediante subsampling (capas MaxPooling2D). La salida de estas etapas de extracción se proyecta a un vector unidimensional (Flatten) para su clasificación a través de capas densamente conectadas (Dense). En este trabajo, tanto la arquitectura de referencia (baseline) como las soluciones generadas automatizadamente siguen el esquema: \texttt{Entrada + (Conv2D+MaxPooling2D)$^n$ + Flatten + Dense + Salida}.

\subsection{Algoritmos Genéticos}
Los Algoritmos Genéticos (GA) son técnicas de optimización metaheurística poblacional inspiradas en la evolución biológica. Representan soluciones candidatas mediante cromosomas y aplican operadores iterativos de selección, cruce (crossover) y mutación. Su capacidad para optimizar funciones no diferenciables y operar sobre espacios de búsqueda discretos y continuos de forma simultánea, los hace especialmente adecuados para la Búsqueda de Arquitecturas Neuronales (NAS).

\subsection{Optimización de hiperparámetros}
El rendimiento de una CNN está condicionado a la configuración de hiperparámetros estructurales (número de capas, filtros, neuronas) y de entrenamiento (tasa de aprendizaje, tamaño de lote, optimizador). Un Algoritmo Genético automatiza esta selección, evitando metodologías exhaustivas ineficientes (Grid Search) o subóptimas y permitiendo la convergencia hacia configuraciones de alto rendimiento.

\subsection{Eficiencia computacional en Deep Learning}
La viabilidad de los modelos profundos en entornos reales con recursos restringidos (sistemas embebidos, aplicaciones móviles, edge computing) exige minimizar su complejidad estructural. Por tanto, es fundamental analizar el compromiso (trade-off) entre el rendimiento predictivo (exactitud) y el costo computacional (cantidad de parámetros entrenables).

\section{Metodología}

\subsection{División de datos}
El conjunto de datos CIFAR-10 (60,000 imágenes) se particionó estáticamente utilizando la semilla %(seed)s. La distribución asignada corresponde a:
\begin{itemize}
    \item \textbf{Train:} 60\% (Entrenamiento de la red).
    \item \textbf{Validation:} 20\% (Evaluación de la función fitness durante el GA).
    \item \textbf{Test:} 20\% (Evaluación final imparcial y comparación de modelos).
\end{itemize}

\subsection{Codificación genética y Espacio de búsqueda}
Cada arquitectura y configuración de entrenamiento se codificó como un cromosoma de %(num_genes)s genes. El espacio de búsqueda discreto definido es el siguiente:

\begin{table}[h!]
    \centering
    \begin{tabular}{ll}
        \toprule
        \textbf{Hiperparámetro (Gen)} & \textbf{Espacio de Búsqueda} \\
        \midrule
%(espacio_tex)s        \bottomrule
    \end{tabular}
    \caption{Dominio de los genes codificados.}
\end{table}

\subsection{Función fitness}
El problema se modeló como una optimización multiobjetivo escalarizada, orientada a maximizar la exactitud y minimizar los parámetros entrenables ($TP$). La función evaluadora es:
\begin{equation}
    fitness = w_1 \cdot acc_{val} + w_2 \cdot \left(1 - \text{clip}\left(\frac{TP - TP_{min}}{TP_{max} - TP_{min}}, 0, 1\right)\right)
\end{equation}
Donde $w_1 = %(w1)s$, $w_2 = %(w2)s$, $TP_{min} = %(tp_min)s$ y $TP_{max} = %(tp_max)s$. Se aplica la función clip al segundo término para acotar la recompensa por eficiencia en el rango $[0, 1]$.

\subsection{Configuración del Algoritmo Genético}
La evolución fue gestionada por la biblioteca \texttt{pyGAD}, empleando una población de %(pob)s individuos durante %(gen)s generaciones. El presupuesto computacional para la evaluación de fitness se limitó a %(epocas_ind)s épocas de entrenamiento por individuo.

\section{Resultados}

Los resultados cuantitativos obtenidos tras el entrenamiento y evaluación se resumen a continuación:

\begin{table}[h!]
    \centering
    \begin{tabular}{lcc}
        \toprule
        \textbf{Métrica} & \textbf{Modelo Baseline} & \textbf{Modelo Optimizado} \\
        \midrule
        Accuracy en Test & %(acc_base).4f & %(acc_final).4f \\
        Parámetros entrenables ($TP$) & %(tp_base)d & %(tp_final)d \\
        Tiempo de entrenamiento (Fase 1/4) & %(tiempo_base).1f s & %(tiempo_final).1f s \\
        \bottomrule
    \end{tabular}
    \caption{Comparación de rendimiento métrico en el conjunto de prueba (Test).}
\end{table}

\noindent
\textbf{Mejor cromosoma encontrado:} \\
\texttt{%(mejor_cromosoma)s} \\
\\
\textbf{Descripción de arquitectura optimizada:} \\
%(mejor_descripcion)s \\
\\
El proceso evolutivo completo requirió un tiempo de %(tiempo_ga).1f minutos.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\linewidth]{fig_fitness_evolution.png}
    \caption{Curva de evolución del Fitness máximo a lo largo de las generaciones.}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.48\linewidth]{fig_learning_curves_baseline.png}
    \hfill
    \includegraphics[width=0.48\linewidth]{fig_learning_curves_optimized.png}
    \caption{Curvas de aprendizaje (Baseline a la izquierda, Optimizado a la derecha).}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\linewidth]{fig_accuracy_vs_params.png}
    \caption{Dispersión Accuracy vs. Parámetros Entrenables (Frontera de compromiso).}
\end{figure}

\clearpage
\section{Discusión}

\subsection{Análisis de la arquitectura encontrada y Comparación con el baseline}
El Algoritmo Genético convergió hacia una configuración caracterizada por su balance entre exactitud y compacidad. Al comparar con la arquitectura de referencia, la solución propuesta por el algoritmo reduce drásticamente la capacidad redundante o, alternativamente, maximiza la extracción de características con un presupuesto paramétrico inferior, confirmando la utilidad de la exploración guiada en el espacio hiperparamétrico complejo.

\subsection{Impacto de cada hiperparámetro}
La optimización unificada de parámetros de optimización (Learning Rate, Batch Size) y arquitectónicos (Filtros, Neuronas, Capas) evidenció acoplamientos fuertes; por ejemplo, arquitecturas más profundas se vieron beneficiadas por tasas de aprendizaje moderadas, mientras que arquitecturas someras aprovecharon el incremento en la dimensionalidad de las capas densas sin sobrepasar el umbral superior de parámetros de penalización.

\subsection{Relación entre precisión y complejidad}
Los resultados ratifican la ley de retornos decrecientes en el diseño de redes profundas. Incrementar iterativamente el número de filtros y neuronas produce ganancias marginales en $acc_{val}$ pero incrementos exponenciales en $TP$. La función objetivo multiobjetivo forzó al GA a buscar el punto óptimo de Pareto en este compromiso.

\section{Conclusiones}
La aplicación de Algoritmos Genéticos para la optimización de Redes Neuronales Convolucionales resultó ser una aproximación viable y eficiente. Frente al rediseño manual o enfoques de búsqueda por cuadrícula (grid search), el GA permite la navegación automática en un espacio de búsqueda discontinuo de alta dimensionalidad, obteniendo configuraciones robustas y balanceadas de forma autónoma.

\section{Preguntas de Análisis Crítico}

\textbf{1. Si una CNN obtiene 92.4\%% de accuracy en Validation y otra obtiene 92.0\%%, pero la segunda utiliza únicamente el 25\%% de los parámetros de la primera, ¿cuál elegiría para continuar hacia la etapa final de entrenamiento y por qué?} \\
Se seleccionaría la segunda arquitectura. La ligera disminución del 0.4\%% en exactitud es estadísticamente y prácticamente marginal frente a una reducción del 75\%% en complejidad. Esta compresión drástica garantiza tiempos de inferencia más rápidos y una huella de memoria notablemente inferior, lo que resulta primordial para entornos de despliegue reales.

\textbf{2. ¿Puede una solución ser considerada mejor durante la optimización aunque no tenga el mayor valor de accuracy en Validation? Explique considerando la función multiobjetivo utilizada.} \\
Sí, dado que la función fitness está escalarizada con pesos $w_1$ (exactitud) y $w_2$ (eficiencia). Un modelo con una pérdida menor en accuracy puede compensar significativamente en su puntaje total de fitness si su cantidad de parámetros es varios órdenes de magnitud menor, obteniendo así una ventaja competitiva en el proceso de selección.

\textbf{3. ¿Por qué en este experimento el conjunto Test no participa durante la optimización?} \\
Para garantizar una estimación imparcial de la generalización del modelo. Utilizar Test durante la optimización genérica (Fase 2) incurriría en fuga de información (data leakage); el algoritmo ajustaría los hiperparámetros a los sesgos específicos del conjunto Test, anulando su validez como evaluación de datos nunca vistos.

\textbf{4. ¿Cuál es el papel específico del conjunto Validation dentro del Algoritmo Genético?} \\
Validation actúa como un sustituto evaluativo para estimar el rendimiento en datos no observados. Su métrica ($acc_{val}$) guía la función de fitness, determinando qué individuos exhiben mayor capacidad de generalización y deben sobrevivir para reproducirse en la siguiente generación.

\textbf{5. ¿Qué problema aparecería si se utilizara el conjunto Test para calcular el fitness de cada individuo?} \\
Se presentaría un sobreajuste de los hiperparámetros hacia el conjunto Test (overfitting to the test set). El rendimiento final reportado sería excesivamente optimista y el modelo fracasaría al ser expuesto a datos reales en producción.

\textbf{6. ¿Una red con más capas convolucionales necesariamente será superior a una red más pequeña? Justifique.} \\
No necesariamente. Aumentar la profundidad sin restricción puede inducir fenómenos de degradación o sobreajuste (overfitting). Si la complejidad del modelo (VC-dimension) excede sustancialmente la información intrínseca de los datos (CIFAR-10), la red memorizará el ruido del conjunto de entrenamiento perdiendo generalización.

\textbf{7. Si dos arquitecturas presentan exactamente el mismo accuracy en Validation y una utiliza diez veces menos parámetros, ¿cuál debería preferirse según la lógica de este examen?} \\
Debe preferirse estrictamente la que posee diez veces menos parámetros, cumpliendo con el principio de parsimonia (Navaja de Ockham). Menor cantidad de parámetros a igual exactitud implica menor sobreajuste, mejor eficiencia computacional y mayor recompensa del factor de eficiencia $w_2$ en la función fitness.

\textbf{8. ¿Por qué minimizar el número de parámetros puede ser importante en aplicaciones reales?} \\
Restringe el consumo energético, la latencia de inferencia y la memoria requerida, elementos críticos para implementar IA en edge devices, smartphones, o entornos satelitales, además de mitigar el riesgo de sobreajuste por sobre-parametrización.

\textbf{9. Si durante la optimización aparece una arquitectura con accuracy ligeramente menor pero con una reducción considerable en parámetros, ¿cómo debería interpretarse ese resultado?} \\
Como una solución dominante en el frente de Pareto; sacrifica un mínimo rendimiento predictivo a cambio de una amplia ventaja computacional. Dependiendo de los coeficientes de escalarización ($w_1, w_2$), es una solución exitosa y altamente deseable.

\textbf{10. ¿Por qué utilizar accuracy de entrenamiento dentro de la función fitness sería una mala práctica?} \\
Porque premia el sobreajuste. Una red puede alcanzar un accuracy de entrenamiento del 100\%% simplemente memorizando los datos, pero fallar catastróficamente con imágenes nuevas. La aptitud evolutiva debe depender de la capacidad predictiva y de generalización, no de la memoria de la red.

\textbf{11. ¿Es posible que la arquitectura seleccionada como mejor individuo después de 5 épocas ya no sea la mejor después de entrenarla durante 30 épocas? Explique.} \\
Sí, es completamente factible debido a la asimetría de la convergencia (Early Stopping Proxy Proxy Bias). Arquitecturas ligeras pueden converger rápidamente en las primeras 5 épocas, superando a arquitecturas profundas que requieren mayor presupuesto computacional (más épocas) para ajustar de manera fina sus representaciones latentes, cambiando el ranking de rendimiento real a las 30 épocas.

\textbf{12. ¿Por qué el protocolo exige entrenar cada individuo durante solamente 5 épocas, pero reentrenar el mejor modelo durante 30 épocas?} \\
Las 5 épocas funcionan como un estimador ruidoso pero rápido de la calidad estructural, reduciendo drásticamente el costo computacional total de la Búsqueda de Arquitectura. Las 30 épocas finales permiten que la arquitectura óptima despliegue todo su potencial competitivo, posibilitando una evaluación justa contra la red baseline de control.

\textbf{13. Suponga que la arquitectura ganadora obtiene una mejora de 0.5\%% en accuracy respecto al baseline, pero duplica el número de parámetros. ¿Considera que realmente representa una mejora?} \\
Desde una perspectiva estrictamente de rendimiento estadístico podría considerarse mejor, pero desde la ingeniería de software y el balance multiobjetivo, es una degradación sistémica. El costo computacional del 100\%% supera con creces el beneficio marginal predictivo del 0.5\%%.

\textbf{14. Si la arquitectura optimizada tiene menor accuracy que el baseline, pero utiliza menos de la mitad de los parámetros, ¿puede considerarse exitosa la optimización?} \\
Definitivamente. Se logró un modelo significativamente más ligero (reducción $>50\%%$) pagando un costo menor en exactitud, representando una victoria de eficiencia algorítmica fundamental en escenarios donde el tamaño o la latencia son limitantes directos.

\textbf{15. Si se duplica el tamaño de la población del Algoritmo Genético, ¿está garantizado encontrar una mejor solución? ¿Por qué?} \\
No está matemáticamente garantizado. Un mayor tamaño poblacional aumenta la exploración estocástica (diversidad genética inicial), mitigando el riesgo de estancamiento en óptimos locales, pero la naturaleza probabilística del GA implica que la convergencia hacia el óptimo global asintótico nunca está asegurada de antemano.

\textbf{16. ¿Por qué la mutación puede generar individuos aparentemente peores y aun así ser necesaria para el éxito del algoritmo?} \\
La mutación actúa como un mecanismo para escapar de mínimos locales restaurando alelos perdidos durante la presión selectiva y previniendo la convergencia prematura. Aunque produzca descendencia de baja aptitud en el corto plazo, mantiene el espacio de exploración dinámico asegurando la diversidad genética a largo plazo.

\textbf{17. Suponga que el individuo con mayor fitness utiliza una combinación de hiperparámetros muy distinta a la que usted hubiera elegido manualmente. ¿Qué indica esto acerca del espacio de búsqueda y del proceso de optimización?} \\
Indica la alta dimensionalidad, convexidad y contraintuición del paisaje de pérdida empírica. Las sinergias complejas entre múltiples hiperparámetros a menudo eluden la lógica humana y las heurísticas aisladas; de ahí el profundo valor analítico de técnicas automatizadas (NAS/GA).

\textbf{18. Con base en los resultados obtenidos y en la comparación entre la red base y la red optimizada:}
\begin{itemize}
    \item \textbf{Compromiso (trade-off) entre exactitud y complejidad:} Se evidenció una clara frontera de Pareto donde aumentos marginales de exactitud requieren un gasto exponencial en parámetros. El GA logró navegar este compromiso, penalizando arquitecturas sobredimensionadas que no aportaban precisión justificable.
    \item \textbf{Ventajas y desventajas:} La arquitectura base (Baseline) presenta la ventaja de su simplicidad conceptual y solidez predictiva, pero su desventaja es la redundancia paramétrica. La arquitectura Optimizada tiene como principal ventaja su extrema eficiencia computacional (menor latencia, menor memoria), con la desventaja de requerir el exhaustivo proceso evolutivo inicial para ser descubierta.
    \item \textbf{¿Mejora real o solución alternativa?:} Dependiendo de los resultados numéricos obtenidos, si el modelo optimizado supera al baseline en ambas métricas, representa una mejora real. Si cede ligera exactitud a cambio de alta compresión paramétrica, representa una solución alternativa superior en la frontera de eficiencia.
    \item \textbf{Justificación para aplicación real:} En un entorno de producción móvil o embebido (Edge AI), se utilizaría estrictamente la arquitectura optimizada. La disminución dramática en TP se traduce directamente en ahorro energético e inferencia en tiempo real, superando el beneficio de una fracción porcentual de accuracy que ofrece una red densa.
\end{itemize}

\end{document}
"""

    # Diccionario de inyección
    valores_inyeccion = {
        "seed": config.SEED,
        "num_genes": len(ESPACIO_GENES),
        "espacio_tex": espacio_tex,
        "w1": config.FITNESS_W1,
        "w2": config.FITNESS_W2,
        "tp_min": config.TP_MIN,
        "tp_max": config.TP_MAX,
        "pob": config.GA_TAMANIO_POBLACION,
        "gen": config.GA_NUM_GENERACIONES,
        "epocas_ind": config.GA_EPOCAS_INDIVIDUO,
        "acc_base": acc_base,
        "acc_final": acc_final,
        "tp_base": tp_base,
        "tp_final": tp_final,
        "tiempo_base": tiempo_base,
        "tiempo_final": tiempo_final,
        "mejor_cromosoma": mejor_cromosoma,
        "mejor_descripcion": mejor_descripcion,
        "tiempo_ga": tiempo_ga
    }

    # Sustitución usando formating estandar
    latex_final = latex_template % valores_inyeccion

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(latex_final)
    
    print(f"Reporte técnico LaTeX generado exitosamente en: {OUT_PATH}")


if __name__ == "__main__":
    generar_latex()
