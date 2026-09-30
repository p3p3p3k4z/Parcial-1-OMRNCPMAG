# Respuestas al Cuestionario de Análisis Crítico

**1. Si una CNN obtiene 92.4% de accuracy en Validation y otra obtiene 92.0%, pero la segunda utiliza únicamente el 25% de los parámetros de la primera, ¿cuál elegiría para continuar hacia la etapa final de entrenamiento y por qué?**
Elegiría definitivamente la segunda. El sacrificio de un marginal 0.4% de exactitud es un precio minúsculo a cambio de reducir la carga computacional en un asombroso 75%. La segunda red será considerablemente más ligera en memoria, más veloz en inferencia y menos propensa al sobreajuste, convirtiéndola en una opción muy superior para entornos reales.

**2. ¿Puede una solución ser considerada mejor durante la optimización aunque no tenga el mayor valor de accuracy en Validation? Explique considerando la función multiobjetivo utilizada.**
Sí, totalmente. Nuestra función *fitness* suma los puntos de exactitud pero resta puntos (penaliza) por el tamaño del modelo. Si una red obtiene un *accuracy* gigantesco pero requiere millones de parámetros, el castigo matemático hundirá su calificación general. Por el contrario, una red ligeramente menos precisa pero extremadamente diminuta obtendrá un *fitness* final mayor.

**3. ¿Por qué en este experimento el conjunto Test no participa durante la optimización?**
Porque si el Algoritmo Genético llegara a "ver" o utilizar las imágenes de Test para tomar decisiones, cometeríamos el pecado capital del *Machine Learning*: *Data Leakage* (Fuga de Datos). Las redes terminarían adaptándose específicamente para pasar el examen final, perdiendo objetividad matemática.

**4. ¿Cuál es el papel específico del conjunto Validation dentro del Algoritmo Genético?**
Sirve como el "árbitro neutral" para calcular el *fitness* de los cromosomas. Mientras el modelo ajusta sus pesos internos usando las imágenes de *Train*, el conjunto de *Validation* pone a prueba qué tan bien logra generalizar la red con datos no vistos, guiando al algoritmo genético para que seleccione a los individuos más inteligentes.

**5. ¿Qué problema aparecería si se utilizara el conjunto Test para calcular el fitness de cada individuo?**
El conjunto de *Test* perdería su pureza. Ya no serviría como métrica final sorpresa para comprobar si la red es buena en el mundo real, porque todo el Algoritmo Genético habría evolucionado y "hecho trampa" específicamente para obtener notas altas en esas fotos particulares.

**6. ¿Una red con más capas convolucionales necesariamente será superior a una red más pequeña? Justifique usando conceptos de complejidad y generalización.**
No. Una red más grande y profunda tiene mayor capacidad teórica de aprendizaje, pero si los datos no requieren de una estructura tan inmensa, la red sufrirá de "Sobreajuste" (*Overfitting*). En lugar de deducir reglas generales de las imágenes, se aprenderá las fotos de memoria. En Deep Learning, más grande no siempre significa más inteligente.

**7. Si dos arquitecturas presentan exactamente el mismo accuracy en Validation y una utiliza diez veces menos parámetros, ¿cuál debería preferirse según la lógica de este examen?**
Sin lugar a dudas la que utiliza diez veces menos parámetros. Al empatar en precisión, la red más pequeña se corona como infinitamente superior debido a su alta eficiencia y compacidad algorítmica.

**8. ¿Por qué minimizar el número de parámetros puede ser importante en aplicaciones reales?**
En la vida real, los modelos suelen instalarse en teléfonos móviles, dispositivos IoT o servidores donde la memoria RAM y el poder de procesamiento cuestan dinero o tienen baterías limitadas. Minimizar los parámetros significa lanzar aplicaciones que gastan menos batería, requieren menos internet y responden de manera casi instantánea.

**9. Si durante la optimización aparece una arquitectura con accuracy ligeramente menor pero con una reducción considerable en parámetros, ¿cómo debería interpretarse ese resultado?**
Debe interpretarse como un rotundo éxito del "Trade-off" (compromiso). Esta es la esencia pura de la optimización multiobjetivo: hallar arquitecturas que están dispuestas a intercambiar décimas porcentuales irrelevantes de exactitud a cambio de ganancias colosales en eficiencia técnica.

**10. ¿Por qué utilizar accuracy de entrenamiento dentro de la función fitness sería una mala práctica?**
El *accuracy* de entrenamiento solo indica qué tan bien la red memorizó los ejemplos que estudió. Basar el *fitness* en eso crearía generaciones de redes gigantescas y sobreajustadas, las cuales al enfrentarse a imágenes nuevas fallarían desastrosamente al no saber generalizar.

**11. ¿Es posible que la arquitectura seleccionada como mejor individuo después de 5 épocas ya no sea la mejor después de entrenarla durante 30 épocas? Explique.**
Sí, es muy posible. Cinco épocas es solo un "vistazo rápido" al potencial de aprendizaje de una red. Algunas redes utilizan algoritmos más precavidos y arrancan con pésimos números en la época 5, pero terminan encontrando patrones magistrales hacia la época 30. El ranking a 5 épocas es una estimación aproximada, no una garantía absoluta.

**12. ¿Por qué el protocolo exige entrenar cada individuo durante solamente 5 épocas, pero reentrenar el mejor modelo durante 30 épocas?**
Por piedad computacional. Nuestro genético evalúa 144 redes distintas. Entrenar las 144 durante 30 épocas tomaría días de procesamiento continuo en la computadora. Darles 5 épocas a cada una ahorra muchísimo tiempo y permite sacar una tabla de líderes confiable para luego sí premiar al campeón con un entrenamiento largo de 30 épocas.

**13. Suponga que la arquitectura ganadora obtiene una mejora de 0.5% en accuracy respecto al baseline, pero duplica el número de parámetros. ¿Considera que realmente representa una mejora? Justifique.**
Bajo la lupa de este proyecto, de ninguna manera representa una mejora. Cambiar un nimio aumento de 0.5% a cambio de castigar la memoria RAM al doble es un suicidio de ingeniería de software. La relación costo-beneficio de esa red es desastrosa.

**14. Si la arquitectura optimizada tiene menor accuracy que el baseline, pero utiliza menos de la mitad de los parámetros, ¿puede considerarse exitosa la optimización? Explique.**
Totalmente exitosa. El objetivo no era únicamente buscar "el 100% de exactitud", era encontrar redes eficientes. Si sacrificamos una fracción tolerable de efectividad para ganar más del 50% de memoria RAM y velocidad, habremos creado una arquitectura sumamente atractiva para despliegues móviles.

**15. Si se duplica el tamaño de la población del Algoritmo Genético, ¿está garantizado encontrar una mejor solución? ¿Por qué?**
No está matemáticamente garantizado. Al duplicar la población agregamos mucha más variedad genética (lo que aumenta nuestras probabilidades matemáticas), pero como la selección y mutación operan bajo métodos estocásticos (aleatorios), no hay garantías de que esa población logre dar en el blanco perfecto.

**16. ¿Por qué la mutación puede generar individuos aparentemente peores y aun así ser necesaria para el éxito del algoritmo?**
Porque a veces hay que dar un paso hacia atrás para dar dos hacia adelante. La mutación, aunque rompa redes exitosas a corto plazo, introduce diversidad biológica forzada. Estos genes "malos" ayudan a las redes a escapar de trampas locales y pueden combinarse en futuras generaciones para construir súper-campeones.

**17. Suponga que el individuo con mayor fitness utiliza una combinación de hiperparámetros muy distinta a la que usted hubiera elegido manualmente. ¿Qué indica esto acerca del espacio de búsqueda y del proceso de optimización?**
Demuestra que los humanos solemos diseñar en base a sesgos o "fórmulas típicas". El espacio hiperdimensional de las CNN es demasiado vasto y oscuro para nosotros. El genético nos prueba que, más allá de la intuición humana de *"más filtros siempre es mejor"*, hay atajos contraintuitivos que rinden de maravilla.

**18. Análisis integrador final sobre el Baseline y la Red Optimizada:**
Con base en los resultados estrictamente obtenidos durante el desarrollo de los experimentos, este análisis comprueba visual y numéricamente por qué delegar la decisión a un Algoritmo Genético ha resultado superior a las asunciones manuales:

* **Compromiso (Trade-off) Exactitud vs Complejidad:** El experimento demostró que es falso el mito de "más grande es mejor". La red base, con sus enormes 315 mil parámetros, sufrió sobreajuste y alcanzó solo 67.61% de exactitud con un *Loss* preocupante de 2.41. La red genéticamente evolucionada optó por la compacidad (115 mil parámetros) logrando un 69.74% de *Accuracy* y mitigando la pérdida casi a la mitad (1.48).
* **Ventajas y Desventajas:** El *Baseline* tuvo como principal desventaja su propensión a la memorización (*Overfitting*) debido al exceso de capacidad injustificada. Nuestra red optimizada (*115,018 parámetros*) requirió más cálculos por época debido a la capa extra y a un lote más pequeño, sumando segundos de entrenamiento; pero a nivel final, sus ventajas en ligereza son indiscutibles.
* **¿Mejora real o solución alternativa?:** Hubo una mejora contundente y de dominancia estricta. No estamos ante un empate táctico. Al haber superado a la red de referencia en exactitud (+2.13%) y habiendo reducido su núcleo de parámetros en un aplastante -63.6%, se trata de una superación científica real.
* **Justificación de uso:** En una aplicación comercial o del mundo real optaría incondicionalmente por la arquitectura Optimizada. Al ser un 63.6% más pequeña, permite ser albergada en la memoria RAM de dispositivos con bajos recursos o embebidos en hardware real (Edge Computing), emitiendo inferencias de alta exactitud con un mínimo consumo energético, un escenario inviable para arquitecturas infladas.
