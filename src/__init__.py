"""
Paquete principal del proyecto: Optimizacion de CNN con Algoritmos Geneticos.

Modulos disponibles:
    config              → Constantes globales y configuracion
    datos               → Carga y particion de CIFAR-10
    cromosoma           → Codificacion genetica y construccion de CNN
    aptitud             → Funcion de fitness multiobjetivo
    modelo_base         → Red neuronal de referencia (baseline)
    algoritmo_genetico  → Ejecucion del GA con pyGAD
    reentrenamiento     → Reentrenamiento del mejor individuo
    ejecutar_todo       → Orquestador del pipeline completo
"""
