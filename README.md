# Optimización de CNN con Algoritmos Genéticos

Proyecto del Parcial 1 de Computación Flexible: optimización de una CNN para
CIFAR-10 usando TensorFlow/Keras y pyGAD.

## Contenido

- `src/`: carga de datos, baseline, codificación genética, fitness, ejecución
  del GA, reentrenamiento y comparación.
- `results/`: métricas JSON, partición reproducible del dataset y modelos
  entrenados (`.keras`).
- `report/`: reporte final DOCX, gráficas, tabla de comparación y código para
  regenerar el reporte.
- `requirements.txt`: dependencias de Python fijadas a las versiones usadas.

El PDF proporcionado para el examen y los logs de ejecución no se incluyen en
el repositorio. El entorno virtual tampoco se versiona.

## Requisitos

- Python 3.11
- Windows, macOS o Linux
- Para el entrenamiento completo se recomienda GPU; TensorFlow nativo en
  Windows ejecuta en CPU.

## Instalación en Windows

Desde esta carpeta, en PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución

Ejecutar las cinco fases del experimento:

```powershell
python -m src.run_all
```

El proceso carga CIFAR-10, crea o reutiliza `results/split_indices.json`,
entrena el baseline, ejecuta la búsqueda genética, reentrena el mejor
individuo y genera las gráficas y la tabla comparativa. Los resultados de
cada fase se guardan en `results/` y `report/`.

El GA está configurado con población de 12, 12 generaciones y 5 épocas por
individuo. El entrenamiento puede tardar bastante en CPU. Si ya existen
resultados, el pipeline omite las fases completadas; para forzar una corrida
nueva:

```powershell
python -m src.run_all --force
```

Para regenerar solamente el reporte DOCX a partir de los resultados guardados:

```powershell
python -m report.generate_report
```
