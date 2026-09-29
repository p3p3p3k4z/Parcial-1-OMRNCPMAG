"""
guardar_salida.py - Wrapper para Registrar la Salida de Consola

Proposito:
Ejecutar el pipeline completo y guardar TODO lo que aparece en consola
(incluyendo las barras de progreso de TensorFlow, mensajes de debug,
resultados de cada generacion del GA, etc.) en un archivo .txt dentro
de la carpeta logs/.

Este script NO modifica ningun archivo dentro de src/. Funciona como
una capa adicional que intercepta la salida estandar y la duplica
hacia un archivo de texto plano.

Uso:
    python guardar_salida.py                    # Ejecuta el pipeline y guarda log
    python guardar_salida.py --force             # Re-ejecuta todo desde cero
    python guardar_salida.py --poblacion 15      # Pasa argumentos al pipeline
    python guardar_salida.py --solo-log mi_log   # Nombre personalizado para el archivo

El archivo de salida se guarda en:
    logs/pipeline_YYYYMMDD_HHMMSS.txt    (con fecha y hora)
    logs/ultima_ejecucion.txt             (siempre apunta a la ejecucion mas reciente)
"""
import os
import sys
import re
import time


# -- Configuracion de rutas -------------------------------------------------

DIRECTORIO_PROYECTO = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_LOGS     = os.path.join(DIRECTORIO_PROYECTO, "logs")
os.makedirs(DIRECTORIO_LOGS, exist_ok=True)

# Patron para limpiar codigos de color/formato ANSI de la terminal
PATRON_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


# -- Clase Duplicadora -------------------------------------------------------

class DuplicadorConsola:
    """
    Intercepta lo que se escribe en la terminal (stdout o stderr) y lo
    envia simultaneamente a la consola original y a uno o mas archivos .txt.

    No altera el comportamiento normal del programa; solo agrega la copia
    en archivo como efecto adicional.
    """
    def __init__(self, stream_original, archivos_destino):
        self.stream_original = stream_original
        self.archivos_destino = archivos_destino

    def write(self, texto):
        self.stream_original.write(texto)
        self.stream_original.flush()

        texto_limpio = PATRON_ANSI.sub("", texto)
        for archivo in self.archivos_destino:
            try:
                archivo.write(texto_limpio)
                archivo.flush()
            except Exception:
                pass

    def flush(self):
        self.stream_original.flush()
        for archivo in self.archivos_destino:
            try:
                archivo.flush()
            except Exception:
                pass

    def isatty(self):
        return getattr(self.stream_original, "isatty", lambda: False)()

    def fileno(self):
        return self.stream_original.fileno()

    @property
    def encoding(self):
        return getattr(self.stream_original, "encoding", "utf-8")


# -- Funciones principales ---------------------------------------------------

def _extraer_argumento_log(argumentos):
    """
    Busca y remueve --solo-log <nombre> de la lista de argumentos.
    Retorna (nombre_personalizado_o_None, argumentos_restantes).
    """
    nombre = None
    restantes = []
    i = 0
    while i < len(argumentos):
        if argumentos[i] == "--solo-log" and i + 1 < len(argumentos):
            nombre = argumentos[i + 1]
            i += 2
        else:
            restantes.append(argumentos[i])
            i += 1
    return nombre, restantes


def main():
    nombre_log, args_pipeline = _extraer_argumento_log(sys.argv[1:])

    marca_tiempo = time.strftime("%Y%m%d_%H%M%S")

    if nombre_log:
        nombre_archivo = f"{nombre_log}_{marca_tiempo}.txt"
    else:
        nombre_archivo = f"pipeline_{marca_tiempo}.txt"

    ruta_log         = os.path.join(DIRECTORIO_LOGS, nombre_archivo)
    ruta_ultima      = os.path.join(DIRECTORIO_LOGS, "ultima_ejecucion.txt")

    archivos = []
    try:
        archivo_log    = open(ruta_log, "w", encoding="utf-8")
        archivo_ultima = open(ruta_ultima, "w", encoding="utf-8")
        archivos = [archivo_log, archivo_ultima]

        sys.stdout = DuplicadorConsola(sys.__stdout__, archivos)
        sys.stderr = DuplicadorConsola(sys.__stderr__, archivos)

        print("=" * 65)
        print(f" REGISTRO DE SALIDA INICIADO")
        print(f" Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Archivo: {ruta_log}")
        print(f" Copia reciente: {ruta_ultima}")
        print("=" * 65)

        # Re-armar sys.argv para que ejecutar_todo.main() reciba los argumentos
        sys.argv = ["guardar_salida.py"] + args_pipeline

        from src.ejecutar_todo import main as ejecutar_pipeline
        ejecutar_pipeline()

        print("\n" + "=" * 65)
        print(f" REGISTRO DE SALIDA FINALIZADO")
        print(f" Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Log guardado en: {ruta_log}")
        print("=" * 65)

    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        for f in archivos:
            try:
                f.close()
            except Exception:
                pass

    print(f"\nLog guardado en: {ruta_log}")
    print(f"Copia rapida en: {ruta_ultima}")


if __name__ == "__main__":
    main()
