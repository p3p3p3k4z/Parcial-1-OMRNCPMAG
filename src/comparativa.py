"""
comparativa.py - Generacion de Graficas y Tabla Comparativa

Graficas que produce:
  - Curva de evolucion del fitness (mejor fitness por generacion).
  - Curvas de aprendizaje (accuracy/loss) del baseline y del modelo optimizado.
  - Grafica Accuracy (Test) vs Parametros entrenables.
  - Tabla comparativa baseline vs optimizada (accuracy, parametros, tiempo).
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

BASELINE_RESULTS_PATH = os.path.join(config.DIRECTORIO_BASE_, "resultados_base.json")
GENERATION_LOG_PATH   = os.path.join(config.DIRECTORIO_GA,    "historial_generaciones.json")
ALL_EVALS_PATH        = os.path.join(config.DIRECTORIO_GA,    "historial_individuos.json")
GA_SUMMARY_PATH       = os.path.join(config.DIRECTORIO_GA,    "resumen_ga.json")
FINAL_RESULTS_PATH    = os.path.join(config.DIRECTORIO_FINAL, "resultados_optimizado.json")

FIG_FITNESS_EVOLUTION        = os.path.join(config.DIRECTORIO_REPORTE, "fig_fitness_evolution.png")
FIG_LEARNING_CURVES_BASELINE = os.path.join(config.DIRECTORIO_REPORTE, "fig_learning_curves_baseline.png")
FIG_LEARNING_CURVES_OPTIMIZED = os.path.join(config.DIRECTORIO_REPORTE, "fig_learning_curves_optimized.png")
FIG_ACC_VS_PARAMS            = os.path.join(config.DIRECTORIO_REPORTE, "fig_accuracy_vs_params.png")
COMPARISON_TABLE_PATH        = os.path.join(config.DIRECTORIO_REPORTE, "comparison_table.json")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_fitness_evolution():
    """Grafica como evoluciona el mejor fitness en cada generacion del GA."""
    gens = _load_json(GENERATION_LOG_PATH)
    x = [g["generacion"] for g in gens]
    y = [g["mejor_fitness"] for g in gens]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", color="#1f77b4")
    plt.xlabel("Generacion")
    plt.ylabel("Mejor fitness")
    plt.title("Evolucion del fitness a traves de las generaciones (pyGAD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_FITNESS_EVOLUTION, dpi=150)
    plt.close()
    print(f"  [OK] Guardado: {FIG_FITNESS_EVOLUTION}")


def plot_learning_curves(history, title, out_path):
    """Grafica accuracy y loss de entrenamiento y validacion por epoca."""
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["accuracy"],     label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["loss"],     label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title} - Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [OK] Guardado: {out_path}")


def plot_accuracy_vs_params(baseline, final):
    """Grafica scatter de todos los individuos del GA vs los modelos finales."""
    all_evals = _load_json(ALL_EVALS_PATH)
    xs = [e["num_parametros"] for e in all_evals]
    ys = [e["val_accuracy"]   for e in all_evals]

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, alpha=0.5, label="Individuos evaluados (GA, val)", color="#7f7f7f")
    plt.scatter(
        [baseline["num_parametros"]], [baseline["accuracy_test"]],
        color="red", marker="*", s=250, label="Baseline (Test)", zorder=5,
    )
    plt.scatter(
        [final["num_parametros"]], [final["accuracy_test"]],
        color="green", marker="*", s=250, label="Optimizado (Test)", zorder=5,
    )
    plt.xscale("log")
    plt.xlabel("Parametros entrenables (escala log)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Parametros entrenables")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_ACC_VS_PARAMS, dpi=150)
    plt.close()
    print(f"  [OK] Guardado: {FIG_ACC_VS_PARAMS}")


def build_comparison_table(baseline, final):
    """Genera y guarda la tabla comparativa en JSON."""
    table = {
        "Accuracy en Test": {
            "Baseline":   round(baseline["accuracy_test"], 4),
            "Optimizada": round(final["accuracy_test"], 4),
        },
        "Parametros entrenables": {
            "Baseline":   baseline["num_parametros"],
            "Optimizada": final["num_parametros"],
        },
        "Tiempo de entrenamiento (s)": {
            "Baseline":   round(baseline["tiempo_segundos"], 1),
            "Optimizada": round(final["tiempo_segundos"], 1),
        },
    }
    with open(COMPARISON_TABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    print("\n  Tabla comparativa:")
    for metrica, valores in table.items():
        print(f"    {metrica}:")
        for modelo, valor in valores.items():
            print(f"      {modelo}: {valor}")
    return table


def generate_all():
    """Ejecuta la generacion de todas las graficas y la tabla comparativa."""
    print("\nFASE 5: Generando reporte comparativo")

    baseline = _load_json(BASELINE_RESULTS_PATH)
    final    = _load_json(FINAL_RESULTS_PATH)

    print("  Generando grafica de evolucion del fitness...")
    plot_fitness_evolution()

    print("  Generando curvas de aprendizaje del Baseline...")
    plot_learning_curves(baseline["historial"], "Baseline", FIG_LEARNING_CURVES_BASELINE)

    print("  Generando curvas de aprendizaje del modelo Optimizado...")
    plot_learning_curves(final["historial"], "Optimizado", FIG_LEARNING_CURVES_OPTIMIZED)

    print("  Generando grafica Accuracy vs Parametros...")
    plot_accuracy_vs_params(baseline, final)

    print("  Construyendo tabla comparativa...")
    table = build_comparison_table(baseline, final)

    print(f"\n  Todas las graficas guardadas en: {config.DIRECTORIO_REPORTE}")
    return table


if __name__ == "__main__":
    generate_all()
