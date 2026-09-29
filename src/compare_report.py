"""
Genera las gráficas y la tabla comparativa exigidas por el enunciado:
  - Curva de evolución del fitness (mejor fitness por generación).
  - Curvas de aprendizaje (accuracy/loss) del baseline y del modelo optimizado.
  - Gráfica Accuracy (Test) vs Parámetros entrenables.
  - Tabla comparativa baseline vs optimizada (accuracy, parámetros, tiempo).
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

BASELINE_RESULTS_PATH = os.path.join(config.BASELINE_DIR, "baseline_results.json")
GENERATION_LOG_PATH = os.path.join(config.GA_DIR, "generation_best_fitness.json")
ALL_EVALS_PATH = os.path.join(config.GA_DIR, "all_individuals_log.json")
GA_SUMMARY_PATH = os.path.join(config.GA_DIR, "ga_summary.json")
FINAL_RESULTS_PATH = os.path.join(config.FINAL_DIR, "final_results.json")

FIG_FITNESS_EVOLUTION = os.path.join(config.REPORT_DIR, "fig_fitness_evolution.png")
FIG_LEARNING_CURVES_BASELINE = os.path.join(config.REPORT_DIR, "fig_learning_curves_baseline.png")
FIG_LEARNING_CURVES_OPTIMIZED = os.path.join(config.REPORT_DIR, "fig_learning_curves_optimized.png")
FIG_ACC_VS_PARAMS = os.path.join(config.REPORT_DIR, "fig_accuracy_vs_params.png")
COMPARISON_TABLE_PATH = os.path.join(config.REPORT_DIR, "comparison_table.json")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_fitness_evolution():
    gens = _load_json(GENERATION_LOG_PATH)
    x = [g["generation"] for g in gens]
    y = [g["best_fitness"] for g in gens]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", color="#1f77b4")
    plt.xlabel("Generación")
    plt.ylabel("Mejor fitness")
    plt.title("Evolución del fitness a través de las generaciones (pyGAD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_FITNESS_EVOLUTION, dpi=150)
    plt.close()
    print(f"Guardado: {FIG_FITNESS_EVOLUTION}")


def plot_learning_curves(history, title, out_path):
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title} - Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Guardado: {out_path}")


def plot_accuracy_vs_params(baseline, final):
    all_evals = _load_json(ALL_EVALS_PATH)
    xs = [e["trainable_params"] for e in all_evals]
    ys = [e["val_accuracy"] for e in all_evals]

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, alpha=0.5, label="Individuos evaluados (GA, val)", color="#7f7f7f")
    plt.scatter(
        [baseline["trainable_params"]], [baseline["test_accuracy"]],
        color="red", marker="*", s=250, label="Baseline (Test)", zorder=5,
    )
    plt.scatter(
        [final["trainable_params"]], [final["test_accuracy"]],
        color="green", marker="*", s=250, label="Optimizado (Test)", zorder=5,
    )
    plt.xscale("log")
    plt.xlabel("Parámetros entrenables (escala log)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Parámetros entrenables")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_ACC_VS_PARAMS, dpi=150)
    plt.close()
    print(f"Guardado: {FIG_ACC_VS_PARAMS}")


def build_comparison_table(baseline, final):
    table = {
        "Accuracy en Test": {
            "Baseline": baseline["test_accuracy"],
            "Optimizada": final["test_accuracy"],
        },
        "Parametros entrenables": {
            "Baseline": baseline["trainable_params"],
            "Optimizada": final["trainable_params"],
        },
        "Tiempo de entrenamiento (s)": {
            "Baseline": baseline["train_time_seconds"],
            "Optimizada": final["train_time_seconds"],
        },
    }
    with open(COMPARISON_TABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))
    return table


def generate_all():
    baseline = _load_json(BASELINE_RESULTS_PATH)
    final = _load_json(FINAL_RESULTS_PATH)

    plot_fitness_evolution()
    plot_learning_curves(baseline["history"], "Baseline", FIG_LEARNING_CURVES_BASELINE)
    plot_learning_curves(final["history"], "Optimizado", FIG_LEARNING_CURVES_OPTIMIZED)
    plot_accuracy_vs_params(baseline, final)
    table = build_comparison_table(baseline, final)
    return table


if __name__ == "__main__":
    generate_all()
