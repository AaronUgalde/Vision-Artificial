"""
main_validation.py
==================
Programa principal que:
  1. Pide al usuario la imagen y las regiones de cada clase.
  2. Pide la función de distancia a usar.
  3. Ejecuta los 3 métodos de validación:
       - Restitución
       - Leave-One-Out
       - 50-50 (20 iteraciones, promediado)
  4. Muestra una gráfica de barras comparando el accuracy de los 3 métodos.
"""

import inspect
import matplotlib.pyplot as plt
import numpy as np

import dist
from image_selector import generate_image_samples, select_rectangles
from validation_restitution import run_restitution, list_distance_functions
from validation_leave_one_out import run_leave_one_out
from validation_50_50 import run_50_50


# ─────────────────────────────────────────────────────────────
# Helpers de consola
# ─────────────────────────────────────────────────────────────

def _pedir_int(msg, min_val=1):
    while True:
        try:
            v = int(input(msg).strip())
            if v >= min_val:
                return v
            print(f"Debe ser >= {min_val}.")
        except ValueError:
            print("Ingresa un entero válido.")


def elegir_distancia():
    dist_names = list_distance_functions()
    print("\nFunciones de distancia disponibles:")
    for i, name in enumerate(dist_names, 1):
        print(f"  {i}. {name}")
    choice = input("\nElige función [enter=euclidean]: ").strip()
    if choice:
        try:
            name = dist_names[int(choice) - 1]
        except (IndexError, ValueError):
            print("Opción inválida, usando euclidean.")
            name = "euclidean"
    else:
        name = "euclidean" if "euclidean" in dist_names else dist_names[0]
    print(f"Usando distancia: {name}\n")
    return getattr(dist, name), name


# ─────────────────────────────────────────────────────────────
# Gráfica de barras comparativa
# ─────────────────────────────────────────────────────────────

def plot_comparison(accuracies: dict, dist_name: str):
    """
    accuracies = {"Restitución": 0.95, "Leave-One-Out": 0.88, "50-50": 0.82}
    """
    methods = list(accuracies.keys())
    values = [accuracies[m] * 100 for m in methods]

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(methods, values, color=colors, edgecolor="black", width=0.5, zorder=3)

    # Etiquetas sobre cada barra
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.2f}%",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"Comparación de Métodos de Validación\n(distancia: {dist_name})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Método de Validación", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Línea de referencia al 100 %
    ax.axhline(100, color="gray", linestyle=":", linewidth=1)

    plt.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# Programa principal
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   COMPARACIÓN DE MÉTODOS DE VALIDACIÓN — CentroidClassifier")
    print("=" * 60)

    # 1. Imagen
    img_name = (
        input("\nRuta de la imagen [enter para 'data_info/img/image.png']: ").strip()
        or "data_info/img/image.png"
    )
    try:
        img = plt.imread(img_name)
    except FileNotFoundError:
        print(f"No se encontró '{img_name}'. Verifica la ruta.")
        return

    # 2. Seleccionar regiones (rectángulos) interactivamente
    print("\nSelecciona las regiones de cada clase en la imagen.")
    print("Dibuja rectángulos con click-arrastrar. Presiona Enter al terminar.\n")
    rects = select_rectangles(img_name)
    if not rects:
        print("No seleccionaste ninguna región. Saliendo.")
        return

    # 3. Asignar nombre y cantidad de puntos a cada región
    data = {}
    for i, rect in enumerate(rects, 1):
        nombre = input(f"\nNombre de la clase {i} (rect {rect}): ").strip() or f"clase_{i}"
        n = _pedir_int(f"  Cantidad de puntos para '{nombre}' [mínimo 10]: ", min_val=2)
        samples = generate_image_samples(img, rect, n)
        if nombre in data:
            data[nombre].extend(samples)
        else:
            data[nombre] = samples
        print(f"  -> {len(samples)} muestras generadas para '{nombre}'.")

    print(f"\nClases: {list(data.keys())}")
    total_pts = sum(len(v) for v in data.values())
    print(f"Total de puntos: {total_pts}")

    # 4. Función de distancia
    distance_fn, dist_name = elegir_distancia()

    # ── Validación 1: Restitución ──────────────────────────────
    print("\n" + "─" * 50)
    print("MÉTODO 1 — RESTITUCIÓN")
    print("─" * 50)
    acc_rest, _, _ = run_restitution(data, distance_fn, show_plot=True)

    # ── Validación 2: Leave-One-Out ────────────────────────────
    print("\n" + "─" * 50)
    print("MÉTODO 2 — LEAVE-ONE-OUT")
    print(f"  (iteraciones: {total_pts})")
    print("─" * 50)
    acc_loo, _, _ = run_leave_one_out(data, distance_fn, show_plot=True)

    # ── Validación 3: 50-50 ───────────────────────────────────
    print("\n" + "─" * 50)
    print("MÉTODO 3 — 50-50  (20 iteraciones)")
    print("─" * 50)
    acc_5050, _, _ = run_50_50(data, distance_fn, n_iterations=20, show_plot=True)

    # ── Resumen y gráfica comparativa ─────────────────────────
    print("\n" + "=" * 50)
    print("RESUMEN FINAL")
    print("=" * 50)
    print(f"  Restitución   : {acc_rest  * 100:.2f}%")
    print(f"  Leave-One-Out : {acc_loo   * 100:.2f}%")
    print(f"  50-50 (prom.) : {acc_5050  * 100:.2f}%")
    print("=" * 50)

    best = max(
        [("Restitución", acc_rest), ("Leave-One-Out", acc_loo), ("50-50", acc_5050)],
        key=lambda x: x[1]
    )
    print(f"\n  Mejor método  : {best[0]} ({best[1]*100:.2f}%)")

    accuracies = {
        "Restitución": acc_rest,
        "Leave-One-Out": acc_loo,
        "50-50": acc_5050,
    }
    plot_comparison(accuracies, dist_name)


if __name__ == "__main__":
    main()
