"""
main_validation.py
==================
Programa principal que:
  1. Pide al usuario la imagen y las regiones de cada clase.
  2. Pide la función de distancia a usar.
  3. Ejecuta los 3 métodos de validación:
       - Restitución
       - Leave-One-Out
       - 50-50 (20 iteraciones, con matrices por iteración en consola)
  4. Por cada método:
       - Muestra la matriz de confusión.
       - Obtiene los True Positives por clase.
       - Plotea una gráfica de barras de accuracy por clase.
       - Calcula el accuracy global.
  5. Guarda toda la información en un JSON con el nombre de la distancia usada.
  6. Al finalizar todos los métodos, guarda en el JSON cuál fue el mejor método
     de validación y su accuracy.
  7. Permite probar con otra distancia y repite el proceso.
  8. Al agotar todas las distancias (o al finalizar), lee todos los JSON generados
     e informa cuál distancia tuvo el mejor desempeño.
"""

import json
import os
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


def elegir_distancia(usadas: list = None):
    """Muestra las distancias disponibles y retorna (fn, nombre).
    Si `usadas` se da, marca cuáles ya fueron probadas."""
    dist_names = list_distance_functions()
    usadas = usadas or []
    print("\nFunciones de distancia disponibles:")
    for i, name in enumerate(dist_names, 1):
        tag = "  [ya usada]" if name in usadas else ""
        print(f"  {i}. {name}{tag}")
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
# Gráfica comparativa de los 3 métodos (accuracy global)
# ─────────────────────────────────────────────────────────────

def plot_comparison(accuracies: dict, dist_name: str):
    methods = list(accuracies.keys())
    values = [accuracies[m] * 100 for m in methods]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, values, color=colors, edgecolor="black", width=0.5, zorder=3)
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
    ax.axhline(100, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────
# Guardar / cargar JSON de resultados
# ─────────────────────────────────────────────────────────────

def _matriz_a_lista(matrix):
    """Convierte numpy array (o array de floats) a lista serializable."""
    if hasattr(matrix, "tolist"):
        return matrix.tolist()
    return [[float(v) for v in row] for row in matrix]


def guardar_json(dist_name: str, resultados: dict):
    """Guarda el JSON con nombre = dist_name.json en el directorio actual."""
    path = f"{dist_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\n  [JSON] Resultados guardados en: {path}")


def cargar_todos_json(dist_names: list) -> dict:
    """Carga todos los JSON generados y retorna dict {dist_name: datos}."""
    datos = {}
    for name in dist_names:
        path = f"{name}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                datos[name] = json.load(f)
    return datos


# ─────────────────────────────────────────────────────────────
# Ejecutar los 3 métodos para una distancia dada
# ─────────────────────────────────────────────────────────────

def ejecutar_validaciones(data: dict, distance_fn, dist_name: str) -> dict:
    """
    Corre los 3 métodos de validación, plotea y retorna el dict de resultados
    listo para serializar a JSON.
    """
    total_pts = sum(len(v) for v in data.values())
    resultados = {"distancia": dist_name, "metodos": {}}

    # ── Restitución ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("MÉTODO 1 — RESTITUCIÓN")
    print("─" * 50)
    acc_rest, mat_rest, labels_rest, tp_rest, ca_rest = run_restitution(
        data, distance_fn, show_plot=True)

    resultados["metodos"]["Restitución"] = {
        "accuracy_global": acc_rest,
        "true_positives_por_clase": tp_rest,
        "accuracy_por_clase": ca_rest,
        "matriz_confusion": _matriz_a_lista(mat_rest),
        "labels": labels_rest,
    }


    # ── Leave-One-Out ────────────────────────────────────────
    print("\n" + "─" * 50)
    print("MÉTODO 2 — LEAVE-ONE-OUT")
    print(f"  (iteraciones: {total_pts})")
    print("─" * 50)
    acc_loo, mat_loo, labels_loo, tp_loo, ca_loo = run_leave_one_out(
        data, distance_fn, show_plot=True)

    resultados["metodos"]["Leave-One-Out"] = {
        "accuracy_global": acc_loo,
        "true_positives_por_clase": tp_loo,
        "accuracy_por_clase": ca_loo,
        "matriz_confusion": _matriz_a_lista(mat_loo),
        "labels": labels_loo,
    }

    # ── 50-50 ────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("MÉTODO 3 — 50-50  (20 iteraciones, matrices en consola)")
    print("─" * 50)
    acc_5050, mat_5050, labels_5050, tp_5050, ca_5050 = run_50_50(
        data, distance_fn, n_iterations=20, show_plot=True)

    resultados["metodos"]["50-50"] = {
        "accuracy_global": acc_5050,
        "true_positives_por_clase": tp_5050,
        "accuracy_por_clase": ca_5050,
        "matriz_confusion_promedio": _matriz_a_lista(mat_5050),
        "labels": labels_5050,
    }

    # ── Resumen y mejor método ───────────────────────────────
    print("\n" + "=" * 50)
    print("RESUMEN FINAL")
    print("=" * 50)
    accs = {
        "Restitución": acc_rest,
        "Leave-One-Out": acc_loo,
        "50-50": acc_5050,
    }
    for metodo, acc in accs.items():
        print(f"  {metodo:<16}: {acc * 100:.2f}%")
    print("=" * 50)

    mejor_metodo, mejor_acc = max(accs.items(), key=lambda x: x[1])
    print(f"\n  ★ Mejor método : {mejor_metodo} ({mejor_acc * 100:.2f}%)")

    resultados["mejor_metodo"] = mejor_metodo
    resultados["mejor_accuracy"] = mejor_acc

    # Gráfica comparativa de los 3 métodos
    plot_comparison(accs, dist_name)

    return resultados


# ─────────────────────────────────────────────────────────────
# Comparación final entre todas las distancias probadas
# ─────────────────────────────────────────────────────────────

def mostrar_mejor_distancia(dist_names_probadas: list):
    """
    Lee todos los JSON generados y muestra en terminal cuál distancia
    tuvo el mejor desempeño (mayor accuracy del mejor método de validación).
    """
    print("\n" + "=" * 60)
    print("  COMPARACIÓN FINAL — TODAS LAS DISTANCIAS PROBADAS")
    print("=" * 60)

    todos = cargar_todos_json(dist_names_probadas)
    if not todos:
        print("  No se encontraron archivos JSON de resultados.")
        return

    ranking = []
    for dname, datos in todos.items():
        mejor_metodo = datos.get("mejor_metodo", "?")
        mejor_acc = datos.get("mejor_accuracy", 0.0)
        ranking.append((dname, mejor_metodo, mejor_acc))
        print(f"  {dname:<35} | Mejor método: {mejor_metodo:<16} | Accuracy: {mejor_acc*100:.2f}%")

    print("-" * 60)
    ganador = max(ranking, key=lambda x: x[2])
    print(f"\n  🏆 MEJOR DISTANCIA : {ganador[0]}")
    print(f"     Mejor método    : {ganador[1]}")
    print(f"     Accuracy        : {ganador[2]*100:.2f}%")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Programa principal
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   COMPARACIÓN DE MÉTODOS DE VALIDACIÓN — CentroidClassifier")
    print("=" * 60)

    # 1. Imagen
    img_name = (
        input("\nRuta de la imagen [enter para 'data_info/img/beach.png']: ").strip()
        or "data_info/img/beach.png"
    )
    try:
        img = plt.imread(img_name)
    except FileNotFoundError:
        print(f"No se encontró '{img_name}'. Verifica la ruta.")
        return

    # 2. Seleccionar regiones interactivamente
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
        n = _pedir_int(f"  Cantidad de puntos para '{nombre}' [mínimo 2]: ", min_val=2)
        samples = generate_image_samples(img, rect, n)
        if nombre in data:
            data[nombre].extend(samples)
        else:
            data[nombre] = samples
        print(f"  -> {len(samples)} muestras generadas para '{nombre}'.")

    print(f"\nClases: {list(data.keys())}")
    total_pts = sum(len(v) for v in data.values())
    print(f"Total de puntos: {total_pts}")


    # 4. Loop de distancias
    all_dist_names = list_distance_functions()
    distancias_usadas = []

    while True:
        # ── Verificar si ya se probaron todas ─────────────────
        restantes = [d for d in all_dist_names if d not in distancias_usadas]
        if not restantes:
            print("\n  ✔ Se han probado TODAS las distancias disponibles.")
            mostrar_mejor_distancia(distancias_usadas)
            break

        # ── Elegir distancia ──────────────────────────────────
        distance_fn, dist_name = elegir_distancia(usadas=distancias_usadas)

        if dist_name in distancias_usadas:
            print(f"  La distancia '{dist_name}' ya fue probada. Elige otra.")
            continue

        # ── Ejecutar validaciones y guardar JSON ──────────────
        resultados = ejecutar_validaciones(data, distance_fn, dist_name)
        guardar_json(dist_name, resultados)
        distancias_usadas.append(dist_name)

        # ── Preguntar si continuar con otra distancia ─────────
        print(f"\n  Distancias probadas hasta ahora: {distancias_usadas}")
        restantes_ahora = [d for d in all_dist_names if d not in distancias_usadas]

        if not restantes_ahora:
            print("\n  ✔ Se han probado TODAS las distancias disponibles.")
            mostrar_mejor_distancia(distancias_usadas)
            break

        otra = input(
            f"\n¿Deseas probar con otra distancia? "
            f"(quedan: {restantes_ahora}) [s/N]: "
        ).strip().lower()

        if otra != "s":
            # El usuario no quiere seguir — comparar lo probado hasta ahora
            if len(distancias_usadas) > 1:
                mostrar_mejor_distancia(distancias_usadas)
            else:
                print("\n  Solo se probó una distancia. ¡Hasta luego!")
            break


if __name__ == "__main__":
    main()
