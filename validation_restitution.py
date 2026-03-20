"""
Método de Validación 1: Restitución
Entrena el CentroidClassifier con TODOS los puntos y predice los mismos puntos.
Genera la matriz de confusión y retorna el accuracy.
"""
import inspect
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import dist
from CentroidClassifier import CentroidClassifier
from Cluster import Cluster


def list_distance_functions():
    names = [
        attr for attr in dir(dist)
        if not attr.startswith("_")
        and inspect.isfunction(getattr(dist, attr))
        and getattr(dist, attr).__module__ == dist.__name__
    ]
    return sorted(names, key=lambda x: (0 if x == "euclidean" else 1, x))


def build_clusters(data: dict) -> dict:
    """Construye clusters desde dict {label: [{"rgb":..., "position":...}]}."""
    clusters = {}
    for label, samples in data.items():
        c = Cluster(label=label)
        for s in samples:
            if isinstance(s, dict):
                c.add_representative(s["rgb"], plot_point=s["position"])
            else:
                c.add_representative(s)
        clusters[label] = c
    return clusters


def compute_confusion_matrix(labels, y_true, y_pred):
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    return matrix


def plot_confusion_matrix(matrix, labels, title="Matriz de Confusión - Restitución", show=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_class_accuracies(class_accuracy: dict, method_name: str, show=True):
    """Gráfica de barras con el accuracy por clase."""
    labels = list(class_accuracy.keys())
    values = [class_accuracy[l] * 100 for l in labels]
    colors = plt.cm.tab10.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Clase", fontsize=12)
    ax.set_title(f"Accuracy por Clase — {method_name}", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.axhline(100, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def run_restitution(data: dict, distance_fn, show_plot=True):
    """
    Ejecuta la validación por Restitución.
    Retorna (accuracy, confusion_matrix, labels).
    """
    labels = sorted(data.keys())
    clusters = build_clusters(data)

    clf = CentroidClassifier(distance=distance_fn)
    clf.fit_from_clusters(list(clusters.values()))

    y_true, y_pred = [], []
    for label, samples in data.items():
        for s in samples:
            rgb = s["rgb"] if isinstance(s, dict) else s
            pred = clf.predict_point(rgb)
            y_true.append(label)
            y_pred.append(pred)

    matrix = compute_confusion_matrix(labels, y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true) if y_true else 0.0

    # True positives por clase (diagonal de la matriz)
    true_positives = {labels[i]: int(matrix[i, i]) for i in range(len(labels))}

    # Accuracy por clase (TP / total real de esa clase)
    class_accuracy = {}
    for i, lbl in enumerate(labels):
        total_real = int(matrix[i, :].sum())
        class_accuracy[lbl] = matrix[i, i] / total_real if total_real > 0 else 0.0

    print(f"\n[Restitución] Accuracy global: {accuracy*100:.2f}%")
    print("Matriz de Confusión:")
    print("Labels:", labels)
    print(matrix)
    print("True Positives por clase:", true_positives)
    print("Accuracy por clase:", {k: f"{v*100:.2f}%" for k, v in class_accuracy.items()})

    if show_plot:
        plot_confusion_matrix(matrix, labels,
                              title=f"Restitución — Accuracy: {accuracy*100:.2f}%")
        _plot_class_accuracies(class_accuracy, "Restitución")

    return accuracy, matrix, labels, true_positives, class_accuracy


if __name__ == "__main__":
    from image_selector import generate_image_samples, select_rectangles

    print("=== Validación por Restitución ===\n")
    img_name = input("Ruta de la imagen [enter para 'data_info/img/image.png']: ").strip() or "data_info/img/image.png"
    img = plt.imread(img_name)

    rects = select_rectangles(img_name)
    if not rects:
        raise ValueError("No seleccionaste ninguna región.")

    data = {}
    for i, rect in enumerate(rects, 1):
        nombre = input(f"Nombre de la clase {i}: ").strip() or f"clase_{i}"
        n = int(input(f"Puntos para '{nombre}': ").strip() or "30")
        data[nombre] = generate_image_samples(img, rect, n)

    dist_names = list_distance_functions()
    print("\nFunciones de distancia disponibles:")
    for i, n in enumerate(dist_names, 1):
        print(f"  {i}. {n}")
    choice = input("Elige función [enter=euclidean]: ").strip()
    dist_name = dist_names[int(choice) - 1] if choice else "euclidean"
    distance_fn = getattr(dist, dist_name)
    print(f"Usando: {dist_name}")

    run_restitution(data, distance_fn, show_plot=True)
