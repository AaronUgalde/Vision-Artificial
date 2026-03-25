"""
Método de Validación 3: 50-50
Divide aleatoriamente los puntos en 50% entrenamiento / 50% validación.
Repite 20 veces y promedia los resultados.
Genera la matriz de confusión promedio y retorna el accuracy promedio.
"""
import inspect
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import dist
from CentroidClassifier import CentroidClassifier
from Cluster import Cluster
from validation_restitution import (
    build_clusters, compute_confusion_matrix, plot_confusion_matrix,
    list_distance_functions, _plot_class_accuracies
)


def run_50_50(data: dict, distance_fn, n_iterations: int = 20, show_plot=True):
    """
    Ejecuta la validación 50-50 durante n_iterations iteraciones.
    Retorna (accuracy_promedio, confusion_matrix_promedio, labels).
    """
    labels = sorted(data.keys())
    n_labels = len(labels)

    # Aplanar todos los puntos con etiqueta

    accumulated_matrix = np.zeros((n_labels, n_labels), dtype=float)
    accuracies = []

    if any(len(s) < 2 for s in data.values()):
        raise ValueError("Cada clase necesita al menos 2 puntos para 50-50.")

    for iteration in range(n_iterations):
        train_pool, test_pool = [], []
        for lbl, samples in data.items():
            shuffled_class = samples.copy()
            random.shuffle(shuffled_class)
            mid = len(shuffled_class) // 2
            for s in shuffled_class[:mid]:
                train_pool.append((lbl, s))
            for s in shuffled_class[mid:]:
                test_pool.append((lbl, s))

        train_data = defaultdict(list)
        for lbl, s in train_pool:
            train_data[lbl].append(s)

        # Si alguna clase no tiene representantes en entrenamiento, saltamos iteración
        if any(lbl not in train_data for lbl in labels):
            continue

        train_clusters = build_clusters(train_data)
        clf = CentroidClassifier(distance=distance_fn)
        clf.fit_from_clusters(list(train_clusters.values()))

        y_true, y_pred = [], []
        for lbl, s in test_pool:
            rgb = s["rgb"] if isinstance(s, dict) else s
            try:
                pred = clf.predict_point(rgb)
            except Exception:
                pred = labels[0]
            y_true.append(lbl)
            y_pred.append(pred)

        mat = compute_confusion_matrix(labels, y_true, y_pred)
        accumulated_matrix += mat

        correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc = correct / len(y_true) if y_true else 0.0
        accuracies.append(acc)

        # Mostrar matriz de cada iteración en consola (requerimiento 50-50)
        print(f"\n  --- Iteración {iteration+1}/{n_iterations} --- accuracy = {acc*100:.2f}%")
        print(f"  Labels: {labels}")
        print(f"  Matriz de Confusión:")
        print(mat)

    if not accuracies:
        raise RuntimeError("Ninguna iteración completó exitosamente.")

    avg_matrix = accumulated_matrix / len(accuracies)
    avg_accuracy = float(np.mean(accuracies))
    std_accuracy = float(np.std(accuracies))

    # True positives promedio por clase (diagonal de la matriz promedio)
    true_positives = {labels[i]: float(avg_matrix[i, i]) for i in range(len(labels))}

    # Accuracy promedio por clase
    class_accuracy = {}
    for i, lbl in enumerate(labels):
        total_real = float(avg_matrix[i, :].sum())
        class_accuracy[lbl] = avg_matrix[i, i] / total_real if total_real > 0 else 0.0

    print(f"\n[50-50] Accuracy promedio: {avg_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    print("Matriz de Confusión Promedio:")
    print("Labels:", labels)
    print(np.round(avg_matrix, 2))
    print("True Positives promedio por clase:", {k: f"{v:.2f}" for k, v in true_positives.items()})
    print("Accuracy por clase:", {k: f"{v*100:.2f}%" for k, v in class_accuracy.items()})

    if show_plot:
        _plot_avg_confusion(avg_matrix, labels, avg_accuracy, std_accuracy)
        _plot_accuracy_per_iteration(accuracies)
        _plot_class_accuracies(class_accuracy, "50-50 (promedio)")

    return avg_accuracy, avg_matrix, labels, true_positives, class_accuracy


def _plot_avg_confusion(matrix, labels, avg_acc, std_acc, show=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"50-50 (20 iters) — Acc promedio: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    color="white" if matrix[i, j] > vmax / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_accuracy_per_iteration(accuracies, show=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    iters = range(1, len(accuracies) + 1)
    ax.plot(iters, [a * 100 for a in accuracies], marker="o", color="steelblue", linewidth=2)
    ax.axhline(np.mean(accuracies) * 100, color="red", linestyle="--",
               label=f"Promedio: {np.mean(accuracies)*100:.2f}%")
    ax.set_xlabel("Iteración"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy por Iteración — Validación 50-50")
    ax.legend(); ax.grid(True); plt.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    from image_selector import generate_image_samples, select_rectangles

    print("=== Validación 50-50 ===\n")
    img_name = input("Ruta de la imagen [enter para 'data_info/img/image.png']: ").strip() or "data_info/img/image.png"
    img = plt.imread(img_name)

    rects = select_rectangles(img_name)
    if not rects:
        raise ValueError("No seleccionaste ninguna región.")

    data = {}
    for i, rect in enumerate(rects, 1):
        nombre = input(f"Nombre de la clase {i}: ").strip() or f"clase_{i}"
        n = int(input(f"Puntos para '{nombre}': ").strip() or "40")
        data[nombre] = generate_image_samples(img, rect, n)

    dist_names = list_distance_functions()
    print("\nFunciones de distancia disponibles:")
    for i, n in enumerate(dist_names, 1):
        print(f"  {i}. {n}")
    choice = input("Elige función [enter=euclidean]: ").strip()
    dist_name = dist_names[int(choice) - 1] if choice else "euclidean"
    distance_fn = getattr(dist, dist_name)
    print(f"Usando: {dist_name}")

    run_50_50(data, distance_fn, n_iterations=20, show_plot=True)
