"""
Método de Validación 2: Leave-One-Out (LOO)
Para cada punto: entrena con TODOS los demás puntos, predice el punto dejado fuera.
Genera la matriz de confusión y retorna el accuracy.
"""
import inspect
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


def run_leave_one_out(data: dict, distance_fn, show_plot=True):
    """
    Ejecuta la validación Leave-One-Out.
    Retorna (accuracy, confusion_matrix, labels).
    """
    labels = sorted(data.keys())

    # Aplanar todos los puntos con su etiqueta
    all_samples = []
    for label, samples in data.items():
        for s in samples:
            all_samples.append((label, s))

    total = len(all_samples)
    if total < 2:
        raise ValueError("Se necesitan al menos 2 puntos para LOO.")

    y_true, y_pred = [], []

    for i in range(total):
        # Construir datos de entrenamiento sin el punto i
        train_data = defaultdict(list)
        for j, (lbl, s) in enumerate(all_samples):
            if j != i:
                train_data[lbl].append(s)

        # Verificar que todas las clases tengan al menos 1 representante
        if any(len(v) == 0 for v in train_data.values()):
            continue

        train_clusters = build_clusters(train_data)

        # Solo predecir si hay al menos 2 clusters
        if len(train_clusters) < 1:
            continue

        clf = CentroidClassifier(distance=distance_fn)
        clf.fit_from_clusters(list(train_clusters.values()))

        test_label, test_sample = all_samples[i]
        rgb = test_sample["rgb"] if isinstance(test_sample, dict) else test_sample

        try:
            pred = clf.predict_point(rgb)
        except Exception:
            pred = list(train_clusters.keys())[0]

        y_true.append(test_label)
        y_pred.append(pred)

    matrix = compute_confusion_matrix(labels, y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true) if y_true else 0.0

    # True positives por clase
    true_positives = {labels[i]: int(matrix[i, i]) for i in range(len(labels))}

    # Accuracy por clase
    class_accuracy = {}
    for i, lbl in enumerate(labels):
        total_real = int(matrix[i, :].sum())
        class_accuracy[lbl] = matrix[i, i] / total_real if total_real > 0 else 0.0

    print(f"\n[Leave-One-Out] Accuracy global: {accuracy*100:.2f}%")
    print("Matriz de Confusión:")
    print("Labels:", labels)
    print(matrix)
    print("True Positives por clase:", true_positives)
    print("Accuracy por clase:", {k: f"{v*100:.2f}%" for k, v in class_accuracy.items()})

    if show_plot:
        plot_confusion_matrix(matrix, labels,
                              title=f"Leave-One-Out — Accuracy: {accuracy*100:.2f}%")
        _plot_class_accuracies(class_accuracy, "Leave-One-Out")

    return accuracy, matrix, labels, true_positives, class_accuracy


if __name__ == "__main__":
    from image_selector import generate_image_samples, select_rectangles

    print("=== Validación Leave-One-Out ===\n")
    img_name = input("Ruta de la imagen [enter para 'data_info/img/image.png']: ").strip() or "data_info/img/image.png"
    img = plt.imread(img_name)

    rects = select_rectangles(img_name)
    if not rects:
        raise ValueError("No seleccionaste ninguna región.")

    data = {}
    for i, rect in enumerate(rects, 1):
        nombre = input(f"Nombre de la clase {i}: ").strip() or f"clase_{i}"
        n = int(input(f"Puntos para '{nombre}': ").strip() or "20")
        data[nombre] = generate_image_samples(img, rect, n)

    dist_names = list_distance_functions()
    print("\nFunciones de distancia disponibles:")
    for i, n in enumerate(dist_names, 1):
        print(f"  {i}. {n}")
    choice = input("Elige función [enter=euclidean]: ").strip()
    dist_name = dist_names[int(choice) - 1] if choice else "euclidean"
    distance_fn = getattr(dist, dist_name)
    print(f"Usando: {dist_name}")

    run_leave_one_out(data, distance_fn, show_plot=True)
