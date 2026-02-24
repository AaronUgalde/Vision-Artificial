import csv
from collections import defaultdict
import inspect

import matplotlib.pyplot as plt
import numpy as np

from Cluster import Cluster
from CentroidClassifier import CentroidClassifier
import dist

BOUNDARIES = np.array([[-100, 100], [-100, 100]], dtype=float)  # shape (D, 2)


def list_distance_functions():
    names = [
        attr for attr in dir(dist)
        if not attr.startswith("_")
        and inspect.isfunction(getattr(dist, attr))
        and getattr(dist, attr).__module__ == dist.__name__
    ]
    return sorted(names, key=lambda x: (0 if x == "euclidean" else 1, x))


def read_representatives_from_csv(path):
    reps = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label") or row.get("cluster") or row.get("cls")
            if label is None:
                raise ValueError("CSV debe tener una columna 'label' (o 'cluster'/'cls').")
            coords = []
            for k, v in row.items():
                if k.lower() == "label":
                    continue
                try:
                    coords.append(float(v))
                except ValueError:
                    pass
            if not coords:
                raise ValueError(f"No se encontraron coordenadas numéricas en la fila: {row}")
            reps[label].append(np.array(coords))
    return reps


def compute_clusters_from_reps(reps_dict):
    return {label: Cluster(label=label, representatives=reps) for label, reps in reps_dict.items()}


def plot_clusters_and_point(clusters, new_point=None, new_label=None, title_suffix=""):
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(clusters)}

    plt.figure(figsize=(8, 6))

    for c in clusters.values():
        arr = np.array(c.representatives)
        plt.scatter(arr[:, 0], arr[:, 1], label=f"Reps {c.label}", alpha=0.6, s=50, color=color_map[c.label])
        cx, cy = c.centroid[:2]
        plt.scatter(cx, cy, marker="X", s=200, edgecolor="k", label=f"Centroid {c.label}", color=color_map[c.label])
        plt.text(cx, cy, f" {c.label}", fontsize=10, fontweight="bold", verticalalignment="center")

    if new_point is not None:
        px, py = new_point[:2]
        color = color_map.get(new_label, "black")
        plt.scatter(px, py, marker="*", s=250, label=f"Nuevo (pred: {new_label})", edgecolor="k", zorder=10, color=color)
        plt.annotate(f"({px:.2f}, {py:.2f})", xy=(px, py), xytext=(5, 5), textcoords="offset points")

    plt.title(f"Clusters y Centroides {title_suffix}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main():
    print("Demo interactivo CentroidClassifier\n")
    path = input("Ruta al CSV de representantes [enter para 'reps.csv']: ").strip() or "reps.csv"

    clusters = compute_clusters_from_reps(read_representatives_from_csv(path))
    dist_names = list_distance_functions()

    while True:
        # Calcular/mostrar centroides
        for c in clusters.values():
            c.compute_centroid()
        print("Centroides calculados:")
        for c in clusters.values():
            print(f"  {c.label} -> {c.centroid}")

        # Elegir función de distancia
        print("\nFunciones de distancia disponibles:")
        for i, name in enumerate(dist_names, start=1):
            print(f"  {i}. {name}")
        choice = input("Elige una función de distancia [enter para 'euclidean']: ").strip()
        if not choice:
            chosen_name = "euclidean" if "euclidean" in dist_names else dist_names[0]
        else:
            idx = int(choice) - 1
            chosen_name = dist_names[idx] if 0 <= idx < len(dist_names) else dist_names[0]
        print(f"Usando distancia: {chosen_name}")

        # Leer punto nuevo
        raw = input("Introduce el nuevo punto (formato: x,y) o 'q' para salir: ").strip()
        if raw.lower() in ("q", "quit", "salir", "exit"):
            print("Saliendo.")
            break
        try:
            sep = "," if "," in raw else None
            coords = [float(v) for v in raw.split(sep) if v.strip()]
            if len(coords) < 2:
                raise ValueError("Se requieren al menos 2 coordenadas.")
        except ValueError as e:
            print("Entrada no válida:", e)
            continue

        new_point = np.array(coords)

        clf = CentroidClassifier(distance=getattr(dist, chosen_name))
        clf.boundaries = BOUNDARIES
        clf.fit_from_clusters(clusters.values())
        label = clf.predict_point(new_point)
        print(f"Predicción para {coords} -> {label}")

        plot_clusters_and_point(clusters, new_point=new_point, new_label=label, title_suffix=f"(dist: {chosen_name})")

        clusters[label].add_representative(new_point)

        if input("¿Clasificar otro punto? [s/N]: ").strip().lower() not in ("s", "si", "y", "yes"):
            print("Fin.")
            break


if __name__ == "__main__":
    main()
