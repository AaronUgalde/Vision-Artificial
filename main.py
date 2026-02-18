import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from Point import Point
from Cluster import Cluster
from CentroidClassifier import CentroidClassifier
import dist
import inspect

BUNDARIES = [(-100, 100), (-100, 100)]


def list_distance_functions():
    names = []
    for attr in dir(dist):
        if attr.startswith("_"):
            continue
        obj = getattr(dist, attr)

        if inspect.isfunction(obj) and obj.__module__ == dist.__name__:
            names.append(attr)

    names_sorted = sorted(names, key=lambda x: (0 if x == "euclidean" else 1, x))
    return names_sorted


def get_distance_wrapper(name):
    fn = getattr(dist, name)

    def wrapper(p1 : Point, p2 : Point):
        return fn(p1, p2)

    return wrapper

def read_representatives_from_csv(path):
    reps = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Esperamos 'label','x','y' (mínimo). Si hay más columnas (x3,x4...) se pueden usar.
        for row in reader:
            label = row.get("label") or row.get("cluster") or row.get("cls")
            if label is None:
                raise ValueError("CSV debe tener una columna 'label' (o 'cluster'/'cls').")
            # intentar detectar todas las columnas numéricas tras 'x' ordenadas
            coords = []
            # Buscamos columnas que parezcan coordenadas: 'x','y','z' o cualquier columna numérica
            # Orden simple: usamos las columnas que no son 'label'
            for k, v in row.items():
                if k.lower() == "label":
                    continue
                try:
                    coords.append(float(v))
                except Exception:
                    # ignorar no numéricas (p. ej. encabezados extra)
                    pass
            if not coords:
                raise ValueError(f"No se encontraron coordenadas numéricas en la fila: {row}")
            p = Point.from_iterable(tuple(coords))
            reps[label].append(p)
    return reps

def compute_clusters_from_reps(reps_dict):
    """
    Crea un diccionario de objetos Cluster desde
    dict label -> list[Point].

    Retorna:
        dict[label -> Cluster]
    """
    clusters = {}
    for label, reps in reps_dict.items():
        clusters[label] = Cluster(label=label, representatives=reps)
    return clusters


def plot_clusters_and_point(clusters, centroids, new_point=None, new_label=None, title_suffix=""):
    """
    Grafica representantes (por cluster), centroides y opcionalmente el punto nuevo.
    Asume datos 2D. Para >2D solo grafica las primeras 2 dimensiones.
    """

    labels = [label for label in clusters.keys()]
    cmap = cm.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    plt.figure(figsize=(8, 6))
    # representantes
    for c in clusters.values():
        xs = []
        ys = []
        for p in c.representatives:
            coords = tuple(p)
            xs.append(coords[0])
            ys.append(coords[1])
        plt.scatter(xs, ys, label=f"Reps {c.label}", alpha=0.6, s=50, color=color_map[c.label])

    # centroides
    for lab, cent in centroids.items():
        coords = tuple(cent)
        plt.scatter([coords[0]], [coords[1]], marker="X", s=200, edgecolor="k",
                    label=f"Centroid {lab}", color=color_map[lab])

        # anotar label en el centroide
        plt.text(coords[0], coords[1], f" {lab}", fontsize=10, fontweight="bold", verticalalignment="center")

    # nuevo punto
    if new_point is not None:
        px, py = tuple(new_point)[:2]
        if new_label is not None and new_label in color_map:
            color = color_map[new_label]
        else:
            color = "black"
        plt.scatter([px], [py], marker="*", s=250, label=f"Nuevo (pred: {new_label})", edgecolor="k", zorder=10, color=color)
        plt.annotate(f"({px:.2f}, {py:.2f})", xy=(px, py), xytext=(5, 5), textcoords="offset points")

    plt.title(f"Clusters y Centroides {title_suffix}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()  # bloquea hasta que se cierre la ventana

def main():
    print("Demo interactivo CentroidClassifier\n")
    path = input("Ruta al CSV de representantes (ej: reps.csv) [enter para usar 'reps.csv']: ").strip()
    if not path:
        path = "reps.csv"

    reps = read_representatives_from_csv(path)

    clusters = compute_clusters_from_reps(reps)


    dist_names = list_distance_functions()

    while True:

        centroids = {}
        for label, c in clusters.items():
            cent = c.compute_centroid()
            centroids[label] = cent

        print("Centroides calculados:")
        for lab, cent in centroids.items():
            print(f"  {lab} -> {tuple(cent)}")
        print()

        print("\nFunciones de distancia disponibles:")
        for i, name in enumerate(dist_names, start=1):
            print(f"  {i}. {name}")
        choice = input("Elige una función de distancia [enter para 'euclidean']: ").strip()
        if not choice:
            chosen_name = "euclidean" if "euclidean" in dist_names else dist_names[0]
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(dist_names):
                chosen_name = dist_names[idx]
            else:
                print("Índice fuera de rango, usando la primera opción.")
                chosen_name = dist_names[0]


        print(f"Usando distancia: {chosen_name}")
        distance_fn = get_distance_wrapper(chosen_name)

        raw = input("Introduce el nuevo punto (formato: x,y) o 'q' para salir: ").strip()
        if raw.lower() in ("q", "quit", "salir", "exit"):
            print("Saliendo.")
            break
        try:
            if "," in raw:
                parts = [p.strip() for p in raw.split(",")]
            else:
                parts = [p.strip() for p in raw.split()]
            coords = [float(p) for p in parts if p != ""]
            if len(coords) < 2:
                raise ValueError("Se requieren al menos 2 coordenadas (x,y).")
        except Exception as e:
            print("Entrada no válida:", e)
            continue

        new_point = Point.from_iterable(tuple(coords))

        # crear clasificador con la distancia elegida y ajustar
        clf = CentroidClassifier(distance=distance_fn)
        clf.bundaries = BUNDARIES
        clf.fit_from_clusters(clusters.values())
        pred = clf.predict_point(new_point)
        label = pred[0] if pred else None
        print(f"Predicción para {tuple(coords)} -> {label}")

        # graficar todo
        plot_clusters_and_point(clusters, centroids, new_point=new_point, new_label=label, title_suffix=f"(dist: {chosen_name})")

        clusters[label].add_representative(new_point)
        clusters[label].compute_centroid()
        centroids[label] = clusters[label].centroid

        cont = input("¿Clasificar otro punto? [s/N]: ").strip().lower()
        if cont not in ("s", "si", "y", "yes"):
            print("Fin.")
            break

if __name__ == "__main__":
    main()
