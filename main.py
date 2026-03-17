import inspect
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import dist
from CentroidClassifier import CentroidClassifier
from Cluster import Cluster
from image_selector import generate_image_samples, select_rectangles


def list_distance_functions():
    names = [
        attr for attr in dir(dist)
        if not attr.startswith("_")
        and inspect.isfunction(getattr(dist, attr))
        and getattr(dist, attr).__module__ == dist.__name__
    ]
    return sorted(names, key=lambda x: (0 if x == "euclidean" else 1, x))


def _pedir_int(msg, min_val=1):
    while True:
        try:
            v = int(input(msg).strip())
            if v >= min_val:
                return v
            print(f"Debe ser >= {min_val}.")
        except ValueError:
            print("Ingresa un entero valido.")


def _rgb_boundaries(img):
    rgb = np.asarray(img[..., :3], dtype=float)
    mins = rgb.reshape(-1, rgb.shape[-1]).min(axis=0)
    maxs = rgb.reshape(-1, rgb.shape[-1]).max(axis=0)
    return np.column_stack((mins, maxs))


def _point_to_rgb(img, coords):
    x, y = int(round(coords[0])), int(round(coords[1]))
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("El punto esta fuera de los limites de la imagen.")
    return np.asarray(img[y, x], dtype=float)[:3]


def _get_cluster_plot_points(cluster):
    if cluster.plot_points:
        return np.array(cluster.plot_points, dtype=float)
    return np.array(cluster.representatives, dtype=float)


def plot_clusters_and_point(clusters, new_point=None, new_label=None, title_suffix="", image_path=None):
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(clusters)}
    plt.figure(figsize=(8, 6))

    if image_path:
        img = plt.imread(image_path)
        h, w = img.shape[:2]
        plt.imshow(img, extent=[0, w - 1, h - 1, 0])

    for c in clusters.values():
        arr = _get_cluster_plot_points(c)
        if arr.size == 0:
            continue
        plt.scatter(arr[:, 0], arr[:, 1], label=f"Reps {c.label}", alpha=0.6, s=50, color=color_map[c.label])
        plot_centroid = c.compute_plot_centroid() or c.centroid[:2]
        cx, cy = plot_centroid[:2]
        plt.scatter(cx, cy, marker="X", s=200, edgecolor="k", label=f"Centroid {c.label}", color=color_map[c.label])
        plt.text(cx, cy, f" {c.label}", fontsize=10, fontweight="bold", verticalalignment="center")

    if new_point is not None:
        px, py = new_point[:2]
        plt.scatter(px, py, marker="*", s=250, label=f"Nuevo (pred: {new_label})", edgecolor="k", zorder=10, color=color_map.get(new_label, "black"))
        plt.annotate(f"({px:.2f}, {py:.2f})", xy=(px, py), xytext=(5, 5), textcoords="offset points")

    plt.title(f"Clusters y Centroides {title_suffix}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


def go_centroid_classifier():
    print("\nDemo interactivo CentroidClassifier\n")

    path = "data_info/img/" + (input("Ruta de la imagen [enter para 'beach.jpg']: ").strip() or "beach.jpg")
    rects = select_rectangles(path)
    if not rects:
        raise ValueError("No seleccionaste ninguna region.")

    img = plt.imread(path)
    clases = defaultdict(list)
    for i, rect in enumerate(rects, start=1):
        nombre = input(f"Nombre de la clase {i}: ").strip() or f"clase_{i}"
        n = _pedir_int(f"Cantidad de puntos para '{nombre}': ", min_val=1)
        clases[nombre].extend(generate_image_samples(img, rect, n))

    clusters = {}
    for label, puntos in clases.items():
        c = Cluster(label=label)
        for p in puntos:
            if isinstance(p, dict):
                c.add_representative(p["rgb"], plot_point=p["position"])
            else:
                c.add_representative(p)
        clusters[label] = c

    h, w = img.shape[:2]
    input_boundaries = np.array([[0, w - 1], [0, h - 1]], dtype=float)
    classifier_boundaries = _rgb_boundaries(img)
    dist_names = list_distance_functions()

    while True:
        for c in clusters.values():
            c.compute_centroid()

        print("\nCentroides calculados:")
        for c in clusters.values():
            print(f"  {c.label} -> {c.centroid}")

        print("\nFunciones de distancia:")
        for i, name in enumerate(dist_names, 1):
            print(f"  {i}. {name}")
        choice = input("\nElige funcion [enter para 'euclidean']: ").strip()
        chosen_name = dist_names[int(choice) - 1] if choice else ("euclidean" if "euclidean" in dist_names else dist_names[0])
        print(f"Usando: {chosen_name}")

        raw = input("\nPunto (x,y) o 'q' para salir: ").strip()
        if raw.lower() in ("q", "quit", "salir", "exit"):
            print("Saliendo."); break

        try:
            coords = [float(v) for v in raw.split(",") if v.strip()]
            if len(coords) < 2:
                raise ValueError("Se requieren al menos 2 coordenadas.")
        except ValueError as e:
            print("Entrada no valida:", e); continue

        display_point = np.array(coords[:2], dtype=float)
        if np.any(display_point < input_boundaries[:, 0]) or np.any(display_point > input_boundaries[:, 1]):
            print("El punto esta fuera de los limites."); continue

        new_point = _point_to_rgb(img, display_point)

        clf = CentroidClassifier(distance=getattr(dist, chosen_name))
        clf.boundaries = classifier_boundaries
        clf.fit_from_clusters(clusters.values())
        label = clf.predict_point(new_point)

        if chosen_name == "probability_gaussian":
            pdfs = np.array([dist.probability_gaussian(new_point, c.centroid, np.array(c.representatives, dtype=float)) for c in clf.clusters])
            s = pdfs.sum()
            if s and np.isfinite(s):
                print("\nProbabilidades:")
                for c, p in sorted(zip(clf.clusters, pdfs / s), key=lambda t: t[1], reverse=True):
                    print(f"  {c.label}: {p*100:.2f}%")

        print(f"Prediccion para {display_point.tolist()} con RGB {new_point.tolist()} -> {label}")
        plot_clusters_and_point(clusters, new_point=display_point, new_label=label, title_suffix=f"(dist: {chosen_name})", image_path=path)

        clusters[label].add_representative(new_point, plot_point=display_point)

        if input("\nClasificar otro punto? [s/N]: ").strip().lower() not in ("s", "si", "y", "yes"):
            print("Fin."); break


if __name__ == "__main__":
    go_centroid_classifier()
