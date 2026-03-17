import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Cluster import Cluster
from CentroidClassifier import CentroidClassifier
import dist


def build_clusters():
    rgb = Cluster(label="RGB")
    for p in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
        rgb.add_representative(np.array(p, dtype=float))

    cmy = Cluster(label="CMY")
    for p in [(1, 1, 0), (0, 1, 1), (1, 0, 1)]:
        cmy.add_representative(np.array(p, dtype=float))

    bw = Cluster(label="negro-blanco")
    for p in [(0, 0, 0), (1, 1, 1)]:
        bw.add_representative(np.array(p, dtype=float))

    return {"RGB": rgb, "CMY": cmy, "negro-blanco": bw}


def plot_cube(clusters, new_point=None, new_label=None):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"RGB": "blue", "CMY": "green", "negro-blanco": "gray"}

    for label, cluster in clusters.items():
        reps = np.array(cluster.representatives)
        ax.scatter(reps[:, 0], reps[:, 1], reps[:, 2],
                   color=colors[label], s=60, alpha=0.7, label=f"Reps {label}")

        c = cluster.centroid
        ax.scatter(*c, color=colors[label], marker="X", s=200,
                   edgecolor="k", zorder=5, label=f"Centroide {label}")
        ax.text(c[0], c[1], c[2], f"  {label}", fontsize=8, fontweight="bold")

    if new_point is not None:
        ax.scatter(*new_point, color="red", marker="*", s=300,
                   edgecolor="k", zorder=10, label=f"Nuevo → {new_label}")
        ax.text(new_point[0], new_point[1], new_point[2],
                f"  pred: {new_label}", fontsize=9, color="red")

    # aristas del cubo unitario
    corners = [0, 1]
    for x in corners:
        for y in corners:
            ax.plot([x, x], [y, y], [0, 1], "k--", alpha=0.2, linewidth=0.8)
    for x in corners:
        for z in corners:
            ax.plot([x, x], [0, 1], [z, z], "k--", alpha=0.2, linewidth=0.8)
    for y in corners:
        for z in corners:
            ax.plot([0, 1], [y, y], [z, z], "k--", alpha=0.2, linewidth=0.8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_title("Cubo RGB — Clusters y Centroides")
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.show()


def pedir_punto():
    print("\nIngresa un punto en el cubo RGB (valores entre 0 y 1).")
    while True:
        raw = input("Punto (r,g,b): ").strip()
        try:
            coords = [float(v) for v in raw.replace(",", " ").split() if v]
            if len(coords) != 3:
                raise ValueError("Se necesitan exactamente 3 valores.")
            p = np.array(coords, dtype=float)
            if np.any(p < 0) or np.any(p > 1):
                print("Los valores deben estar entre 0 y 1.")
                continue
            return p
        except ValueError as e:
            print("Entrada invalida:", e)


def list_distance_functions():
    import inspect
    names = [
        attr for attr in dir(dist)
        if not attr.startswith("_")
        and inspect.isfunction(getattr(dist, attr))
        and getattr(dist, attr).__module__ == dist.__name__
    ]
    return sorted(names, key=lambda x: (0 if x == "euclidean" else 1, x))


def main():
    clusters = build_clusters()

    print("=== Problema 1: Clasificacion en cubo RGB ===\n")
    print("Clases y representantes:")
    for label, c in clusters.items():
        print(f"  {label}: {[r.tolist() for r in c.representatives]}")
        print(f"    Centroide: {c.centroid.tolist()}")

    dist_names = list_distance_functions()
    print("\nFunciones de distancia disponibles:")
    for i, name in enumerate(dist_names, start=1):
        print(f"  {i}. {name}")

    choice = input("\nElige funcion de distancia [enter = euclidean]: ").strip()
    if not choice:
        chosen = "euclidean" if "euclidean" in dist_names else dist_names[0]
    else:
        idx = int(choice) - 1
        chosen = dist_names[idx] if 0 <= idx < len(dist_names) else dist_names[0]
    print(f"Usando: {chosen}\n")

    plot_cube(clusters)

    while True:
        punto = pedir_punto()

        clf = CentroidClassifier(distance=getattr(dist, chosen))
        clf.boundaries = np.array([[0, 1], [0, 1], [0, 1]], dtype=float)
        clf.fit_from_clusters(clusters.values())
        label = clf.predict_point(punto)

        print(f"\nPunto {punto.tolist()} clasificado como: {label}\n")
        plot_cube(clusters, new_point=punto, new_label=label)

        if input("Clasificar otro punto? [s/N]: ").strip().lower() not in ("s", "si", "y", "yes"):
            print("Fin.")
            break


if __name__ == "__main__":
    main()
