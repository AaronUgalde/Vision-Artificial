import csv
import dist
import numpy as np
import inspect
import matplotlib.pyplot as plt

from Cluster            import Cluster
from collections        import defaultdict
from CentroidClassifier import CentroidClassifier
from random_csv         import generar_csv_clusters_por_clase
from image_selector     import get_rgb_at, select_rectangles, generate_points

BOUNDARIES = np.array([[-100, 100], [-100, 100]], dtype=float)  # shape (D, 2)
IMAGE_BOUNDS = np.array([[-1, 257], [-1, 257], [-1, 257]], dtype=float)  # shape (D, 2)


def list_distance_functions():
    names = [
        attr for attr in dir(dist)
        if not attr.startswith("_")
        and inspect.isfunction(getattr(dist, attr))
        and getattr(dist, attr).__module__ == dist.__name__
    ]
    return sorted(names, key=lambda x: (0 if x == "euclidean" else 1, x))

def classes_by_image():
    path = "data_info/img/" + (input("Ruta de la imagen [enter para 'beach.jsp']: ").strip() or "beach.jpg")
    rects = select_rectangles(path)
    
    if not rects:
        raise ValueError("No seleccionaste ninguna región.")

    img = plt.imread(path)
    clases = defaultdict(list)

    for i, r in enumerate(rects, start=1):
        nombre = input(f"Nombre de la clase {i}: ").strip()
        if not nombre:
            nombre = f"clase_{i}"

        n = _pedir_int(f"Cantidad de puntos para la clase '{nombre}': ", min_val=1)

        puntos = generate_points(r, n, img)
        clases[nombre].extend(puntos)

    return clases, path

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

def plot_clusters_and_point(clusters, new_point=None, new_label=None, title_suffix="", image_path=None):
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(clusters)}

    plt.figure(figsize=(8, 6))

    if image_path is not None:
        img = plt.imread(image_path)
        h, w = img.shape[:2]
        plt.imshow(img, extent=[0, w - 1, h - 1, 0])

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
    plt.tight_layout()
    plt.show()

def _pedir_int(msg: str, min_val: int = 1) -> int:
    while True:
        try:
            v = int(input(msg).strip())
            if v < min_val:
                print(f"Debe ser >= {min_val}.")
                continue
            return v
        except ValueError:
            print("Ingresa un entero válido.")

def _parsear_par(s: str) -> tuple[float, float]:
    partes = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(partes) == 1:
        v = float(partes[0])
        return (v, v)
    if len(partes) == 2:
        return (float(partes[0]), float(partes[1]))
    raise ValueError("Formato inválido. Usa 'v' o 'x,y'.")

def _pedir_par(msg: str) -> tuple[float, float]:
    while True:
        try:
            return _parsear_par(input(msg).strip())
        except ValueError as e:
            print(e)

def _pedir_dispersion(msg: str) -> tuple[float, float]:
    while True:
        try:
            dx, dy = _parsear_par(input(msg).strip())
            if dx <= 0 or dy <= 0:
                print("La dispersión debe ser > 0.")
                continue
            return dx, dy
        except ValueError as e:
            print(e)

def go_centroid_classifier():

    print("\nDemo interactivo CentroidClassifier\n")

    while True:
        modo = input("Modo de entrada (1=CSV, 2=Imagen): ")

        if modo == "1":
            path = "data_info/" + (input("Ruta al CSV de representantes [enter para 'reps.csv']: ").strip() or "reps.csv")
            if bool(input("\n¿Actualizar datos? [s/N]: ").strip().lower() in ("s", "si", "y", "yes")):
                num_clases = _pedir_int("\n¿Cuántas clases quieres?: ", min_val=1)
                centros_por_clase = []
                reps_por_clase = []
                disp_por_clase = []

                for i in range(num_clases):
                    cx, cy = _pedir_par(f"\nClase {i}: Centro: ")
                    reps = _pedir_int(f"Clase {i}: ¿Cuántos representantes?: ", min_val=1)
                    dx, dy = _pedir_dispersion(f"Clase {i}: Dispersión: ")

                    centros_por_clase.append((cx, cy))
                    reps_por_clase.append(reps)
                    disp_por_clase.append((dx, dy))

                generar_csv_clusters_por_clase(
                    centros_por_clase=centros_por_clase,
                    reps_por_clase=reps_por_clase,
                    disp_por_clase=disp_por_clase,
                    archivo_csv=path,
                )
                print(f"\n{path} actualizado con {num_clases} clases.")

            clusters = compute_clusters_from_reps(read_representatives_from_csv(path))
            boundaries = BOUNDARIES
            image_path = None
            dist_names = list_distance_functions()
            break
        elif modo == "2":
            reps_dict, image_path = classes_by_image()
            clusters = compute_clusters_from_reps(defaultdict(list, {
                k: [t[-3:] for t in v[2:]]   # v[2:] quita 2 primeros puntos; t[-3:] deja (r,g,b)
                for k, v in reps_dict.items()
            }))

            plot_clusters = compute_clusters_from_reps(defaultdict(list, {
                k: [t[:2] for t in v[:2]]   # t[:2] => (x, y)
                for k, v in reps_dict.items()
            })) 

            img = plt.imread(image_path)
            h, w = img.shape[:2]

            boundaries = np.array([[0, w - 1], [0, h - 1]], dtype=float)
            boundaries_rgb = IMAGE_BOUNDS
            dist_names = list_distance_functions()
            break
        else :
            print("Opción inválida.")  

    while True:
        # Calcular/mostrar centroides
        for c in clusters.values():
            c.compute_centroid()
        print("\n-------------------------------------------\n\nCentroides calculados:")
        for c in clusters.values():
            print(f"  {c.label} -> {c.centroid}")

        # Elegir función de distancia
        print("\n-------------------------------------------\n\nFunciones de distancia disponibles:")
        for i, name in enumerate(dist_names, start=1):
            print(f"  {i}. {name}")
        choice = input("\nElige una función de distancia [enter para 'euclidean']: ").strip()
        if not choice:
            chosen_name = "euclidean" if "euclidean" in dist_names else dist_names[0]
        else:
            idx = int(choice) - 1
            chosen_name = dist_names[idx] if 0 <= idx < len(dist_names) else dist_names[0]

        print(f"\n>>>>>> Usando distancia: {chosen_name}")

        # Leer punto nuevo
        raw = input("\nIntroduce el nuevo punto (formato: x,y) o 'q' para salir: ").strip()
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

        # coords = (x, y)
        x, y = coords

        r, g, b = get_rgb_at(img, x, y)   # usa la función helper que te pasé
        new_point = (float(r), float(g), float(b))
        new_point_plot = (x, y)  

        clf = CentroidClassifier(distance=getattr(dist, chosen_name))
        clf.fit_from_clusters(clusters.values())
        label = clf.predict_point(new_point)

        clf1 = CentroidClassifier(distance=getattr(dist, chosen_name))
        clf1.fit_from_clusters(plot_clusters.values())

        if chosen_name == "probability_gaussian":
            pdfs = np.array([
                dist.probability_gaussian(new_point, c.centroid, np.array(c.representatives))
                for c in clf.clusters
            ], dtype=float)

            s = pdfs.sum()
            if s == 0 or not np.isfinite(s):
                print("No se pudieron normalizar probabilidades (sum=0 o no finita).")
            else:
                probs = pdfs / s
                print("\nProbabilidades normalizadas:")
                for c, p in sorted(zip(clf.clusters, probs), key=lambda t: t[1], reverse=True):
                    print(f"  Clase {c.label}: {p*100:.2f}%")
        
        print(f"Predicción para {coords} -> {label}")

        plot_clusters_and_point(
            plot_clusters,
            new_point=new_point_plot,
            new_label=label,
            title_suffix=f"(dist: {chosen_name})",
            image_path=image_path
        )

        clusters[label].add_representative(new_point)
        plot_clusters[label].add_representative(new_point_plot)

        if input("\n¿Clasificar otro punto? [s/N]: ").strip().lower() not in ("s", "si", "y", "yes"):
            print("\nFin.")
            break


if __name__ == "__main__":
    go_centroid_classifier()