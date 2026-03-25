import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from Cluster import Cluster
from CentroidClassifier import CentroidClassifier
import dist


N_REPRESENTANTES = 150

def cargar_imagen(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def construir_canales(img_rgb):
    r = np.zeros_like(img_rgb)
    g = np.zeros_like(img_rgb)
    b = np.zeros_like(img_rgb)

    r[:, :, 0] = img_rgb[:, :, 0]
    g[:, :, 1] = img_rgb[:, :, 1]
    b[:, :, 2] = img_rgb[:, :, 2]

    gris = np.mean(img_rgb, axis=2).astype(np.uint8)
    gris_rgb = np.stack([gris, gris, gris], axis=2)

    return r, g, b, gris_rgb


def samplear_puntos_en_quadrante(h, w, n, quadrant):
    """
    Devuelve n puntos 2D (x,y) dentro del rectángulo del cuadrante.
    Sistema:
      QI   -> x:[0,w],   y:[0,h]     (verde)
      QII  -> x:[-w,0],  y:[0,h]     (rojo)
      QIII -> x:[-w,0],  y:[-h,0]    (azul)
      QIV  -> x:[0,w],   y:[-h,0]    (gris)
    """
    puntos = []

    for _ in range(n):
        px = random.uniform(0, w)
        py = random.uniform(0, h)

        if quadrant == 1:      # verde
            x = px
            y = py
        elif quadrant == 2:    # rojo
            x = px - w
            y = py
        elif quadrant == 3:    # azul
            x = px - w
            y = py - h
        elif quadrant == 4:    # gris
            x = px
            y = py - h
        else:
            raise ValueError("Quadrant inválido")

        puntos.append(np.array([x, y], dtype=float))

    return puntos


def construir_clusters_desde_imagen(img_rgb, n_rep=150):
    h, w, _ = img_rgb.shape
    img_r, img_g, img_b, img_gray = construir_canales(img_rgb)

    # Clases por cuadrante:
    # QI  -> verde  -> clase2
    # QII -> rojo   -> clase1
    # QIII-> azul   -> clase3
    # QIV -> gris   -> clase4 (rechazo)
    clusters = {
        "clase1": Cluster("clase1", samplear_puntos_en_quadrante(h, w, n_rep, 2)),
        "clase2": Cluster("clase2", samplear_puntos_en_quadrante(h, w, n_rep, 1)),
        "clase3": Cluster("clase3", samplear_puntos_en_quadrante(h, w, n_rep, 3)),
        "clase4": Cluster("clase4", samplear_puntos_en_quadrante(h, w, n_rep, 4)),
    }

    for c in clusters.values():
        c.compute_centroid()

    imagenes = {
        "rojo": img_r,
        "verde": img_g,
        "azul": img_b,
        "gris": img_gray,
    }

    return clusters, imagenes, h, w


def clasificar_punto(clf, p):
    etiqueta = clf.predict_point(np.array(p, dtype=float))
    if etiqueta == "clase4":
        return "ninguna clase"
    return etiqueta


def graficar_espacio(imagenes, clusters, h, w, punto=None, etiqueta=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Extents para cuadrantes
    # rojo   -> QII
    ax.imshow(imagenes["rojo"],  extent=[-w, 0, 0, h], origin="lower")
    # verde  -> QI
    ax.imshow(imagenes["verde"], extent=[0, w, 0, h], origin="lower")
    # azul   -> QIII
    ax.imshow(imagenes["azul"],  extent=[-w, 0, -h, 0], origin="lower")
    # gris   -> QIV
    ax.imshow(imagenes["gris"],  extent=[0, w, -h, 0], origin="lower")

    # Ejes cartesianos
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    # Representantes
    for label, cluster in clusters.items():
        arr = np.array(cluster.representatives)
        ax.scatter(arr[:, 0], arr[:, 1], s=12, alpha=0.45, label=f"{label}")

        cx, cy = cluster.centroid
        ax.scatter(cx, cy, marker="X", s=180, edgecolor="black")
        ax.text(cx, cy, f" {label}", fontsize=10, weight="bold")

    if punto is not None:
        x, y = punto
        ax.scatter(x, y, marker="*", s=260, color="white", edgecolor="black", zorder=20)
        ax.text(x, y, f" {etiqueta}", fontsize=11, weight="bold", color="white")

    ax.set_xlim(-w, w)
    ax.set_ylim(-h, h)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Clasificación por cuadrantes: verde(QI), rojo(QII), azul(QIII), gris(QIV)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def pedir_punto():
    while True:
        raw = input("Ingresa el punto x,y (o 'q' para salir): ").strip()
        if raw.lower() in ("q", "quit", "salir", "exit"):
            return None
        try:
            x_str, y_str = raw.split(",")
            return float(x_str), float(y_str)
        except ValueError:
            print("Formato inválido. Usa: x,y")


def main():
    path = input("Ruta de la imagen: ").strip()
    if not path:
        print("Debes indicar una ruta.")
        return

    img_rgb = cargar_imagen(path)
    clusters, imagenes, h, w = construir_clusters_desde_imagen(img_rgb, n_rep=N_REPRESENTANTES)

    clf = CentroidClassifier(distance=dist.euclidean)
    clf.boundaries = np.array([[-w, w], [-h, h]], dtype=float)
    clf.fit_from_clusters(clusters.values())

    while True:
        punto = pedir_punto()
        if punto is None:
            print("Saliendo.")
            break

        try:
            etiqueta = clasificar_punto(clf, punto)
            print(f"El punto {punto} pertenece a: {etiqueta}")
            graficar_espacio(imagenes, clusters, h, w, punto=punto, etiqueta=etiqueta)
        except ValueError as e:
            print("Error:", e)


if __name__ == "__main__":
    main()