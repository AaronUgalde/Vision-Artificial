"""
Problema 2 - Clasificación por canales de color con CentroidClassifier
=======================================================================
1. Carga una imagen y la muestra dividida en 4 cuadrantes:
   - Superior izquierdo : canal Rojo
   - Superior derecho   : canal Verde
   - Inferior izquierdo : canal Azul
   - Inferior derecho   : escala de grises

2. Toma 150 puntos aleatorios de cada clase y obtiene sus valores RGB
   para formar un Cluster con su centroide.

3. El usuario hace CLICK en la imagen para seleccionar un punto.
   Se obtiene el RGB del pixel y se clasifica con CentroidClassifier.
   Si se clasifica como "Gris" → "no pertenece a ninguna clase".

4. Se muestra un plot RGB 3-D con los representantes de todas las clases,
   los centroides, y el punto clasificado.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # backend interactivo; cambia a "Qt5Agg" si no tienes Tk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Cluster import Cluster
from CentroidClassifier import CentroidClassifier
import dist as dist_module

# ────────────────────────────────────────────────────────────────────────────
# Constantes
# ────────────────────────────────────────────────────────────────────────────
NUM_SAMPLES = 150
LABEL_ROJO  = "Rojo"
LABEL_VERDE = "Verde"
LABEL_AZUL  = "Azul"
LABEL_GRIS  = "Gris"

COLOR_MAP = {
    LABEL_ROJO:  "red",
    LABEL_VERDE: "limegreen",
    LABEL_AZUL:  "dodgerblue",
    LABEL_GRIS:  "gray",
}

# ────────────────────────────────────────────────────────────────────────────
# Helpers de imagen
# ────────────────────────────────────────────────────────────────────────────

def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga la imagen y devuelve array H×W×3 en uint8."""
    img = plt.imread(ruta)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def canal_rojo(img):
    out = np.zeros_like(img); out[:, :, 0] = img[:, :, 0]; return out

def canal_verde(img):
    out = np.zeros_like(img); out[:, :, 1] = img[:, :, 1]; return out

def canal_azul(img):
    out = np.zeros_like(img); out[:, :, 2] = img[:, :, 2]; return out

def escala_grises(img):
    g = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


# ────────────────────────────────────────────────────────────────────────────
# Visualización cuadrantes — una sola imagen compuesta
# ────────────────────────────────────────────────────────────────────────────

# Desplazamiento (dx, dy) de cada cuadrante dentro de la imagen compuesta
# layout:  [Rojo  | Verde]
#          [Azul  | Gris ]
def _offsets(h, w):
    return {
        LABEL_ROJO:  (0,  0),
        LABEL_VERDE: (w,  0),
        LABEL_AZUL:  (0,  h),
        LABEL_GRIS:  (w,  h),
    }


def construir_figura_cuadrantes(img: np.ndarray,
                                titulo: str = "Canales de la imagen",
                                clusters: dict | None = None):
    """
    Muestra los 4 canales como UNA SOLA IMAGEN numpy 2×2 (un único imshow).
    Devuelve (fig, ax, imgs_filtradas, h, w) donde h, w son las dimensiones
    de cada cuadrante individual (no de la imagen compuesta).

    Si se pasa `clusters`, dibuja los plot_points encima, desplazados al
    cuadrante correspondiente.
    """
    h, w = img.shape[:2]

    imgs_filtradas = {
        LABEL_ROJO:  canal_rojo(img),
        LABEL_VERDE: canal_verde(img),
        LABEL_AZUL:  canal_azul(img),
        LABEL_GRIS:  escala_grises(img),
    }

    # Construir imagen compuesta: fila_top = [Rojo | Verde], fila_bot = [Azul | Gris]
    fila_top = np.concatenate([imgs_filtradas[LABEL_ROJO],
                                imgs_filtradas[LABEL_VERDE]], axis=1)
    fila_bot = np.concatenate([imgs_filtradas[LABEL_AZUL],
                                imgs_filtradas[LABEL_GRIS]],  axis=1)
    img_compuesta = np.concatenate([fila_top, fila_bot], axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    ax.imshow(img_compuesta)
    ax.axis("off")

    # Representantes de cada cluster desplazados a su cuadrante
    if clusters:
        offs = _offsets(h, w)
        for label, cluster in clusters.items():
            if not cluster.plot_points:
                continue
            dx, dy = offs[label]
            pts = np.array(cluster.plot_points, dtype=float)
            ax.scatter(pts[:, 0] + dx, pts[:, 1] + dy,
                       c=COLOR_MAP[label], s=6, alpha=0.55,
                       linewidths=0, zorder=3)

    plt.tight_layout()
    return fig, ax, imgs_filtradas, h, w


# ────────────────────────────────────────────────────────────────────────────
# Muestreo y construcción de clusters
# ────────────────────────────────────────────────────────────────────────────

def muestrear_cluster(img_filtrada: np.ndarray, label: str,
                      n: int = NUM_SAMPLES, seed=None) -> Cluster:
    """
    Toma n píxeles aleatorios de img_filtrada (ya filtrada por canal).
    Los representantes tienen exactamente los valores del canal filtrado,
    p.ej. Rojo → [R, 0, 0], Verde → [0, G, 0], Azul → [0, 0, B], Gris → [Y, Y, Y].
    """
    rng = np.random.default_rng(seed)
    h, w = img_filtrada.shape[:2]
    indices = rng.choice(h * w, size=n, replace=False)
    ys, xs = np.unravel_index(indices, (h, w))
    cluster = Cluster(label=label)
    for y, x in zip(ys, xs):
        rgb = img_filtrada[y, x].astype(float)       # ya filtrado: [R,0,0] / [0,G,0] / etc.
        pos = np.array([float(x), float(y)])
        cluster.add_representative(rgb, plot_point=pos)
    return cluster


def construir_clusters(img: np.ndarray, n: int = NUM_SAMPLES) -> dict:
    """Cada clase muestrea desde su imagen de cuadrante filtrada."""
    return {
        LABEL_ROJO:  muestrear_cluster(canal_rojo(img),    LABEL_ROJO,  n, seed=0),
        LABEL_VERDE: muestrear_cluster(canal_verde(img),   LABEL_VERDE, n, seed=1),
        LABEL_AZUL:  muestrear_cluster(canal_azul(img),    LABEL_AZUL,  n, seed=2),
        LABEL_GRIS:  muestrear_cluster(escala_grises(img), LABEL_GRIS,  n, seed=3),
    }


# ────────────────────────────────────────────────────────────────────────────
# Selección interactiva de punto en la imagen (click del mouse)
# ────────────────────────────────────────────────────────────────────────────

def seleccionar_punto_en_imagen(img: np.ndarray,
                                clusters: dict | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Abre la imagen compuesta (4 cuadrantes en un solo imshow) y espera un click.
    Detecta en qué cuadrante cayó según (x < w, y < h), lee el RGB de la
    imagen filtrada correspondiente y lo devuelve.
    Devuelve (rgb_filtrado [3,], pos_en_cuadrante [2,]) o None si se cierra.
    """
    resultado = {"rgb": None, "pos": None}

    fig, ax, imgs_filtradas, h, w = construir_figura_cuadrantes(
        img,
        titulo="Haz CLICK en cualquier cuadrante para clasificar  (cierra para salir)",
        clusters=clusters,
    )

    crosshair, = ax.plot([], [], "+", color="yellow",
                         markersize=18, markeredgewidth=2, zorder=10)
    info_text = ax.text(0.01, 0.01, "", transform=ax.transAxes,
                        color="white", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="black", alpha=0.65))

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return

        cx = int(np.clip(round(event.xdata), 0, 2*w - 1))
        cy = int(np.clip(round(event.ydata), 0, 2*h - 1))

        # determinar cuadrante y offset
        if   cx < w and cy < h:
            label, dx, dy = LABEL_ROJO,  0, 0
        elif cx >= w and cy < h:
            label, dx, dy = LABEL_VERDE, w, 0
        elif cx < w and cy >= h:
            label, dx, dy = LABEL_AZUL,  0, h
        else:
            label, dx, dy = LABEL_GRIS,  w, h

        # coordenada local dentro del cuadrante
        lx = cx - dx
        ly = cy - dy
        rgb = imgs_filtradas[label][ly, lx].astype(float)

        resultado["rgb"] = rgb
        resultado["pos"] = np.array([float(lx), float(ly)])

        crosshair.set_data([cx], [cy])
        info_text.set_text(
            f"{label}  ({lx},{ly})  R={rgb[0]:.0f} G={rgb[1]:.0f} B={rgb[2]:.0f}"
        )
        fig.canvas.draw_idle()
        plt.pause(0.8)
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)

    if resultado["rgb"] is None:
        return None
    return resultado["rgb"], resultado["pos"]


# ────────────────────────────────────────────────────────────────────────────
# Plot 3-D: representantes + centroides + punto clasificado
# ────────────────────────────────────────────────────────────────────────────

def mostrar_resultado_3d(clusters: dict,
                         nuevo_rgb: np.ndarray,
                         etiqueta_pred: str,
                         pos_imagen: np.ndarray | None = None):
    """
    Grafica en espacio RGB:
      - Representantes de cada clase (puntos pequeños, alpha bajo)
      - Centroide de cada clase  (marca X grande)
      - Nuevo punto clasificado  (estrella dorada)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    resultado_str = (
        f"No pertenece a ninguna clase  (clasificado como {etiqueta_pred})"
        if etiqueta_pred == LABEL_GRIS
        else f"Clase predicha: {etiqueta_pred}"
    )
    ax.set_title(f"Clusters RGB — {resultado_str}", fontsize=11, fontweight="bold")

    legend_handles = []

    for label, cluster in clusters.items():
        color = COLOR_MAP[label]
        reps  = np.array(cluster.representatives, dtype=float)   # (N, 3)

        # ── representantes ──────────────────────────────────────────────
        ax.scatter(reps[:, 0], reps[:, 1], reps[:, 2],
                   c=color, alpha=0.25, s=15, depthshade=True)

        # ── centroide ───────────────────────────────────────────────────
        c = cluster.centroid
        ax.scatter(c[0], c[1], c[2],
                   c=color, marker="X", s=220, edgecolors="black",
                   linewidths=0.8, zorder=6, depthshade=False)
        ax.text(c[0], c[1], c[2],
                f"  {label}\n  ({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})",
                fontsize=7, color=color, fontweight="bold")

        legend_handles.append(
            mpatches.Patch(color=color,
                           label=f"{label}  centroide=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})")
        )

    # ── nuevo punto ──────────────────────────────────────────────────────
    star_color = COLOR_MAP.get(etiqueta_pred, "black")
    ax.scatter(nuevo_rgb[0], nuevo_rgb[1], nuevo_rgb[2],
               c="gold", marker="*", s=350, edgecolors=star_color,
               linewidths=1.2, zorder=10, depthshade=False,
               label=f"Nuevo  RGB=({nuevo_rgb[0]:.0f},{nuevo_rgb[1]:.0f},{nuevo_rgb[2]:.0f})")
    legend_handles.append(
        mpatches.Patch(facecolor="gold", edgecolor=star_color,
                       label=f"★ Punto  RGB=({nuevo_rgb[0]:.0f},{nuevo_rgb[1]:.0f},{nuevo_rgb[2]:.0f})")
    )

    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)


# ────────────────────────────────────────────────────────────────────────────
# Bucle principal de clasificación con click
# ────────────────────────────────────────────────────────────────────────────

def clasificar_interactivo(img: np.ndarray, clusters: dict):
    """
    Ciclo principal:
      1. Muestra la imagen → usuario hace click → se obtiene RGB
      2. Clasifica con CentroidClassifier
      3. Imprime resultado y muestra plot 3-D con representantes
      4. Pregunta si clasificar otro punto
    """
    clf = CentroidClassifier(distance=dist_module.euclidean)
    clf.fit_from_clusters(list(clusters.values()))

    print("\n" + "═" * 58)
    print("  CLASIFICADOR POR CLICK  (CentroidClassifier)")
    print("═" * 58)
    print("  Haz click en la imagen para seleccionar un pixel.")
    print("  Cierra la ventana sin hacer click para terminar.")
    print("─" * 58)

    while True:
        res = seleccionar_punto_en_imagen(img, clusters=clusters)
        if res is None:
            print("\nNo se seleccionó ningún punto. Fin.")
            break

        rgb, pos = res
        print(f"\n  Cuadrante clickeado : (el RGB ya corresponde al canal filtrado)")
        print(f"  Pixel seleccionado  : ({pos[0]:.0f}, {pos[1]:.0f})")
        print(f"  Valor RGB filtrado  : R={rgb[0]:.0f}  G={rgb[1]:.0f}  B={rgb[2]:.0f}")

        try:
            etiqueta = clf.predict_point(rgb)
        except Exception as e:
            print(f"  ✗ Error al clasificar: {e}")
            continue

        # ── distancias informativas ──────────────────────────────────────
        print("  Distancias a centroides:")
        for c in clf.clusters:
            d = dist_module.euclidean(rgb, c.centroid, None)
            marker = " ◄" if c.label == etiqueta else ""
            print(f"    • {c.label:<8}: {d:.2f}{marker}")

        # ── veredicto ────────────────────────────────────────────────────
        if etiqueta == LABEL_GRIS:
            print(f"\n  ➜  El vector NO pertenece a ninguna clase definida.")
        else:
            print(f"\n  ➜  Clase predicha: {etiqueta}")
        print("─" * 58)

        # ── plot 3-D ─────────────────────────────────────────────────────
        mostrar_resultado_3d(clusters, rgb, etiqueta, pos)

        resp = input("\n  ¿Clasificar otro punto? [s/N]: ").strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print("Fin del clasificador.")
            break


# ────────────────────────────────────────────────────────────────────────────
# Función principal
# ────────────────────────────────────────────────────────────────────────────

def main():
    ruta_default = "data_info/img/image.png"
    ruta = input(f"Ruta de la imagen [Enter para '{ruta_default}']: ").strip() or ruta_default

    try:
        img = cargar_imagen(ruta)
    except Exception as e:
        print(f"No se pudo cargar la imagen: {e}")
        sys.exit(1)

    print(f"\nImagen cargada: {img.shape[1]}×{img.shape[0]} px")

    # 2. Construir clusters (150 puntos por clase)
    print(f"\nMuestreando {NUM_SAMPLES} puntos aleatorios por clase...")
    clusters = construir_clusters(img, n=NUM_SAMPLES)

    print("\nCentroides (RGB):")
    for label, cluster in clusters.items():
        c = cluster.centroid
        print(f"  {label:<8}: R={c[0]:.1f}  G={c[1]:.1f}  B={c[2]:.1f}")

    # 3. Clasificar por click
    clasificar_interactivo(img, clusters)

    plt.show()


if __name__ == "__main__":
    main()
