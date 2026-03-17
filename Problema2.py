import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from collections import defaultdict

from Cluster import Cluster
from CentroidClassifier import CentroidClassifier
import dist


# ── Imagen con las 4 letras ──────────────────────────────────────────────────

IMG_W, IMG_H = 600, 200

def build_letter_image():
    img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 255  # fondo blanco
    fig, ax = plt.subplots(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    ax.set_xlim(0, IMG_W); ax.set_ylim(0, IMG_H)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fp = FontProperties(family="DejaVu Sans", weight="bold")
    letters = [
        ("U", IMG_W * 0.12, IMG_H * 0.5, "green"),
        ("T", IMG_W * 0.37, IMG_H * 0.5, "blue"),
        ("A", IMG_W * 0.63, IMG_H * 0.5, "red"),
        ("P", IMG_W * 0.88, IMG_H * 0.5, "black"),
    ]
    for ch, x, y, color in letters:
        ax.text(x, y, ch, fontsize=110, fontproperties=fp,
                color=color, ha="center", va="center")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return arr[:, :, :3].copy()   # RGB sin canal alpha


# ── Muestreo de representantes ───────────────────────────────────────────────

# Rangos de color aproximados para cada letra (en RGB 0-255)
COLOR_MASKS = {
    "U": lambda r, g, b: (g > 100) & (r < 150) & (b < 150),      # verde
    "T": lambda r, g, b: (b > 100) & (r < 150) & (g < 150),      # azul
    "A": lambda r, g, b: (r > 150) & (g < 100) & (b < 100),      # rojo
}


def sample_representatives(img, label, n=150, seed=42):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = COLOR_MASKS[label](r, g, b)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError(f"No se encontraron pixeles para la clase '{label}'")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    samples = []
    for i in idx:
        px, py = xs[i], ys[i]
        rgb_norm = img[py, px, :3].astype(float) / 255.0
        samples.append({"rgb": rgb_norm, "position": np.array([px, py], dtype=float)})
    return samples


def build_clusters(img):
    clusters = {}
    for label in ("U", "T", "A"):
        c = Cluster(label=label)
        for s in sample_representatives(img, label):
            c.add_representative(s["rgb"], plot_point=s["position"])
        clusters[label] = c
    return clusters


# ── Visualizacion ────────────────────────────────────────────────────────────

LABEL_COLORS = {"U": "green", "T": "blue", "A": "red"}


def plot_image_with_point(img, new_point=None, new_label=None, clusters=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- panel izq: imagen con representantes y punto nuevo
    ax = axes[0]
    ax.imshow(img)
    if clusters:
        for label, c in clusters.items():
            if c.plot_points:
                pts = np.array(c.plot_points)
                ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.4,
                           color=LABEL_COLORS[label], label=f"Reps {label}")
            if c.centroid is not None and c.plot_points:
                pc = np.mean(c.plot_points, axis=0)
                ax.scatter(pc[0], pc[1], marker="X", s=180, edgecolor="k",
                           color=LABEL_COLORS[label], zorder=5)
    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], marker="*", s=350,
                   color="magenta", edgecolor="k", zorder=10,
                   label=f"Nuevo → {new_label}")
    ax.set_title("Imagen con representantes y centroides")
    ax.legend(loc="upper right", fontsize=7, markerscale=1)
    ax.axis("off")

    # -- panel der: cubo RGB con centroides
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    if clusters:
        for label, c in clusters.items():
            reps = np.array(c.representatives)
            ax3.scatter(reps[:, 0], reps[:, 1], reps[:, 2], s=10,
                        alpha=0.3, color=LABEL_COLORS[label])
            cent = c.centroid
            ax3.scatter(*cent, marker="X", s=200, edgecolor="k",
                        color=LABEL_COLORS[label], zorder=5)
            ax3.text(cent[0], cent[1], cent[2], f"  {label}", fontsize=8)
    if new_point is not None:
        rgb = img[int(new_point[1]), int(new_point[0]), :3].astype(float) / 255.0
        ax3.scatter(*rgb, marker="*", s=350, color="magenta", edgecolor="k",
                    zorder=10, label=f"Nuevo → {new_label}")
    ax3.set_xlabel("R"); ax3.set_ylabel("G"); ax3.set_zlabel("B")
    ax3.set_title("Espacio RGB")
    if new_point is not None:
        ax3.legend(loc="upper left", fontsize=7, markerscale=0.5)
    plt.tight_layout()
    plt.show()


# ── Distancias disponibles ───────────────────────────────────────────────────

def list_distance_functions():
    import inspect
    names = [
        attr for attr in dir(dist)
        if not attr.startswith("_")
        and inspect.isfunction(getattr(dist, attr))
        and getattr(dist, attr).__module__ == dist.__name__
    ]
    return sorted(names, key=lambda x: (0 if x == "euclidean" else 1, x))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Problema 2: Clasificacion por color de letra ===\n")
    print("Generando imagen con letras U (verde), T (azul), A (rojo), P (negro)...")
    img = build_letter_image()

    print("Muestreando 150 representantes por clase (U, T, A)...")
    clusters = build_clusters(img)

    print("\nCentroides RGB (normalizados 0-1):")
    for label, c in clusters.items():
        print(f"  {label}: {np.round(c.centroid, 4).tolist()}")

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

    h, w = img.shape[:2]
    plot_image_with_point(img, clusters=clusters)

    while True:
        print(f"\nIngresa un punto en la imagen (x: 0-{w-1}, y: 0-{h-1})")
        raw = input("Punto (x,y): ").strip()
        if raw.lower() in ("q", "quit", "salir"):
            print("Fin."); break
        try:
            parts = [float(v) for v in raw.replace(",", " ").split() if v]
            if len(parts) != 2:
                raise ValueError("Se necesitan exactamente 2 valores.")
            px, py = int(round(parts[0])), int(round(parts[1]))
            if not (0 <= px < w and 0 <= py < h):
                print(f"Fuera de rango. x: 0-{w-1}, y: 0-{h-1}")
                continue
        except ValueError as e:
            print("Entrada invalida:", e); continue

        rgb_norm = img[py, px, :3].astype(float) / 255.0
        print(f"Color RGB del pixel ({px},{py}): {np.round(rgb_norm, 4).tolist()}")

        # Detectar fondo (blanco) o letra P (negro) — no pertenecen a ninguna clase
        r, g, b = rgb_norm
        es_blanco = r > 0.85 and g > 0.85 and b > 0.85
        es_negro  = r < 0.15 and g < 0.15 and b < 0.15
        if es_blanco or es_negro:
            razon = "fondo (blanco)" if es_blanco else "letra P (negro)"
            print(f">>> El punto pertenece al {razon} — no pertenece a ninguna clase.\n")
            plot_image_with_point(img, new_point=np.array([px, py]), new_label="Ninguna", clusters=clusters)
        else:
            clf = CentroidClassifier(distance=getattr(dist, chosen))
            clf.fit_from_clusters(clusters.values())
            label = clf.predict_point(rgb_norm)
            print(f">>> Clasificado como: {label}\n")
            plot_image_with_point(img, new_point=np.array([px, py]), new_label=label, clusters=clusters)
        plot_image_with_point(img, new_point=np.array([px, py]), new_label=label, clusters=clusters)

        if input("Clasificar otro punto? [s/N]: ").strip().lower() not in ("s", "si", "y", "yes"):
            print("Fin."); break


if __name__ == "__main__":
    main()
