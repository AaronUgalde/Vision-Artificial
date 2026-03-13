import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def select_rectangles(image_path):
    img = plt.imread(image_path)

    rectangles = []
    start_point = [None, None]
    is_drawing = [False]
    preview_patch = [None]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
    ax.set_title(
        "Selecciona clases con rectángulos\n"
        "Click y arrastra para dibujar | Enter = terminar | r = reiniciar"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    def redraw():
        ax.clear()
        ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
        ax.set_title(
            "Selecciona clases con rectángulos\n"
            "Click y arrastra para dibujar | Enter = terminar | r = reiniciar"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

        for i, (x1, y1, x2, y2) in enumerate(rectangles, start=1):
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 5), f"C{i}", fontsize=10, fontweight="bold")

        if preview_patch[0] is not None:
            ax.add_patch(preview_patch[0])

        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        is_drawing[0] = True
        start_point[0] = int(round(event.xdata))
        start_point[1] = int(round(event.ydata))

    def on_move(event):
        if not is_drawing[0]:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        x0, y0 = start_point
        x1 = int(round(event.xdata))
        y1 = int(round(event.ydata))

        xa, xb = sorted([x0, x1])
        ya, yb = sorted([y0, y1])

        preview_patch[0] = Rectangle(
            (xa, ya),
            xb - xa,
            yb - ya,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
            linestyle="--"
        )
        redraw()

    def on_release(event):
        if not is_drawing[0]:
            return

        is_drawing[0] = False

        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            preview_patch[0] = None
            redraw()
            return

        x0, y0 = start_point
        x1 = int(round(event.xdata))
        y1 = int(round(event.ydata))

        xa, xb = sorted([x0, x1])
        ya, yb = sorted([y0, y1])

        if (xb - xa) >= 2 and (yb - ya) >= 2:
            rectangles.append((xa, ya, xb, yb))

        preview_patch[0] = None
        redraw()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
        elif event.key == "r":
            rectangles.clear()
            preview_patch[0] = None
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    return rectangles

def get_rgb_at(img, x, y):
    """
    Regresa (r,g,b) en 0-255 (int) para el pixel en (x,y).
    Soporta RGB o RGBA y valores float (0-1) o uint8 (0-255).
    """
    h, w = img.shape[0], img.shape[1]

    # Clamp por si acaso
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    px = img[y, x]  # (y, x) !

    # Si viene grayscale (H,W), conviértelo a "RGB"
    if np.isscalar(px) or (hasattr(px, "shape") and px.shape == ()):
        v = float(px)
        if v <= 1.0:
            v = int(round(v * 255))
        else:
            v = int(round(v))
        return (v, v, v)

    # Si viene RGB/RGBA
    r, g, b = px[:3]

    # Normaliza a 0-255 int
    if np.issubdtype(img.dtype, np.floating):
        r = int(round(float(r) * 255))
        g = int(round(float(g) * 255))
        b = int(round(float(b) * 255))
    else:
        r = int(r); g = int(g); b = int(b)

    return (r, g, b)

def generate_points(rect, n_points, img):
    """
    Devuelve una lista de tuplas (x, y, r, g, b).
    """
    x1, y1, x2, y2 = rect
    points = []

    # Asegura límites válidos (y que randint no truene si x1==x2, etc.)
    xa, xb = sorted([int(x1), int(x2)])
    ya, yb = sorted([int(y1), int(y2)])

    h, w = img.shape[0], img.shape[1]
    xa = max(0, min(xa, w - 1))
    xb = max(0, min(xb, w - 1))
    ya = max(0, min(ya, h - 1))
    yb = max(0, min(yb, h - 1))

    for _ in range(n_points):
        x = random.randint(xa, xb)
        y = random.randint(ya, yb)
        r, g, b = get_rgb_at(img, x, y)
        points.append(np.array([x, y, r, g, b]))

    return points