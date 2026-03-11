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


def generate_points(rect, n_points):
    x1, y1, x2, y2 = rect
    points = []

    for _ in range(n_points):
        x = random.randint(x1, x2)
        y = random.randint(y1, y2)
        points.append(np.array([float(x), float(y)]))

    return points