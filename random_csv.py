import csv
import random
from typing import List, Tuple


def _generar_puntos_cluster(
    centro_x: float,
    centro_y: float,
    n: int,
    disp_x: float,
    disp_y: float
) -> List[Tuple[float, float]]:
    puntos = []
    for _ in range(n):
        x = random.gauss(centro_x, disp_x)
        y = random.gauss(centro_y, disp_y)
        puntos.append((x, y))
    return puntos


def generar_csv_clusters_por_clase(
    centros_por_clase: List[Tuple[float, float]],
    reps_por_clase: List[int],
    disp_por_clase: List[Tuple[float, float]],
    archivo_csv: str = "reps.csv",
    seed: int | None = None,
) -> None:
    """
    Genera un CSV con columnas: label,x,y.
    - centros_por_clase: lista (cx, cy) por clase.
    - reps_por_clase: lista con # representantes por clase.
    - disp_por_clase: lista (dx, dy) por clase.
    """
    if seed is not None:
        random.seed(seed)

    k = len(centros_por_clase)
    if len(reps_por_clase) != k or len(disp_por_clase) != k:
        raise ValueError("centros_por_clase, reps_por_clase y disp_por_clase deben tener el mismo largo.")

    with open(archivo_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "x", "y"])

        for label in range(k):
            cx, cy = centros_por_clase[label]
            n = reps_por_clase[label]
            dx, dy = disp_por_clase[label]

            puntos = _generar_puntos_cluster(cx, cy, n, dx, dy)
            for (x, y) in puntos:
                writer.writerow([label, x, y])