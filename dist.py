from Point import Point
import numpy as np

def euclidean(a: Point, b: Point) -> float:
    if len(a) != len(b):
        raise ValueError("Dimensiones diferentes al calcular euclidiana")
    return float(np.linalg.norm(np.array(a.coords) - np.array(b.coords)))


def manhattan(a: Point, b: Point) -> float:
    if len(a) != len(b):
        raise ValueError("Dimensiones diferentes al calcular manhattan")
    return float(np.abs(np.array(a.coords) - np.array(b.coords)).sum())