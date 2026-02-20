from typing import List
import numpy as np
from Matrix import Matrix
from Point import Point

def calculate_covariance_matrix(points: List[Point], unbiased: bool = True) -> Matrix:
    """
    Calcula la matriz de covarianza a partir de una lista de objetos Point.

    Args:
        points: Lista de objetos Point (nuestras observaciones).
        unbiased: Si es True, divide por (N-1). Si es False, divide por N.
    """
    if not points:
        raise ValueError("La lista de puntos no puede estar vacía.")

    arr = np.array([p.coords for p in points], dtype=float)  # shape (N, D)
    ddof = 1 if unbiased and len(points) > 1 else 0
    cov = np.cov(arr, rowvar=False, ddof=ddof)

    # np.cov devuelve un escalar si hay una sola feature; lo convertimos a 2D
    return Matrix(np.atleast_2d(cov))
