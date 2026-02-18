import math
from typing import List

import numpy as np

from Point import Point
from utilis import calculate_covariance_matrix


def _validate_dimensions(a: Point, b: Point) -> None:
    if len(a) != len(b):
        raise ValueError("Dimensiones diferentes al calcular distancia")


def euclidean(a: Point, b: Point, points: List[Point]) -> float:
    _validate_dimensions(a, b)
    return float(np.linalg.norm(np.array(a.coords) - np.array(b.coords)))


def manhattan(a: Point, b: Point, points: List[Point]) -> float:
    _validate_dimensions(a, b)
    return float(np.abs(np.array(a.coords) - np.array(b.coords)).sum())


def mahalanobis_distance_matrix(x: Point, mu: Point, points: List[Point]) -> float:
    """
    d = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
    """
    _validate_dimensions(x, mu)

    if not points:
        raise ValueError("Se requieren puntos de referencia para Mahalanobis")

    cov_matrix = calculate_covariance_matrix(points)
    diff = x - mu

    try:
        cov_inv = cov_matrix.inverse()
    except np.linalg.LinAlgError:
        cov_inv = cov_matrix.copy()
        cov_inv._data = np.linalg.pinv(cov_inv._data)

    # Sigma^{-1} * (x - mu)T
    temp = cov_inv * diff

    # (x - mu) * temp
    distance_sq = sum(d * t for d, t in zip(diff, temp))

    return math.sqrt(max(distance_sq, 0.0))
