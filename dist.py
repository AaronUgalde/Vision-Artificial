import numpy as np
from typing import List
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
    return float(np.sum(np.abs(np.array(a.coords) - np.array(b.coords))))


def mahalanobis_distance_matrix(x: Point, mu: Point, points: List[Point]) -> float:
    """
    d = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
    """
    _validate_dimensions(x, mu)
    if not points:
        raise ValueError("Se requieren puntos de referencia para Mahalanobis")

    cov_matrix = calculate_covariance_matrix(points)
    diff = np.array(x.coords) - np.array(mu.coords)

    try:
        cov_inv = np.linalg.inv(cov_matrix._data)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_matrix._data)

    distance_sq = diff @ cov_inv @ diff
    return float(np.sqrt(max(distance_sq, 0.0)))
