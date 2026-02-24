import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray, points: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def manhattan(a: np.ndarray, b: np.ndarray, points: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def mahalanobis_distance_matrix(x: np.ndarray, mu: np.ndarray, points: np.ndarray) -> float:
    """
    d = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
    """
    if points is None or len(points) == 0:
        raise ValueError("Se requieren puntos de referencia para Mahalanobis")

    ddof = 1 if len(points) > 1 else 0
    cov = np.cov(points, rowvar=False, ddof=ddof)
    cov = np.atleast_2d(cov)

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    diff = x - mu
    return float(np.sqrt(max(diff @ cov_inv @ diff, 0.0)))
