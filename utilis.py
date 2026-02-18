from typing import List
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
    
    n_samples = len(points)
    n_features = len(points[0])
    
    # 1. Calcular el punto medio (Vector de Medias - mu)
    # Sumamos todos los puntos y escalamos por 1/N
    sum_point = points[0]
    for i in range(1, n_samples):
        sum_point += points[i]
    
    mean_point = sum_point.scale(1.0 / n_samples)
    
    # 2. Crear la matriz de datos centrada (X_centered)
    # Restamos la media a cada punto
    centered_data = []
    for p in points:
        diff = p - mean_point
        centered_data.append(diff.to_list())
    
    # Convertimos a Matrix para usar las bondades de NumPy internamente
    X = Matrix(centered_data)
    
    # 3. Calcular la covarianza: (X^T * X) / (N - 1)
    # La fórmula es: Cov = (1 / df) * (X.T @ X)
    xt_x = X.T() * X
    
    divisor = (n_samples - 1) if unbiased and n_samples > 1 else n_samples
    covariance_matrix = xt_x * (1.0 / divisor)
    
    return covariance_matrix