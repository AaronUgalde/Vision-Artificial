import numpy as np
from Cluster import Cluster
from typing import List, Callable, Sequence, Tuple
from dist import euclidean


class CentroidClassifier:
    def __init__(self, distance: Callable = euclidean):
        self.distance = distance
        self.clusters: List[Cluster] = []
        self.boundaries: np.ndarray = None  # shape (D, 2) -> [[min, max], ...]

    def _check_boundaries(self, p: np.ndarray) -> None:
        if self.boundaries is not None:
            if np.any(p < self.boundaries[:, 0]) or np.any(p > self.boundaries[:, 1]):
                raise ValueError("El punto está fuera de los límites establecidos")

    def fit_from_clusters(self, clusters: Sequence[Cluster]) -> None:
        self.clusters = []
        for c in clusters:
            if c.centroid is None:
                c.compute_centroid()
            self.clusters.append(c)
        if not self.clusters:
            raise ValueError("Se debe proporcionar al menos un cluster")

    def predict_point(self, p: np.ndarray) -> str:
        if not self.clusters:
            raise RuntimeError("El clasificador no está entrenado. Llama a fit_from_clusters() primero.")
        self._check_boundaries(p)

        reps_arrays = [np.array(c.representatives) for c in self.clusters]
        distances = [self.distance(p, c.centroid, reps) for c, reps in zip(self.clusters, reps_arrays)]
        return self.clusters[int(np.argmin(distances))].label
