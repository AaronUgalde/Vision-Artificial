from Point import Point
from Cluster import Cluster
from typing import List, Callable, Sequence, Tuple
from dist import euclidean
import json

class CentroidClassifier:
    def __init__(self,
                 distance: Callable[[Point, Point], float] = euclidean):
        self.distance = distance
        self.clusters: List[Cluster] = []
        self.bundaries: List[Tuple] = []

    def check_bundaries(self, p : Point):
        for mn, mx in self.bundaries:
            for coord in p:
                if coord < mn or coord > mx:
                    raise ValueError("El punto debe estar dentro de los limites establecidos")

    def fit_from_clusters(self, clusters: Sequence[Cluster]) -> None:
        # valida y asegura centroides calculados
        self.clusters = []
        for c in clusters:
            if c.centroid is None:
                c.compute_centroid()
            self.clusters.append(c)
        if not self.clusters:
            raise ValueError("Se debe proporcionar al menos un cluster")

        # validar dimensiones consistentes entre centroides
        dim = len(self.clusters[0].centroid)
        for c in self.clusters:
            if len(c.centroid) != dim:
                raise ValueError("Centroides con diferentes dimensiones")

    def predict_point(self, p: Point) -> str:
        if not self.clusters:
            raise RuntimeError("El clasificador no está entrenado. Llama a fit(...) primero.")
        # validar dimensión
        if len(p) != len(self.clusters[0].centroid):
            raise ValueError("Dimensión del punto no coincide con centroides")
        
        self.check_bundaries(p)
        # calcular distancias
        best_label = None
        best_dist = None
        for c in self.clusters:
            d = self.distance(p, c.centroid)
            if (best_dist is None) or (d < best_dist):
                best_dist = d
                best_label = c.label
        return best_label
