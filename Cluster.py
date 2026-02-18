import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from Point import Point

@dataclass
class Cluster:
    label: str
    representatives: List[Point] = field(default_factory=list)
    centroid: Optional[Point] = None

    def compute_centroid(self) -> Point:
        if not self.representatives:
            raise ValueError("No hay representantes para calcular el centroid")
        dim = len(self.representatives[0])
        # validar dimensiones
        for p in self.representatives:
            if len(p) != dim:
                raise ValueError("Representantes con diferentes dimensiones")
        
        arr = np.array([p.coords for p in self.representatives], dtype=float)
        mean = arr.mean(axis=0).tolist()
        self.centroid = Point(tuple(mean))
        
        return self.centroid

    def set_centroid(self, point: Point) -> None:
        self.centroid = point

    def add_representative(self, p: Point) -> None:
        if self.representatives and len(p) != len(self.representatives[0]):
            raise ValueError("Dimensión distinta en representante añadido")
        self.representatives.append(p)
        # opcional: actualizar centroid inmediatamente
        self.compute_centroid()

    def __repr__(self) -> str:
        return f"Cluster(label={self.label!r}, centroid={self.centroid})"