import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Cluster:
    label: str
    representatives: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None

    def compute_centroid(self) -> np.ndarray:
        if not self.representatives:
            raise ValueError("No hay representantes para calcular el centroid")
        self.centroid = np.mean(self.representatives, axis=0)
        return self.centroid

    def add_representative(self, p: np.ndarray) -> None:
        if self.representatives and p.shape != self.representatives[0].shape:
            raise ValueError("Dimensión distinta en representante añadido")
        self.representatives.append(p)
        self.compute_centroid()

    def __repr__(self) -> str:
        return f"Cluster(label={self.label!r}, centroid={self.centroid})"
