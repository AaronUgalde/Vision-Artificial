import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Cluster:
    label: str
    representatives: List[np.ndarray] = field(default_factory=list)
    plot_points: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None

    def compute_centroid(self) -> np.ndarray:
        if not self.representatives:
            raise ValueError("No hay representantes para calcular el centroid")
        self.centroid = np.mean(self.representatives, axis=0)
        return self.centroid

    def compute_plot_centroid(self) -> Optional[np.ndarray]:
        if not self.plot_points:
            return None
        return np.mean(self.plot_points, axis=0)

    def add_representative(self, p: np.ndarray, plot_point: Optional[np.ndarray] = None) -> None:
        p = np.asarray(p, dtype=float)
        if self.representatives and p.shape != self.representatives[0].shape:
            raise ValueError("Dimension distinta en representante anadido")

        self.representatives.append(p)

        if plot_point is not None:
            plot_point = np.asarray(plot_point, dtype=float)
            if plot_point.shape[0] < 2:
                raise ValueError("El punto para graficar debe tener al menos dos coordenadas")
            self.plot_points.append(plot_point)

        self.compute_centroid()

    def __repr__(self) -> str:
        return f"Cluster(label={self.label!r}, centroid={self.centroid})"
