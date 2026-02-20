from dataclasses import field, dataclass
from typing import Tuple, Iterable, List
import numpy as np

@dataclass(frozen=True)
class Point:
    coords: Tuple[float, ...] = field(default_factory=tuple)

    @classmethod
    def from_iterable(cls, it: Iterable[float]) -> "Point":
        return cls(tuple(float(x) for x in it))

    def __len__(self) -> int:
        return len(self.coords)

    def __iter__(self):
        return iter(self.coords)

    def __add__(self, other: "Point") -> "Point":
        return Point.from_iterable(np.array(self.coords) + np.array(other.coords))

    def __sub__(self, other: "Point") -> "Point":
        return Point.from_iterable(np.array(self.coords) - np.array(other.coords))

    def scale(self, scalar: float) -> "Point":
        return Point.from_iterable(np.array(self.coords) * scalar)

    def to_list(self) -> List[float]:
        return list(self.coords)

    def __repr__(self) -> str:
        return f"Point({', '.join(f'{c:.4g}' for c in self.coords)})"
