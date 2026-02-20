import numpy as np
from Point import Point
from typing import Union, List

Number = (int, float)

class Matrix:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            arr = data.astype(float)
        else:
            if not data or not isinstance(data, list):
                raise ValueError("data must be a non-empty list of lists or a numpy array")
            processed_data = [list(row) if isinstance(row, Point) else row for row in data]
            arr = np.array(processed_data, dtype=float)

        if arr.ndim != 2:
            raise ValueError("Matrix must be 2-dimensional")
        self._data = arr

    @property
    def shape(self):
        return tuple(self._data.shape)

    def copy(self):
        return Matrix(self._data.copy())

    def __repr__(self):
        rows = ["[" + ", ".join(f"{v:.6g}" for v in r) + "]" for r in self._data.tolist()]
        return "Matrix([\n  " + ",\n  ".join(rows) + "\n])"

    def __getitem__(self, idx) -> Union[Point, np.ndarray]:
        row = self._data[idx]
        if isinstance(idx, int):
            return Point.from_iterable(row)
        return row

    def __eq__(self, other):
        if not isinstance(other, Matrix): return False
        return self.shape == other.shape and np.allclose(self._data, other._data, atol=1e-9)

    def __add__(self, other):
        if not isinstance(other, Matrix): raise TypeError("Can only add Matrix + Matrix")
        return Matrix(self._data + other._data)

    def __sub__(self, other):
        if not isinstance(other, Matrix): raise TypeError("Can only subtract Matrix - Matrix")
        return Matrix(self._data - other._data)

    def __mul__(self, other):
        if isinstance(other, Number):
            return Matrix(self._data * float(other))

        if isinstance(other, (Point, list, tuple, np.ndarray)):
            vec = np.array(other.coords if isinstance(other, Point) else other, dtype=float)
            return Point.from_iterable(self._data @ vec)

        if isinstance(other, Matrix):
            return Matrix(self._data @ other._data)

        raise TypeError(f"Unsupported multiplication with {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def T(self):
        return Matrix(self._data.T)

    def inverse(self):
        if self.shape[0] != self.shape[1]: raise ValueError("Matrix must be square")
        return Matrix(np.linalg.inv(self._data))
