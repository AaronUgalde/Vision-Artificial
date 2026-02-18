import numpy as np
from Point import Point
from typing import Union, List
import math

Number = (int, float)

class Matrix:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            arr = data.astype(float)
        else:
            if not data or not isinstance(data, list):
                raise ValueError("data must be a non-empty list of lists or a numpy array")
            # Permitir que la data sea una lista de objetos Point
            processed_data = [list(row) if isinstance(row, Point) else row for row in data]
            row_len = len(processed_data[0])
            for row in processed_data:
                if not isinstance(row, (list, tuple)) or len(row) != row_len:
                    raise ValueError("All rows must have the same length")
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
        # Al pedir una fila (m[i]), devolvemos un Point
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
        # Escalar
        if isinstance(other, Number):
            return Matrix(self._data * float(other))

        # Point o Vector (Retorna Point)
        if isinstance(other, (Point, list, tuple, np.ndarray)):
            vec = np.array(other.coords if isinstance(other, Point) else other, dtype=float)
            if vec.ndim != 1: raise ValueError("Vector must be 1D")
            if vec.shape[0] != self.shape[1]: raise ValueError("Dimension mismatch")
            
            res = self._data.dot(vec)
            return Point.from_iterable(res)

        # Matrix * Matrix
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]: raise ValueError("Inner dimensions mismatch")
            return Matrix(self._data.dot(other._data))

        raise TypeError(f"Unsupported multiplication with {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def T(self):
        return Matrix(self._data.T)

    def inverse(self):
        if self.shape[0] != self.shape[1]: raise ValueError("Matrix must be square")
        return Matrix(np.linalg.inv(self._data))

    def mv(self, vector: Union[Point, List[float]]) -> Point:
        """Matrix-Vector multiplication explicitly returning a Point."""
        return self * vector

