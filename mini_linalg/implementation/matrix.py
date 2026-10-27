"""A rectangular matrix of numeric values."""

from numbers import Number


class Matrix:
    def __init__(self, rows):
        self._data = tuple(tuple(row) for row in rows)
        columns = len(self._data[0]) if self._data else 0
        if any(len(row) != columns for row in self._data):
            raise ValueError("Matrix rows must have equal lengths")
        if any(not isinstance(value, Number) for row in self._data for value in row):
            raise TypeError("Matrix entries must be numeric")

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return (len(self), len(self._data[0]) if self._data else 0)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, column = index
            return self._data[row][column]
        return self._data[index]

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.data == other.data

    def __repr__(self):
        return f"Matrix({self.data!r})"

    def __matmul__(self, other):
        from .core import matrix_matrix_product, matrix_vector_product
        from .vector import Vector

        if isinstance(other, Matrix):
            return matrix_matrix_product(self, other)
        if isinstance(other, Vector):
            return matrix_vector_product(self, other)
        return NotImplemented
