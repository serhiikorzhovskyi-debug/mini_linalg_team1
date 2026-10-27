from __future__ import annotations

from collections.abc import Iterable, Iterator
from numbers import Number

from .vector import Vector


class Matrix:
    def __init__(self, rows: Iterable[Iterable[Number]]) -> None:
        data = tuple(tuple(row) for row in rows)
        columns = len(data[0]) if data else 0
        if any(len(row) != columns for row in data):
            raise ValueError("Matrix rows must all have the same length")
        if any(not isinstance(value, Number) for row in data for value in row):
            raise TypeError("Matrix entries must be numbers")
        self._data = data

    @property
    def data(self) -> tuple[tuple[Number, ...], ...]:
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._data), len(self._data[0]) if self._data else 0)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[tuple[Number, ...]]:
        return iter(self._data)

    def __getitem__(
        self, index: int | slice | tuple[int, int]
    ) -> Number | tuple[Number, ...] | tuple[tuple[Number, ...], ...]:
        if isinstance(index, tuple):
            row, column = index
            return self._data[row][column]
        return self._data[index]

    def __matmul__(self, other: object) -> Matrix | Vector:
        from .core import matrix_matrix_product, matrix_vector_product

        if isinstance(other, Vector):
            return matrix_vector_product(self, other)
        if isinstance(other, Matrix):
            return matrix_matrix_product(self, other)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        return f"Matrix({[list(row) for row in self._data]!r})"
