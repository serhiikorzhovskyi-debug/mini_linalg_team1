from collections.abc import Iterable, Iterator
from numbers import Number


class Vector:
    def __init__(self, values: Iterable[Number]) -> None:
        data = tuple(values)
        if any(not isinstance(value, Number) for value in data):
            raise TypeError("Vector entries must be numbers")
        self._data = data

    @property
    def data(self) -> tuple[Number, ...]:
        return self._data

    @property
    def shape(self) -> tuple[int]:
        return (len(self._data),)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Number]:
        return iter(self._data)

    def __getitem__(self, index: int | slice) -> Number | tuple[Number, ...]:
        return self._data[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        return f"Vector({list(self._data)!r})"
