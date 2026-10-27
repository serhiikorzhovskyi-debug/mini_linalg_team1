"""A vector of numeric values."""

from numbers import Number


class Vector:
    def __init__(self, values):
        self._data = tuple(values)
        if any(not isinstance(value, Number) for value in self._data):
            raise TypeError("Vector entries must be numeric")

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return (len(self),)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.data == other.data

    def __repr__(self):
        return f"Vector({self.data!r})"
