"""Matrix multiplication operations."""

from .matrix import Matrix
from .vector import Vector


def matrix_vector_product(matrix, vector):
    """Multiply a matrix by a vector without modifying either operand."""
    if not isinstance(matrix, Matrix) or not isinstance(vector, Vector):
        raise TypeError("Expected a Matrix and a Vector")
    if matrix.shape[1] != len(vector):
        raise ValueError("Matrix columns must match vector length")
    return Vector(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def matrix_matrix_product(left, right):
    """Multiply matrices with matching inner dimensions."""
    if not isinstance(left, Matrix) or not isinstance(right, Matrix):
        raise TypeError("Expected two Matrix operands")
    if left.shape[1] != right.shape[0]:
        raise ValueError("Left columns must match right rows")
    columns = tuple(zip(*right))
    return Matrix(
        [sum(a * b for a, b in zip(row, column)) for column in columns]
        for row in left
    )
