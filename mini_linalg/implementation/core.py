from .matrix import Matrix
from .vector import Vector


def matrix_vector_product(matrix: Matrix, vector: Vector) -> Vector:
    if not isinstance(matrix, Matrix) or not isinstance(vector, Vector):
        raise TypeError("matrix_vector_product expects a Matrix and a Vector")
    if matrix.shape[1] != len(vector):
        raise ValueError(
            f"Cannot multiply matrix of shape {matrix.shape} "
            f"by vector of shape {vector.shape}"
        )
    return Vector(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def matrix_matrix_product(left: Matrix, right: Matrix) -> Matrix:
    if not isinstance(left, Matrix) or not isinstance(right, Matrix):
        raise TypeError("matrix_matrix_product expects two Matrix operands")
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"Cannot multiply matrices of shapes {left.shape} and {right.shape}"
        )
    columns = tuple(zip(*right))
    return Matrix(
        [sum(a * b for a, b in zip(row, column)) for column in columns]
        for row in left
    )
