from mini_linalg import Matrix, Vector, matrix_matrix_product, matrix_vector_product


def main() -> None:
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    vector = Vector([2, -1, 3])
    other = Matrix([[7, 8], [9, 10], [11, 12]])

    vector_result = matrix_vector_product(matrix, vector)
    matrix_result = matrix_matrix_product(matrix, other)

    assert vector_result == Vector([9, 21])
    assert matrix_result == Matrix([[58, 64], [139, 154]])
    assert matrix @ vector == vector_result
    assert matrix @ other == matrix_result

    print(f"Matrix-vector product: {vector_result}")
    print(f"Matrix-matrix product: {matrix_result}")
    print("All example checks passed.")


if __name__ == "__main__":
    main()
