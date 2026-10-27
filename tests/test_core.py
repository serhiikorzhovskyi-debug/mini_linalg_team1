import unittest

from mini_linalg.implementation.core import matrix_matrix_product, matrix_vector_product
from mini_linalg.implementation.matrix import Matrix
from mini_linalg.implementation.vector import Vector


class MatrixVectorProductTests(unittest.TestCase):
    def test_rectangular_product(self):
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        vector = Vector([2, -1, 3])

        self.assertEqual(matrix_vector_product(matrix, vector), Vector([9, 21]))

    def test_matmul_operator(self):
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        vector = Vector([2, -1, 3])

        self.assertEqual(matrix @ vector, Vector([9, 21]))

    def test_float_and_complex_entries(self):
        self.assertEqual(
            matrix_vector_product(Matrix([[0.5, -2.0]]), Vector([4.0, 1.5])),
            Vector([-1.0]),
        )
        self.assertEqual(
            matrix_vector_product(Matrix([[1j, 2]]), Vector([1j, 3])),
            Vector([5]),
        )

    def test_zero_vector(self):
        self.assertEqual(
            matrix_vector_product(Matrix([[1, -2], [3, 4]]), Vector([0, 0])),
            Vector([0, 0]),
        )

    def test_empty_dimensions(self):
        self.assertEqual(matrix_vector_product(Matrix([]), Vector([])), Vector([]))
        self.assertEqual(
            matrix_vector_product(Matrix([[], []]), Vector([])), Vector([0, 0])
        )

    def test_dimension_mismatch(self):
        matrix = Matrix([[1, 2]])

        for vector in (Vector([]), Vector([1]), Vector([1, 2, 3])):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    matrix_vector_product(matrix, vector)
                with self.assertRaises(ValueError):
                    matrix @ vector

    def test_invalid_operand_types(self):
        for matrix, vector in (([[1]], Vector([1])), (Matrix([[1]]), [1])):
            with self.subTest(matrix=matrix, vector=vector):
                with self.assertRaises(TypeError):
                    matrix_vector_product(matrix, vector)
        with self.assertRaises(TypeError):
            Matrix([[1]]) @ [1]

    def test_preserves_inputs_and_returns_vector(self):
        matrix = Matrix([[1, 2], [3, 4]])
        vector = Vector([5, 6])
        result = matrix_vector_product(matrix, vector)

        self.assertIsInstance(result, Vector)
        self.assertIsNot(result, vector)
        self.assertEqual(matrix, Matrix([[1, 2], [3, 4]]))
        self.assertEqual(vector, Vector([5, 6]))


class MatrixMatrixProductTests(unittest.TestCase):
    def test_rectangular_product(self):
        left = Matrix([[1, 2, 3], [4, 5, 6]])
        right = Matrix([[7, 8], [9, 10], [11, 12]])
        result = matrix_matrix_product(left, right)

        self.assertIsInstance(result, Matrix)
        self.assertEqual(result, Matrix([[58, 64], [139, 154]]))
        self.assertEqual(result.shape, (2, 2))

    def test_matmul_operator(self):
        self.assertEqual(
            Matrix([[1, 2, 3], [4, 5, 6]])
            @ Matrix([[7, 8], [9, 10], [11, 12]]),
            Matrix([[58, 64], [139, 154]]),
        )

    def test_outer_product(self):
        self.assertEqual(
            matrix_matrix_product(Matrix([[2], [-1]]), Matrix([[3, 4, 5]])),
            Matrix([[6, 8, 10], [-3, -4, -5]]),
        )

    def test_float_and_complex_entries(self):
        self.assertEqual(
            matrix_matrix_product(Matrix([[0.5, -2.0]]), Matrix([[4.0], [1.5]])),
            Matrix([[-1.0]]),
        )
        self.assertEqual(
            matrix_matrix_product(Matrix([[1j]]), Matrix([[1j]])), Matrix([[-1]])
        )

    def test_identity(self):
        matrix = Matrix([[1, 2], [3, 4]])
        identity = Matrix([[1, 0], [0, 1]])

        self.assertEqual(matrix_matrix_product(matrix, identity), matrix)
        self.assertEqual(matrix_matrix_product(identity, matrix), matrix)

    def test_multiplication_order(self):
        left = Matrix([[1, 2], [0, 1]])
        right = Matrix([[1, 0], [3, 1]])

        self.assertEqual(matrix_matrix_product(left, right), Matrix([[7, 2], [3, 1]]))
        self.assertEqual(matrix_matrix_product(right, left), Matrix([[1, 2], [3, 7]]))

    def test_empty_dimensions(self):
        for left, right, expected in (
            (Matrix([]), Matrix([]), Matrix([])),
            (Matrix([[], []]), Matrix([]), Matrix([[], []])),
            (Matrix([[1, 2]]), Matrix([[], []]), Matrix([[]])),
        ):
            with self.subTest(left=left, right=right):
                self.assertEqual(matrix_matrix_product(left, right), expected)

    def test_dimension_mismatch(self):
        left = Matrix([[1, 2]])

        for right in (Matrix([]), Matrix([[1]]), Matrix([[1], [2], [3]])):
            with self.subTest(right=right):
                with self.assertRaises(ValueError):
                    matrix_matrix_product(left, right)
                with self.assertRaises(ValueError):
                    left @ right

    def test_invalid_operand_types(self):
        for left, right in (([[1]], Matrix([[1]])), (Matrix([[1]]), Vector([1]))):
            with self.subTest(left=left, right=right):
                with self.assertRaises(TypeError):
                    matrix_matrix_product(left, right)
        with self.assertRaises(TypeError):
            Matrix([[1]]) @ 2

    def test_preserves_inputs_and_returns_new_matrix(self):
        left = Matrix([[1, 2], [3, 4]])
        right = Matrix([[5, 6], [7, 8]])
        result = matrix_matrix_product(left, right)

        self.assertIsNot(result, left)
        self.assertIsNot(result, right)
        self.assertEqual(left, Matrix([[1, 2], [3, 4]]))
        self.assertEqual(right, Matrix([[5, 6], [7, 8]]))


if __name__ == "__main__":
    unittest.main()
