import unittest

from mini_linalg import Matrix, Vector, matrix_matrix_product, matrix_vector_product


class MatrixVectorProductTests(unittest.TestCase):
    def test_rectangular_product(self):
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])
        vector = Vector([2, -1, 3])

        self.assertEqual(matrix_vector_product(matrix, vector), Vector([9, 21]))
        self.assertEqual(matrix @ vector, Vector([9, 21]))
        self.assertEqual(matrix, Matrix([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(vector, Vector([2, -1, 3]))

    def test_float_and_complex_entries(self):
        self.assertEqual(Matrix([[0.5, -2.0]]) @ Vector([4.0, 1.5]), Vector([-1.0]))
        self.assertEqual(Matrix([[1j, 2]]) @ Vector([1j, 3]), Vector([5]))

    def test_empty_dimensions(self):
        self.assertEqual(Matrix([]) @ Vector([]), Vector([]))
        self.assertEqual(Matrix([[], []]) @ Vector([]), Vector([0, 0]))

    def test_dimension_mismatch(self):
        for vector in (Vector([1]), Vector([1, 2, 3])):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                matrix_vector_product(Matrix([[1, 2]]), vector)

    def test_invalid_operand_types(self):
        for matrix, vector in (([[1]], Vector([1])), (Matrix([[1]]), [1])):
            with self.subTest(matrix=matrix, vector=vector):
                with self.assertRaises(TypeError):
                    matrix_vector_product(matrix, vector)
        with self.assertRaises(TypeError):
            Matrix([[1]]) @ [1]


class MatrixMatrixProductTests(unittest.TestCase):
    def test_rectangular_product(self):
        left = Matrix([[1, 2, 3], [4, 5, 6]])
        right = Matrix([[7, 8], [9, 10], [11, 12]])
        expected = Matrix([[58, 64], [139, 154]])

        self.assertEqual(matrix_matrix_product(left, right), expected)
        self.assertEqual(left @ right, expected)
        self.assertEqual((left @ right).shape, (2, 2))
        self.assertEqual(left, Matrix([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(right, Matrix([[7, 8], [9, 10], [11, 12]]))

    def test_outer_product(self):
        self.assertEqual(
            Matrix([[2], [-1]]) @ Matrix([[3, 4, 5]]),
            Matrix([[6, 8, 10], [-3, -4, -5]]),
        )

    def test_float_and_complex_entries(self):
        self.assertEqual(
            Matrix([[0.5, -2.0]]) @ Matrix([[4.0], [1.5]]), Matrix([[-1.0]])
        )
        self.assertEqual(Matrix([[1j]]) @ Matrix([[1j]]), Matrix([[-1]]))

    def test_identity(self):
        matrix = Matrix([[1, 2], [3, 4]])
        identity = Matrix([[1, 0], [0, 1]])

        self.assertEqual(matrix @ identity, matrix)
        self.assertEqual(identity @ matrix, matrix)
        self.assertIsNot(matrix @ identity, matrix)

    def test_empty_dimensions(self):
        self.assertEqual(Matrix([]) @ Matrix([]), Matrix([]))
        self.assertEqual(Matrix([[], []]) @ Matrix([]), Matrix([[], []]))
        self.assertEqual(Matrix([[1, 2]]) @ Matrix([[], []]), Matrix([[]]))

    def test_dimension_mismatch(self):
        for right in (Matrix([[1]]), Matrix([[1], [2], [3]])):
            with self.subTest(right=right), self.assertRaises(ValueError):
                matrix_matrix_product(Matrix([[1, 2]]), right)

    def test_invalid_operand_types(self):
        for left, right in (([[1]], Matrix([[1]])), (Matrix([[1]]), Vector([1]))):
            with self.subTest(left=left, right=right):
                with self.assertRaises(TypeError):
                    matrix_matrix_product(left, right)
        with self.assertRaises(TypeError):
            Matrix([[1]]) @ 2


if __name__ == "__main__":
    unittest.main()
