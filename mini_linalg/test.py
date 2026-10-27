import unittest

from mini_linalg.implementation.matrix import Matrix
from mini_linalg.implementation.vector import Vector


class VectorTests(unittest.TestCase):
    def test_shape_and_length(self):
        vector = Vector([1, -2, 3.5])

        self.assertEqual(vector.shape, (3,))
        self.assertEqual(len(vector), 3)

    def test_indexing_and_iteration(self):
        vector = Vector([1, -2, 3.5])

        self.assertEqual(list(vector), [1, -2, 3.5])
        self.assertEqual(vector[0], 1)
        self.assertEqual(vector[-1], 3.5)
        self.assertEqual(vector[1:], (-2, 3.5))
        with self.assertRaises(IndexError):
            vector[3]

    def test_generator_input(self):
        self.assertEqual(Vector(value for value in [1, 2, 3]), Vector([1, 2, 3]))

    def test_input_is_copied(self):
        values = [1, 2]
        vector = Vector(values)
        values[0] = 99
        values.append(3)

        self.assertEqual(vector.data, (1, 2))

    def test_data_is_read_only(self):
        vector = Vector([1, 2])

        with self.assertRaises(TypeError):
            vector.data[0] = 99

    def test_value_equality(self):
        self.assertEqual(Vector([1, 2]), Vector([1, 2]))
        self.assertNotEqual(Vector([1, 2]), Vector([2, 1]))
        self.assertNotEqual(Vector([1, 2]), Vector([1, 2, 3]))
        self.assertNotEqual(Vector([1, 2]), [1, 2])

    def test_empty_vector(self):
        vector = Vector([])

        self.assertEqual(vector.shape, (0,))
        self.assertEqual(len(vector), 0)
        self.assertEqual(list(vector), [])

    def test_rejects_non_numeric_values(self):
        for value in ("2", None, [2]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                Vector([1, value])


class MatrixTests(unittest.TestCase):
    def test_rectangular_shape_and_length(self):
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(len(matrix), 2)

    def test_indexing_and_iteration(self):
        matrix = Matrix([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(list(matrix), [(1, 2, 3), (4, 5, 6)])
        self.assertEqual(matrix[1], (4, 5, 6))
        self.assertEqual(matrix[0, 2], 3)
        self.assertEqual(matrix[-1, -1], 6)
        self.assertEqual(matrix[:1], ((1, 2, 3),))
        for index in ((2, 0), (0, 3)):
            with self.subTest(index=index), self.assertRaises(IndexError):
                matrix[index]

    def test_generator_input(self):
        matrix = Matrix(iter(row) for row in [[1, 2], [3, 4]])

        self.assertEqual(matrix, Matrix([[1, 2], [3, 4]]))

    def test_input_is_copied(self):
        rows = [[1, 2], [3, 4]]
        matrix = Matrix(rows)
        rows[0][0] = 99
        rows.append([5, 6])

        self.assertEqual(matrix.data, ((1, 2), (3, 4)))

    def test_data_is_read_only(self):
        matrix = Matrix([[1, 2]])

        with self.assertRaises(TypeError):
            matrix.data[0][0] = 99

    def test_value_equality(self):
        self.assertEqual(Matrix([[1, 2]]), Matrix([[1, 2]]))
        self.assertNotEqual(Matrix([[1, 2]]), Matrix([[2, 1]]))
        self.assertNotEqual(Matrix([[1, 2]]), Matrix([[1], [2]]))
        self.assertNotEqual(Matrix([[1, 2]]), [[1, 2]])

    def test_empty_dimensions(self):
        self.assertEqual(Matrix([]).shape, (0, 0))
        self.assertEqual(Matrix([[], []]).shape, (2, 0))

    def test_rejects_ragged_rows(self):
        for rows in ([[1, 2], [3]], [[], [1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                Matrix(rows)

    def test_rejects_non_numeric_values(self):
        for value in ("2", None, [2]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                Matrix([[1, value]])


if __name__ == "__main__":
    unittest.main()
