import unittest

from mini_linalg import Matrix, Vector


class VectorTests(unittest.TestCase):
    def test_sequence_interface(self):
        vector = Vector(value for value in [1, -2, 3.5])

        self.assertEqual(vector.shape, (3,))
        self.assertEqual(len(vector), 3)
        self.assertEqual(list(vector), [1, -2, 3.5])
        self.assertEqual(vector[0], 1)
        self.assertEqual(vector[-1], 3.5)
        self.assertEqual(vector[1:], (-2, 3.5))
        with self.assertRaises(IndexError):
            vector[3]

    def test_input_is_copied(self):
        values = [1, 2]
        vector = Vector(values)
        values[0] = 99

        self.assertEqual(vector, Vector([1, 2]))
        self.assertEqual(vector.data, (1, 2))
        with self.assertRaises(TypeError):
            vector.data[0] = 99

    def test_empty_vector(self):
        self.assertEqual(Vector([]).shape, (0,))

    def test_rejects_non_numeric_values(self):
        with self.assertRaises(TypeError):
            Vector([1, "2"])


class MatrixTests(unittest.TestCase):
    def test_sequence_interface(self):
        matrix = Matrix(iter(row) for row in [[1, 2, 3], [4, 5, 6]])

        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(len(matrix), 2)
        self.assertEqual(list(matrix), [(1, 2, 3), (4, 5, 6)])
        self.assertEqual(matrix[1], (4, 5, 6))
        self.assertEqual(matrix[0, 2], 3)
        self.assertEqual(matrix[-1, -1], 6)
        self.assertEqual(matrix[:1], ((1, 2, 3),))
        with self.assertRaises(IndexError):
            matrix[0, 3]

    def test_input_is_copied(self):
        rows = [[1, 2], [3, 4]]
        matrix = Matrix(rows)
        rows[0][0] = 99
        rows.append([5, 6])

        self.assertEqual(matrix, Matrix([[1, 2], [3, 4]]))
        self.assertEqual(matrix.data, ((1, 2), (3, 4)))
        with self.assertRaises(TypeError):
            matrix.data[0][0] = 99

    def test_empty_dimensions(self):
        self.assertEqual(Matrix([]).shape, (0, 0))
        self.assertEqual(Matrix([[], []]).shape, (2, 0))

    def test_rejects_ragged_rows(self):
        for rows in ([[1, 2], [3]], [[], [1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                Matrix(rows)

    def test_rejects_non_numeric_values(self):
        with self.assertRaises(TypeError):
            Matrix([[1, None]])


if __name__ == "__main__":
    unittest.main()
