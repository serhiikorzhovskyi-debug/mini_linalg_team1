class Vector:
    def __init__(self, data: list[float]):
        self.data = list(data)

    def __repr__(self) -> str:
        return f"Vector({self.data})"


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = [list(r) for r in data]

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def batch_matvec(self, vectors: list[Vector]) -> list[Vector]:
        """Multiply this matrix by every vector in the list."""
        n_cols = len(self.data[0])
        results = []
        for v in vectors:
            if len(v.data) != n_cols:
                raise ValueError("Dimension mismatch in batch_matvec")
            entries = []
            for row in self.data:
                total = 0.0
                for a, b in zip(row, v.data):
                    total += a * b
                entries.append(total)
            results.append(Vector(entries))
        return results

    def batch_matmul(self, matrices: list["Matrix"]) -> list["Matrix"]:
        """Multiply this matrix by every matrix in the list."""
        n_cols = len(self.data[0])
        results = []
        for other in matrices:
            if len(other.data) != n_cols:
                raise ValueError("Dimension mismatch in batch_matmul")
            product = []
            for row in self.data:
                new_row = []
                for j in range(len(other.data[0])):
                    total = 0.0
                    for k in range(n_cols):
                        total += row[k] * other.data[k][j]
                    new_row.append(total)
                product.append(new_row)
            results.append(Matrix(product))
        return results