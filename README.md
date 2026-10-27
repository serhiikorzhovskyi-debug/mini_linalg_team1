# mini_linalg

A small Python linear algebra library with no external dependencies.

```python
from mini_linalg import Matrix, Vector, matrix_matrix_product, matrix_vector_product

a = Matrix([[1, 2, 3], [4, 5, 6]])
v = Vector([2, -1, 3])
b = Matrix([[7, 8], [9, 10], [11, 12]])

print(a.shape)                 # (2, 3)
print(a[1, 2])                 # 6
print(a @ v)                   # Vector([9, 21])
print(a @ b)                   # Matrix([[58, 64], [139, 154]])

assert a @ v == matrix_vector_product(a, v)
assert a @ b == matrix_matrix_product(a, b)
```

`Matrix(rows)` accepts an iterable of equally sized numeric rows;
`Vector(values)` accepts an iterable of numbers. Both copy their inputs and
provide read-only indexing, iteration, value equality, and a tuple-valued
`data` property. `len(matrix)` counts rows and `len(vector)` counts entries.
Shapes are `(rows, columns)` for matrices and `(length,)` for vectors.

Multiplication lives in `mini_linalg.implementation.core` and returns a new
`Matrix` or `Vector`. Invalid operand types or nonnumeric entries raise
`TypeError`; ragged rows or incompatible multiplication dimensions raise
`ValueError`. Matrix multiplication does not conjugate complex entries.

Empty vectors have shape `(0,)`. `Matrix([])` has shape `(0, 0)` and
`Matrix([[], []])` has shape `(2, 0)`. Products with compatible empty dimensions
are supported.

Run the tests from the repository root:

```sh
python -m unittest discover -v
```
