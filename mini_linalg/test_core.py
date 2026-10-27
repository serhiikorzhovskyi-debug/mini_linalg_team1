import pytest

from mini_linalg import Vector, Matrix


def test_vector_creation():
    v = Vector([1.0, 2.0, 3.0])
    assert v.data == [1.0, 2.0, 3.0]


def test_matrix_creation():
    m = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.data == [[1.0, 2.0], [3.0, 4.0]]


def test_batch_matvec():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    vectors = [Vector([1.0, 0.0]), Vector([0.0, 1.0]), Vector([1.0, 1.0])]
    results = A.batch_matvec(vectors)
    assert len(results) == 3
    assert list(results[0].data) == [1.0, 3.0]
    assert list(results[1].data) == [2.0, 4.0]
    assert list(results[2].data) == [3.0, 7.0]


def test_batch_matvec_empty_list():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert A.batch_matvec([]) == []


def test_batch_matvec_dimension_mismatch():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        A.batch_matvec([Vector([1.0, 2.0]), Vector([1.0, 2.0, 3.0])])


def test_batch_matmul():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    identity = Matrix([[1.0, 0.0], [0.0, 1.0]])
    B = Matrix([[2.0, 0.0], [1.0, 3.0]])
    results = A.batch_matmul([identity, B])
    assert len(results) == 2
    assert [list(r) for r in results[0].data] == [[1.0, 2.0], [3.0, 4.0]]
    assert [list(r) for r in results[1].data] == [[4.0, 6.0], [10.0, 12.0]]


def test_batch_matmul_dimension_mismatch():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    wrong = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError):
        A.batch_matmul([wrong])