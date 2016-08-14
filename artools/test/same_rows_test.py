import sys
sys.path.append('../')
from artools import same_rows

import scipy as sp


def test_same_1():
    # identical matrices
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    assert (same_rows(A, B) is True)


def test_same_2():
    # same rows, different order
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = sp.array([[1., 0, 0],
                  [0, 1, 3],
                  [0, 1, 2]])

    assert (same_rows(A, B) is True)


def test_same_3():
    # 1-D rows
    A = sp.array([[1., 0, 0]])

    B = sp.array([[1., 0, 0]])

    assert (same_rows(A, B) is True)


def test_same_4():
    # 0-D numpy arrays rows
    A = sp.array([1., 0, 0])

    B = sp.array([1., 0, 0])

    assert (same_rows(A, B) is True)


def test_same_5():
    # identical copies
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = A

    assert (same_rows(A, B) is True)


def test_elements_1():
    # same shape, different elements
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = sp.array([[1., 0, 0],
                  [1, 0, 0],
                  [1, 0, 0]])

    assert (same_rows(A, B) is False)


def test_elements_2():
    # same shape, different elements
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = sp.array([[0., 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])

    assert (same_rows(A, B) is False)


def test_elements_3():
    # same shape, just transposed
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = A.T

    assert (same_rows(A, B) is False)


def test_elements_4():
    # 0-D numpy arrays, with different elements
    A = sp.array([1., 0, 0])

    B = sp.array([1, 2, -3])

    assert (same_rows(A, B) is False)


def test_elements_5():
    # same shape, just the negative of one another
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = -A

    assert (same_rows(A, B) is False)


def test_shape_1():
    # different shapes
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = sp.array([[1., 0, 0],
                  [0, 1, 2]])

    assert (same_rows(A, B) is False)


def test_shape_2():
    # different shapes
    A = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    B = sp.array([[1., 0],
                  [0, 2],
                  [0, 3]])

    assert (same_rows(A, B) is False)


def test_shape_3():
    # different shapes
    A = sp.array([[1., 0],
                  [0, 1],
                  [0, 1]])

    B = sp.array([[1., 0, 0],
                  [0, 1, 2],
                  [0, 1, 3]])

    assert (same_rows(A, B) is False)
