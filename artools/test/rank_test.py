import sys
sys.path.append('../')
from artools import rank

import scipy as sp


def test_1():
    # identity matrix
    A = sp.array([[1, 0], [0, 1]])
    assert rank(A) == 2


def test_2():
    # two linearly dependent rows
    A = sp.array([[1, 1], [1, 1]])
    assert rank(A) == 1


def test_3():
    # row vector
    A = sp.array([[1, 0]])
    assert rank(A) == 1


def test_4():
    # numpy 0-D array
    A = sp.array([1, 0])
    assert rank(A) == 1


def test_5():
    # linear combination of preceeding vectors
    A = sp.array([[1, 1, 1], [2, 1, 2], [3, 2, 3], [1, 1, 1]])
    assert rank(A) == 2


def test_6():
    # matrix and its transpose
    A = sp.array([[1, 1, 1], [2, 1, 2], [3, 2, 3], [1, 1, 1]])
    A_transpose = A.T

    assert rank(A) == rank(A_transpose)


def test_7():
    # square matrix of independent vectors
    A = sp.array([[1, 2], [3, 1]])

    # rank(A) = number of rows
    assert rank(A) == A.shape[0]


def test_8():
    # square matrix of independent vectors
    A = sp.array([[1, 2], [3, 1]])

    # rank(A) = number of columns
    assert rank(A) == A.shape[1]
