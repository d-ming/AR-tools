import sys
sys.path.append('../')
import artools

import scipy as sp


def test_1():
    # standard 2-D column vector
    A = sp.array([[1., 1, 1]]).T

    assert (artools.isColMatrix(A) is True)


def test_2():
    # row vector
    A = sp.array([[1., 1, 1]])

    assert (artools.isColMatrix(A) is False)


def test_3():
    # 0-D array
    A = sp.array([1., 1, 1])

    assert (artools.isColMatrix(A) is False)


def test_4():
    # A list
    A = [1., 1, 1]

    assert (artools.isColMatrix(A) is False)


def test_5():
    # A matrix
    A = sp.array([[1., 1, 1],
                  [2, 2, 2]])

    assert (artools.isColMatrix(A) is False)
