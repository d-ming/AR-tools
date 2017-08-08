import sys
sys.path.append('../')
import artools

import scipy as sp


def test_isCol_1():
    # standard 2-D column vector
    A = sp.array([[1., 1, 1]]).T

    assert (artools.isColVector(A) is True)


def test_isCol_2():
    # row vector
    A = sp.array([[1., 1, 1]])

    assert (artools.isColVector(A) is False)


def test_isCol_3():
    # 0-D array
    A = sp.array([1., 1, 1])

    assert (artools.isColVector(A) is False)


def test_isCol_4():
    # A list
    A = [1., 1, 1]

    assert (artools.isColVector(A) is False)


def test_isCol_5():
    # A matrix
    A = sp.array([[1., 1, 1],
                  [2, 2, 2]])

    assert (artools.isColVector(A) is False)


def test_isRow_1():
    # standard 2-D column vector
    A = sp.array([[1., 1, 1]]).T

    assert (artools.isRowVector(A) is False)


def test_isRow_2():
    # row vector
    A = sp.array([[1., 1, 1]])

    assert (artools.isRowVector(A) is True)


def test_isRow_3():
    # 0-D array
    A = sp.array([1., 1, 1])

    assert (artools.isRowVector(A) is False)


def test_isRow_4():
    # A list
    A = [1., 1, 1]

    assert (artools.isRowVector(A) is False)


def test_isRow_5():
    # A matrix
    A = sp.array([[1., 1, 1],
                  [2, 2, 2]])

    assert (artools.isRowVector(A) is False)
