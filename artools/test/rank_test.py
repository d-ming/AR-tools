import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp


def test_1():
    # identity matrix
    A = sp.array([[1, 0], [0, 1]])
    assert artools.rank(A) == 2


def test_2():
    # two linearly dependent rows
    A = sp.array([[1, 1], [1, 1]])
    assert artools.rank(A) == 1


def test_3():
    # row vector
    A = sp.array([[1, 0]])
    assert artools.rank(A) == 1


def test_4():
    # numpy 0-D array
    A = sp.array([1, 0])
    assert artools.rank(A) == 1