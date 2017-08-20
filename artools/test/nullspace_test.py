import sys
sys.path.append('../')
from artools import sameRows, nullspace, rank

import scipy as sp


def test_1():
    # 0-D nullspace Identity matrix
    I = sp.eye(5)

    N = nullspace(I)

    assert (N.shape == (5, 0))


def test_2():
    # 1-D nullspace
    A = sp.array([[1., 0, 0],
                  [0, 1, 0]])

    N = nullspace(A)
    N_ref = sp.array([[0., 0, 1]]).T

    assert (sameRows(N, N_ref) is True)


def test_3():
    # 2-D nullspace
    A = sp.array([[1., 0, 0]])

    N = nullspace(A)
    N_ref = sp.array([[0., 0],
                      [1, 0],
                      [0, 1]])

    assert (sameRows(N, N_ref) is True)


def test_4():
    # 1-D subspace in 5-D
    A = sp.array([[1., 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0]])

    N = nullspace(A)
    N_ref = sp.array([[0., 0, 0, 0, 1]]).T

    assert (sameRows(N, N_ref) is True)


def test_5():
    # test a 3x2 matrix
    A = sp.array([[1., 0],
                  [0, 1],
                  [0, 0]])

    N = nullspace(A)

    assert N.shape == (2, 0)


def test_6():
    # dimension of the nullspace
    A = sp.array([[1., 2., 3., 1.],
                  [1., 1., 2., 1.],
                  [1., 2., 3., 1.]])

    N = nullspace(A)
    dimension = rank(N)

    assert dimension == 2
