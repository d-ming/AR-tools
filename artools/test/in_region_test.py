import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp


def test_in_1():
    # 0-D b vector
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([0.1, 0.1])

    assert artools.in_region(xi, A, b)


def test_in_2():
    # 0-D b vector
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([0.25, 0.1])

    assert artools.in_region(xi, A, b)


def test_out_1():
    # negative space
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([-1., -1])

    assert (artools.in_region(xi, A, b) is False)


def test_out_2():
    # positive space
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([2., 2])

    assert (artools.in_region(xi, A, b) is False)


def test_on_1():
    # on a hyperplane
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([0.5, 0.5])

    assert artools.in_region(xi, A, b)


def test_on_2():
    # on an extreme point
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([1., 0])

    assert artools.in_region(xi, A, b)


def test_tol_in_1():
    # slightly in the region at the origin
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    tol = 1e-8
    xi = sp.array([0.0 + tol, 0.0 + tol])

    assert artools.in_region(xi, A, b, tol=tol)


def test_tol_in_2():
    # slightly out of the region at the origin, but still within the accepted
    # tolerance
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    tol = 1e-8
    xi = sp.array([0.0 - tol, 0.0 - tol])

    assert artools.in_region(xi, A, b, tol=tol)


def test_tol_out_1():
    # slightly out of the region at the origin, and not within the accepted
    # tolerance
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    tol = 1e-10
    xi = sp.array([0.0 - 2*tol, 0.0 + tol])

    assert (artools.in_region(xi, A, b, tol=tol) is False)
