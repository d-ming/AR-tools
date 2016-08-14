import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp
import pytest


def test_in_1():
    # simple 2-D triangle case
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([0.1, 0.1])

    assert artools.out_region(xi, A, b) is False


def test_in_2():
    # simple 2-D triangle case
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([0.25, 0.1])

    assert artools.out_region(xi, A, b) is False


def test_in_3():
    # simple 3-D case
    A = sp.array([[-1., 0, 0],
                  [0, -1, 0],
                  [0, 0, -1],
                  [1, 1, 1]])

    b = sp.array([0., 0, 0, 1])

    xi = sp.array([0.25, 0.1, 0.25])

    assert artools.out_region(xi, A, b) is False


def test_out_1():
    # negative space
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([-1., -1])

    assert artools.out_region(xi, A, b)


def test_out_2():
    # positive space
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([2., 2])

    assert artools.out_region(xi, A, b)


def test_out_3():
    # simple 3-D case
    A = sp.array([[-1., 0, 0],
                  [0, -1, 0],
                  [0, 0, -1],
                  [1, 1, 1]])

    b = sp.array([0., 0, 0, 1])

    xi = sp.array([0.7, 0.7, 0.7])

    assert artools.out_region(xi, A, b)


def test_on_1():
    # on a 2-D hyperplane
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([0.5, 0.5])

    assert artools.out_region(xi, A, b) is False


def test_on_2():
    # on an extreme point
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    xi = sp.array([1., 0])

    assert artools.out_region(xi, A, b) is False


def test_tol_in_1():
    # slightly in the region at the origin
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    tol = 1e-8
    xi = sp.array([0.0 + tol, 0.0 + tol])

    assert artools.out_region(xi, A, b, tol=tol) is False


def test_tol_in_2():
    # slightly out of the region at the origin, but still within the accepted
    # tolerance
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    tol = 1e-8
    xi = sp.array([0.0 - tol, 0.0 - tol])

    assert artools.out_region(xi, A, b, tol=tol) is False


def test_tol_out_1():
    # slightly out of the region at the origin, and not within the accepted
    # tolerance
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([0., 0, 1])

    tol = 1e-10
    xi = sp.array([0.0 - 2*tol, 0.0 + tol])

    assert artools.out_region(xi, A, b, tol=tol)


def test_shape_1():
    # b is a 2-D column vector
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([[0., 0, 1]]).T

    xi = sp.array([-0.25, -0.1])

    assert artools.out_region(xi, A, b)


def test_shape_2():
    # b is a 2-D row vector
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([[0., 0, 1]])

    xi = sp.array([-0.25, -0.1])

    assert artools.out_region(xi, A, b)


def test_shape_3():
    # A and b have incompatible shapes
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([[0., 1]])

    xi = sp.array([0.25, 0.1])

    with pytest.raises(ValueError):
        artools.out_region(xi, A, b)


def test_shape_4():
    # xi has incompatible shape
    A = sp.array([[-1., 0],
                  [0, -1],
                  [1, 1]])

    b = sp.array([[0., 0, 1]])

    xi = sp.array([0.25, 0.1, 0.1])

    with pytest.raises(ValueError):
        artools.out_region(xi, A, b)
