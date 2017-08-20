import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp
import pytest


class TestIn:

    def test_1(self):
        # simple 2-D triangle case
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        xi = sp.array([0.1, 0.1])

        assert artools.inRegion(xi, A, b)


    def test_2(self):
        # simple 2-D triangle case
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        xi = sp.array([0.25, 0.1])

        assert artools.inRegion(xi, A, b)


    def test_3(self):
        # simple 3-D case
        A = sp.array([[-1., 0, 0],
                      [0, -1, 0],
                      [0, 0, -1],
                      [1, 1, 1]])

        b = sp.array([0., 0, 0, 1])

        xi = sp.array([0.25, 0.1, 0.25])

        assert artools.inRegion(xi, A, b)


class TestOut:

    def test_1(self):
        # negative space
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        xi = sp.array([-1., -1])

        assert (artools.inRegion(xi, A, b) is False)


    def test_2(self):
        # positive space
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        xi = sp.array([2., 2])

        assert (artools.inRegion(xi, A, b) is False)


    def test_3(self):
        # simple 3-D case
        A = sp.array([[-1., 0, 0],
                      [0, -1, 0],
                      [0, 0, -1],
                      [1, 1, 1]])

        b = sp.array([0., 0, 0, 1])

        xi = sp.array([0.7, 0.7, 0.7])

        assert (artools.inRegion(xi, A, b) is False)


class TestOn:

    def test_1(self):
        # on a 2-D hyperplane
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        xi = sp.array([0.5, 0.5])

        assert artools.inRegion(xi, A, b)


    def test_2(self):
        # on an extreme point
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        xi = sp.array([1., 0])

        assert artools.inRegion(xi, A, b)


class TestUnbounded:

    def test_1(self):
        # check in an unbounded region
        A = sp.array([[-1., 0],
                      [0, -1]])

        b = sp.array([0., 0])

        xi = sp.array([0.5, 0.5])

        assert artools.inRegion(xi, A, b)


    def test_2(self):
        # check out an unbounded region
        A = sp.array([[-1., 0],
                      [0, -1]])

        b = sp.array([0., 0])

        xi = sp.array([-0.5, -0.5])

        assert artools.inRegion(xi, A, b) is False


    def test_3(self):
        # check on an unbounded region
        A = sp.array([[-1., 0],
                      [0, -1]])

        b = sp.array([0., 0])

        xi = sp.array([0.5, 0])

        assert artools.inRegion(xi, A, b)


class TestToleranceIn:

    def test_1(self):
        # slightly in the region at the origin
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        tol = 1e-8
        xi = sp.array([0.0 + tol, 0.0 + tol])

        assert artools.inRegion(xi, A, b, tol=tol)


    def test_2(self):
        # slightly out of the region at the origin, but still within the
        # accepted tolerance
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        tol = 1e-8
        xi = sp.array([0.0 - tol, 0.0 - tol])

        assert artools.inRegion(xi, A, b, tol=tol)


class TestToleranceOut:

    def test_1(self):
        # slightly out of the region at the origin, and not within the accepted
        # tolerance
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        tol = 1e-10
        xi = sp.array([0.0 - 2*tol, 0.0 + tol])

        assert (artools.inRegion(xi, A, b, tol=tol) is False)


class TestToleranceOn:

    def test_1(self):
        # on an extreme point specifying tolerance
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([0., 0, 1])

        xi = sp.array([1., 0])

        tol = 1e-2

        assert artools.inRegion(xi, A, b, tol=tol)


class TestShape:

    def test_1(self):
        # b is a 2-D column vector
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([[0., 0, 1]]).T

        xi = sp.array([0.25, 0.1])

        assert artools.inRegion(xi, A, b)


    def test_2(self):
        # b is a 2-D row vector
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([[0., 0, 1]])

        xi = sp.array([0.25, 0.1])

        assert artools.inRegion(xi, A, b)


    def test_3(self):
        # A and b have incompatible shapes
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([[0., 1]])

        xi = sp.array([0.25, 0.1])

        with pytest.raises(ValueError):
            artools.inRegion(xi, A, b)

    
    def test_4(self):
        # xi has incompatible shape
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([[0., 0, 1]])

        xi = sp.array([0.25, 0.1, 0.1])

        with pytest.raises(ValueError):
            artools.inRegion(xi, A, b)
