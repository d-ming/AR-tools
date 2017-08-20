import sys
sys.path.append('../')
from artools import sameRows, con2vert

import scipy as sp


class Test2D:

    def test_1(self):
        # 2-D mass balance triangle
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([[0., 0, 1]]).T

        vs = con2vert(A, b)
        vs_ref = sp.array([[0., 1], [0, 0], [1, 0]])

        assert (sameRows(vs, vs_ref) is True)

    def test_2(self):
        # 2-D unit square
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 0],
                      [0, 1]])

        b = sp.array([[0., 0, 1, 1]]).T

        vs = con2vert(A, b)
        vs_ref = sp.array([[0., 1],
                           [0, 0],
                           [1, 0],
                           [1, 1]])

        assert (sameRows(vs, vs_ref) is True)

    def test_3(self):
        # mass balance triangle cut at y = 0.5
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1],
                      [0, 1]])

        b = sp.array([[0., 0, 1, 0.5]]).T

        vs = con2vert(A, b)
        vs_ref = sp.array([[0, 0.5],
                           [0, 0],
                           [1, 0],
                           [0.5, 0.5]])

        assert (sameRows(vs, vs_ref) is True)


class Test3D:

    def test_1(self):
        # 3-D mass balance triangle
        A = sp.array([[-1., 0, 0],
                      [0, -1, 0],
                      [0, 0, -1],
                      [1, 1, 1]])
        b = sp.array([[0., 0, 0, 1]]).T

        vs = con2vert(A, b)
        vs_ref = sp.array([[1., 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])

        assert (sameRows(vs, vs_ref) is True)

    def test_2(self):
        # 3-D unit cube
        A = sp.array([[-1., 0, 0],
                      [0, -1, 0],
                      [0, 0, -1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

        b = sp.array([[0., 0, 0, 1, 1, 1]]).T

        vs = con2vert(A, b)
        vs_ref = sp.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [1, 0, 1],
                           [1, 1, 1],
                           [0, 1, 1]])

        assert (sameRows(vs, vs_ref) is True)


class TestMisc:

    def test_interior_1(self):

        """
        This set of constraints produces an initial guess that is not an
        interior point of the region.
        """

        A = sp.array([[1.0, 0.0],
                      [1.0, 1.0],
                      [-1.0, 1.0],
                      [-3.0, -1.0],
                      [0.0, -1.0]])

        b = sp.array([[1.0, 1.0, 0.0, 0.0, 0.0]]).T

        vs_ref = sp.array([[0.5, 0.5],
                           [0.0, 0.0],
                           [1.0, 0.0]])

        vs = con2vert(A, b)

        assert (sameRows(vs, vs_ref) is True)
