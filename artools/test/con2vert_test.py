import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp


class Test2D:

    def test_1(self):
        # 2-D mass balance triangle
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1]])

        b = sp.array([[0., 0, 1]]).T

        vs = artools.con2vert(A, b)
        vs_ref = sp.array([[0., 1], [0, 0], [1, 0]])

        assert (artools.same_rows(vs, vs_ref) is True)

    def test_2(self):
        # 2-D unit square
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 0],
                      [0, 1]])

        b = sp.array([[0., 0, 1, 1]]).T

        vs = artools.con2vert(A, b)
        vs_ref = sp.array([[0., 1],
                           [0, 0],
                           [1, 0],
                           [1, 1]])

        assert (artools.same_rows(vs, vs_ref) is True)

    def test_3(self):
        # mass blanace triangle cut at y = 0.5
        A = sp.array([[-1., 0],
                      [0, -1],
                      [1, 1],
                      [0, 1]])

        b = sp.array([[0., 0, 1, 0.5]]).T

        vs = artools.con2vert(A, b)
        vs_ref = sp.array([[0, 0.5],
                           [0, 0],
                           [1, 0],
                           [0.5, 0.5]])

        assert (artools.same_rows(vs, vs_ref) is True)


class Test3D:

    def test_1(self):
        # 3-D mass balance triangle
        A = sp.array([[-1., 0, 0],
                      [0, -1, 0],
                      [0, 0, -1],
                      [1, 1, 1]])
        b = sp.array([[0., 0, 0, 1]]).T

        vs = artools.con2vert(A, b)
        vs_ref = sp.array([[1., 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])

        assert (artools.same_rows(vs, vs_ref) is True)

    def test_2(self):
        # 3-D unit cube
        A = sp.array([[-1., 0, 0],
                      [0, -1, 0],
                      [0, 0, -1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

        b = sp.array([[0., 0, 0, 1, 1, 1]]).T

        vs = artools.con2vert(A, b)
        vs_ref = sp.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [1, 0, 1],
                           [1, 1, 1],
                           [0, 1, 1]])

        assert (artools.same_rows(vs, vs_ref) is True)
