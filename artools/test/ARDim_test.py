import sys
sys.path.append('../')
from artools import ARDim, stoichSubspace

import scipy as sp


class TestZero:

    def test_1(self):
        x = sp.zeros([1, 3])
        assert (ARDim(x) == 0)


    def test_2(self):
        x = sp.zeros([1, 3]).T
        assert (ARDim(x) == 0)


    def test_3(self):
        x = sp.zeros([3, 3])
        assert (ARDim(x) == 0)

    def test_4(self):
        # row vector
        x = sp.array([[1.0, 2.0, 3.0, 4.0]])
        assert (ARDim(x) == 0)


    def test_5(self):
        # column vector
        x = sp.array([[1.0, 2.0, 3.0, 4.0]]).T
        assert (ARDim(x) == 0)


    def test_6(self):
        # 1-D numpy array
        x = sp.array([1.0, 2.0, 3.0, 4.0])
        assert (ARDim(x) == 0)


    def test_7(self):
        # repeated row
        x = sp.array([[1.0, 2.0],
                      [1.0, 2.0]])
        assert (ARDim(x) == 0)


class Test1D:

    def test_1(self):
        # two points in 2-D space
        Xs = sp.array([[0.0, 0.0],
                       [1.0, 1.0]])

        assert (ARDim(Xs) == 1)


    def test_2(self):
        # two points in 3-D space
        Xs = sp.array([[0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0]])

        assert (ARDim(Xs) == 1)


class Test2D:

    def test_1(self):
        # three points in 2-D space
        Xs = sp.array([[0.0, 0.0],
                       [1.0, 1.0],
                       [1.0, 0.0]])

        assert (ARDim(Xs) == 2)


class TestAR:

    def test_NH3(self):
        # N2 + 3H2 -> 2NH3

        stoich_mat = sp.array([[-1.0, -3.0, 2.0]]).T
        Cf0 = sp.array([1.0, 1.0, 0.0])

        Vs = stoichSubspace(Cf0, stoich_mat)['all_Cs']

        assert (ARDim(Vs) == 1)


    def test_VDV_2D(self):
        # A -> B -> C

        stoich_mat = sp.array([[-1.0, 0.0],
                               [1.0, -1.0],
                               [0.0, 1.0]])

        Cf0 = sp.array([1.0, 0.0, 0.0])

        Vs = stoichSubspace(Cf0, stoich_mat)['all_Cs']

        assert (ARDim(Vs) == 2)


    def test_VDV_3D(self):
        # A -> B -> C
        # 2A -> D

        stoich_mat = sp.array([[-1.0, 0.0, -2.0],
                               [1.0, -1.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])

        Cf0 = sp.array([1.0, 0.0, 0.0, 0.0])

        Vs = stoichSubspace(Cf0, stoich_mat)['all_Cs']

        assert (ARDim(Vs) == 3)


class TestRand:

    def test_0a(self):

        Xs = sp.rand(10, 1)
        assert ARDim(Xs) == 0


    def test_0b(self):

        Xs = sp.rand(1, 10)
        assert ARDim(Xs) == 0


    def test_1a(self):

        Xs = sp.rand(2, 3)
        assert ARDim(Xs) == 1


    def test_2(self):

        Xs = sp.rand(10, 2)
        assert ARDim(Xs) == 2


    def test_3(self):

        Xs = sp.rand(10, 3)
        assert ARDim(Xs) == 3


    def test_4(self):

        Xs = sp.rand(10, 4)
        assert ARDim(Xs) == 4
