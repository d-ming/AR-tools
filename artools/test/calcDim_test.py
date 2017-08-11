import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp

from artools import calcDim


class TestZero:

    def test_1(self):
        x = sp.zeros([1, 3])
        assert (calcDim(x) == 0)


    def test_2(self):
        x = sp.zeros([1, 3]).T
        assert (calcDim(x) == 0)


    def test_3(self):
        x = sp.zeros([3, 3])
        assert (calcDim(x) == 0)

    def test_4(self):
        # row vector
        x = sp.array([[1.0, 2.0, 3.0, 4.0]])
        assert (calcDim(x) == 0)


    def test_5(self):
        # column vector
        x = sp.array([[1.0, 2.0, 3.0, 4.0]]).T
        assert (calcDim(x) == 0)


    def test_6(self):
        # 1-D numpy array
        x = sp.array([1.0, 2.0, 3.0, 4.0])
        assert (calcDim(x) == 0)


    def test_7(self):
        # repeated row
        x = sp.array([[1.0, 2.0],
                      [1.0, 2.0]])
        assert (calcDim(x) == 0)


#class Test1D:


class TestAR:

    def test_NH3(self):
        # N2 + 3H2 -> 2NH3

        stoich_mat = sp.array([[-1.0, -3.0, 2.0]]).T
        Cf0 = sp.array([1.0, 1.0, 0.0])

        Vs = artools.stoich_subspace(Cf0, stoich_mat)['all_Cs']

        assert (calcDim(Vs) == 1)


    def test_VDV_2D(self):
        # A -> B -> C

        stoich_mat = sp.array([[-1.0, 0.0],
                               [1.0, -1.0],
                               [0.0, 1.0]])

        Cf0 = sp.array([1.0, 0.0, 0.0])

        Vs = artools.stoich_subspace(Cf0, stoich_mat)['all_Cs']

        assert (calcDim(Vs) == 2)


    def test_VDV_3D(self):
        # A -> B -> C
        # 2A -> D

        stoich_mat = sp.array([[-1.0, 0.0, -2.0],
                               [1.0, -1.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])

        Cf0 = sp.array([1.0, 0.0, 0.0, 0.0])

        Vs = artools.stoich_subspace(Cf0, stoich_mat)['all_Cs']

        assert (calcDim(Vs) == 3)


class TestRand:

    def test_1a(self):

        Xs = sp.rand(10, 1)
        assert calcDim(Xs) == 0


    def test_1b(self):

        Xs = sp.rand(1, 10)
        assert calcDim(Xs) == 0


    def test_2(self):

        Xs = sp.rand(10, 2)
        assert calcDim(Xs) == 2


    def test_3(self):

        Xs = sp.rand(10, 3)
        assert calcDim(Xs) == 3


    def test_4(self):

        Xs = sp.rand(10, 4)
        assert calcDim(Xs) == 4
