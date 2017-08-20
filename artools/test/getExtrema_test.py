import scipy as sp
import pytest

import sys
sys.path.append('../')
import artools
artools = reload(artools)

from artools import getExtrema, sameRows

class TestMatrixFormat:

    def test_1(self):
        Xs = sp.array([[1.0, 2.0, 3.0],
                       [2.0, -1.0, 3.0]])

        bounds_ref = sp.array([[1.0, -1.0, 3.0],
                               [2.0, 2.0, 3.0]])

        bounds = getExtrema(Xs)

        assert (sameRows(bounds, bounds_ref) is True)


    def test_2(self):
        Xs = sp.array([[1.0, 2.0, 3.0],
                       [2.0, -1.0, 3.0]])

        bounds_ref = sp.array([[1.0, -1.0, 3.0],
                               [2.0, 2.0, 3.0]])

        bounds = getExtrema(Xs, axis=0)

        assert (sameRows(bounds, bounds_ref) is True)


    def test_3(self):
        # test along rows instead of columns

        Xs = sp.array([[1.0, 2.0, 3.0],
                       [2.0, -1.0, 3.0]])

        bounds_ref = sp.array([[1.0, -1.0],
                               [3.0, 3.0]])

        bounds = getExtrema(Xs, axis=1)

        assert (sameRows(bounds, bounds_ref) is True)


class TestListFormat:

    def test_1(self):
        Xs_1 = sp.array([[1.0, 2.0, 3.0],
                        [2.0, -1.0, 3.0]])

        Xs_2 = sp.array([[-1.0, -4.5, 6.5],
                       [0.0, 1.0, -10.0]])

        Xs_list = [Xs_1, Xs_2]

        bounds_ref = sp.array([[-1.0, -4.5, -10.0],
                               [2.0, 2.0, 6.5]])

        bounds = getExtrema(Xs_list)

        assert (sameRows(bounds, bounds_ref) is True)


    def test_2(self):
        Xs_1 = sp.array([[1.0, 2.0, 3.0],
                        [2.0, -1.0, 3.0]])

        Xs_2 = sp.array([[-1.0, -4.5, 6.5],
                       [0.0, 1.0, -10.0]])

        Xs_list = [Xs_1, Xs_2, Xs_2]

        bounds_ref = sp.array([[-1.0, -4.5, -10.0],
                               [2.0, 2.0, 6.5]])

        bounds = getExtrema(Xs_list)

        assert (sameRows(bounds, bounds_ref) is True)


    def test_3(self):
        # test along rows instead of columns
        
        Xs_1 = sp.array([[1.0, 2.0, 3.0],
                        [2.0, -1.0, 3.0]])

        Xs_2 = sp.array([[-1.0, -4.5, 6.5],
                       [0.0, 1.0, -10.0]])

        Xs_list = [Xs_1, Xs_2, Xs_2]

        bounds_ref = sp.array([[1.0, -1.0, -4.5, -10.0, -4.5, -10.0],
                               [3.0, 3.0, 6.5, 1.0, 6.5, 1.0]])

        bounds = getExtrema(Xs_list, axis=1)

        assert (sameRows(bounds, bounds_ref) is True)


class Test1D:

    def test_1(self):
        Xs = sp.array([[1.0, 2.0, 3.0]])

        bounds_ref = sp.array([[1.0, 2.0, 3.0],
                                [1.0, 2.0, 3.0]])

        bounds = getExtrema(Xs)

        assert (sameRows(bounds, bounds_ref) is True)
