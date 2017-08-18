import scipy as sp
import pytest

import sys
sys.path.append('../')
import artools
artools = reload(artools)

from artools import stoich_S_1D, stoich_S_nD, stoichSubspace, same_rows


class TestStd:

    def test_1(self):
        # 2-D system
        # A -> B -> C

        # 0-D array feed
        Cf0 = sp.array([1., 0, 0])

        stoich_mat = sp.array([[-1., 0],
                               [1, -1],
                               [0, 1]])

        Cs, Es = stoich_S_nD(Cf0, stoich_mat)

        Cs_ref = sp.array([[1., 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        Es_ref = sp.array([[0., 0],
                           [1, 0],
                           [1, 1]])

        assert(same_rows(Es, Es_ref))
        assert(same_rows(Cs, Cs_ref))


    def test_2(self):
        # 3-D van de Vusse system
        # A -> B -> C
        # 2A -> D

        # 0-D array feed
        Cf0 = sp.array([1., 0, 0, 0])

        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        Cs, Es = stoich_S_nD(Cf0, stoich_mat)

        Cs_ref = sp.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0.5]])

        Es_ref = sp.array([[0, 0, 0.5],
                           [1, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0]])

        assert(same_rows(Es, Es_ref))
        assert(same_rows(Cs, Cs_ref))


    def test_3(self):
        # 3-D van de Vusse system
        # A -> B -> C
        # 2A -> D

        # row vector feed
        Cf0 = sp.array([[1., 0, 0, 0]])

        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        Cs, Es = stoich_S_nD(Cf0, stoich_mat)

        Cs_ref = sp.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0.5]])

        Es_ref = sp.array([[0, 0, 0.5],
                           [1, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0]])

        assert(same_rows(Es, Es_ref))
        assert(same_rows(Cs, Cs_ref))


    def test_4(self):
        # 3-D van de Vusse system
        # A -> B -> C
        # 2A -> D

        # column vector feed
        Cf0 = sp.array([[1., 0, 0, 0]]).T

        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        Cs, Es = stoich_S_nD(Cf0, stoich_mat)

        Cs_ref = sp.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0.5]])

        Es_ref = sp.array([[0, 0, 0.5],
                           [1, 1, 0],
                           [0, 0, 0],
                           [1, 0, 0]])

        assert(same_rows(Es, Es_ref))
        assert(same_rows(Cs, Cs_ref))


class TestNegative:

    def test_1(self):
        # 2-D system
        # A -> B -> C

        # Test negative conentrations
        Cf0 = sp.array([-1., 0, 0])

        stoich_mat = sp.array([[-1., 0],
                               [1, -1],
                               [0, 1]])

        with pytest.raises(Exception):
            stoich_S_nD(Cf0, stoich_mat)


    def test_2(self):
        # 2-D system
        # A -> B -> C

        # Test negative zero
        Cf0 = sp.array([1., -0, -0])

        stoich_mat = sp.array([[-1., 0],
                               [1, -1],
                               [0, 1]])

        Cs, Es = stoich_S_nD(Cf0, stoich_mat)

        Cs_ref = sp.array([[1., 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        Es_ref = sp.array([[0., 0],
                           [1, 0],
                           [1, 1]])

        assert(same_rows(Es, Es_ref))
        assert(same_rows(Cs, Cs_ref))
