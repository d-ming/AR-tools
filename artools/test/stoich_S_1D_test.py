import scipy as sp
import pytest

import sys
sys.path.append('../')
from artools import stoich_S_1D, stoich_S_nD, stoichSubspace, sameRows


class TestStd:

    def test_1(self):
        # A + B -> C

        # 0-D array feed
        Cf0 = sp.array([1., 1, 0])

        # column vector stoich matrix
        stoich_mat = sp.array([[-1., -1, 1]]).T

        Cs, Es = stoich_S_1D(Cf0, stoich_mat)

        Cs_ref = sp.array([[1., 1, 0],
                           [0, 0, 1]])

        Es_ref = sp.array([[0., 1]]).T

        assert(sameRows(Cs, Cs_ref))
        assert(sameRows(Es, Es_ref))


    def test_2(self):
        # A + B -> C

        # 0-D array feed
        Cf0 = sp.array([1., 1, 0])

        # row vector stoich matrix
        stoich_mat = sp.array([[-1., -1, 1]])

        Cs, Es = stoich_S_1D(Cf0, stoich_mat)

        Cs_ref = sp.array([[1., 1, 0],
                           [0, 0, 1]])

        Es_ref = sp.array([[0., 1]]).T

        assert(sameRows(Cs, Cs_ref))
        assert(sameRows(Es, Es_ref))


    def test_3(self):
        # A + B -> C

        # 0-D array feed
        Cf0 = sp.array([1., 1, 0])

        # 0-D array stoich matrix
        stoich_mat = sp.array([-1., -1, 1])

        Cs, Es = stoich_S_1D(Cf0, stoich_mat)

        Cs_ref = sp.array([[1., 1, 0],
                           [0, 0, 1]])

        Es_ref = sp.array([[0., 1]]).T

        assert(sameRows(Cs, Cs_ref))
        assert(sameRows(Es, Es_ref))


    def test_4(self):
        # A + B -> C

        # column vector feed
        Cf0 = sp.array([[1., 1, 0]]).T

        # column vector stoich matrix
        stoich_mat = sp.array([[-1., -1, 1]]).T

        Cs, Es = stoich_S_1D(Cf0, stoich_mat)

        Cs_ref = sp.array([[1., 1, 0],
                           [0, 0, 1]])

        Es_ref = sp.array([[0., 1]]).T

        assert(sameRows(Cs, Cs_ref))
        assert(sameRows(Es, Es_ref))


    def test_5(self):
        # A + B -> C

        # row vector feed
        Cf0 = sp.array([[1., 1, 0]])

        # column vector stoich matrix
        stoich_mat = sp.array([[-1., -1, 1]]).T

        Cs, Es = stoich_S_1D(Cf0, stoich_mat)

        Cs_ref = sp.array([[1., 1, 0],
                           [0, 0, 1]])

        Es_ref = sp.array([[0., 1]]).T

        assert(sameRows(Cs, Cs_ref))
        assert(sameRows(Es, Es_ref))


    def test_6(self):
        # A -> B

        # binary system
        Cf0 = sp.array([1., 0])

        stoich_mat = sp.array([[-1., 1]]).T

        Cs, Es = stoich_S_1D(Cf0, stoich_mat)

        Cs_ref = sp.array([[1., 0],
                           [0, 1]])

        Es_ref = sp.array([[0., 1]]).T

        assert (sameRows(Cs, Cs_ref))
        assert (sameRows(Es, Es_ref))


class TestNegative:

    def test_1(self):
        # 2-D system
        # A -> B

        # Test negative conentrations
        Cf0 = sp.array([-1., 0])

        stoich_mat = sp.array([[-1.],
                               [1]])

        with pytest.raises(Exception):
            stoich_S_1D(Cf0, stoich_mat)


    def test_2(self):
        # 2-D system
        # A -> B

        # Test negative zero
        Cf0 = sp.array([1., -0.0])

        stoich_mat = sp.array([[-1.],
                               [1]])

        assert stoich_S_1D(Cf0, stoich_mat)


class TestAR:

    def test_1(self):
        """
        Methane steam reforming + water-gas shift with a single feed
        CH4 + H2O -> CO + 3H2
        CO + H2O -> CO2 + H2
        """

        Cf0 = sp.array([1., 1, 1, 0, 0])

        stoich_mat = sp.array([[-1., 0],
                               [-1, -1],
                               [1, -1],
                               [3, 1],
                               [0, 1]])

        S = stoichSubspace(Cf0, stoich_mat)
        Cs = S["all_Cs"]
        Es = S["all_Es"]

        Cs_ref = sp.array([[0, 0, 2, 3, 0],
                           [1.25, 0.5, 0, 0, 0.75],
                           [1, 1, 1, 0, 0],
                           [1, 0, 0, 1, 1]])

        Es_ref = sp.array([[1, 0],
                           [-0.25, 0.75],
                           [0, 0],
                           [0, 1]])

        assert (sameRows(Cs, Cs_ref) is True)
        assert (sameRows(Es, Es_ref) is True)


# test incompatible size feed and stoichiometric matrix

# test reversible reactions
