import scipy as sp
import pytest

import sys
sys.path.append('../')
from artools import stoich_S_1D, stoich_S_nD, stoichSubspace, sameRows


class TestSingleFeed:

    def test_3D_1(self):
        # single feed, as a 0-D array

        # A -> B -> C
        # 2A -> D
        Cf = sp.array([1., 0, 0, 0])
        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        S = stoichSubspace(Cf, stoich_mat)
        Cs = S["all_Cs"]
        Es = S["all_Es"]

        Cs_ref = sp.array([[1., 0, 0, 0],
                           [0, 0, 0, 0.5],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])

        Es_ref = sp.array([[0, 0, 0],
                           [0, 0, 0.5],
                           [1, 0, 0],
                           [1, 1, 0]])

        assert (sameRows(Cs, Cs_ref) is True)
        assert (sameRows(Es, Es_ref) is True)


    def test_3D_2(self):
        # single feed, as a 0-D array, in packed into a one-element list

        # A -> B -> C
        # 2A -> D
        Cf = sp.array([1., 0, 0, 0])
        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        S = stoichSubspace([Cf], stoich_mat)
        Cs = S["all_Cs"]
        Es = S["all_Es"]

        Cs_ref = sp.array([[1., 0, 0, 0],
                           [0, 0, 0, 0.5],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])

        Es_ref = sp.array([[0, 0, 0],
                           [0, 0, 0.5],
                           [1, 0, 0],
                           [1, 1, 0]])

        assert (sameRows(Cs, Cs_ref) is True)
        assert (sameRows(Es, Es_ref) is True)


class TestMultiFeed:

    def test_3D_1(self):
        # multiple feeds in a list, as 0-D arrays

        Cf1 = sp.array([1., 0, 0, 0])
        Cf2 = sp.array([1., 1., 0, 0])

        feed_list = [Cf1, Cf2]

        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        S = stoichSubspace(feed_list, stoich_mat)

        Cs1 = S["all_Cs"][0]
        Cs2 = S["all_Cs"][1]
        Es1 = S["all_Es"][0]
        Es2 = S["all_Es"][1]
        Es_bounds = S["bounds_Es"]
        Cs_bounds = S["bounds_Cs"]

        Cs1_ref = sp.array([[1., 0, 0, 0],
                            [0, 0, 0, 0.5],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])

        Es1_ref = sp.array([[0, 0, 0],
                            [0, 0, 0.5],
                            [1, 0, 0],
                            [1, 1, 0]])

        Cs2_ref = sp.array([[0, 2, 0, 0],
                            [0, 0, 2, 0],
                            [2, 0, 0, 0],
                            [0, 0, 0, 1]])

        Es2_ref = sp.array([[1., 0, 0],
                            [1, 2, 0],
                            [-1, 0, 0],
                            [-1, 0, 1]])

        Cs_bounds_ref = sp.array([[0., 0, 0, 0],
                                  [2, 2, 2, 1]])

        Es_bounds_ref = sp.array([[-1., 0, 0],
                                  [1, 2, 1]])

        assert (sameRows(Cs1, Cs1_ref) is True)
        assert (sameRows(Es1, Es1_ref) is True)
        assert (sameRows(Cs2, Cs2_ref) is True)
        assert (sameRows(Es2, Es2_ref) is True)
        assert (sameRows(Es_bounds, Es_bounds_ref) is True)
        assert (sameRows(Cs_bounds, Cs_bounds_ref) is True)


    def test_3D_2(self):
        # multiple feeds in a list, as row vectors

        Cf1 = sp.array([[1., 0, 0, 0]])
        Cf2 = sp.array([[1., 1., 0, 0]])

        feed_list = [Cf1, Cf2]

        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        S = stoichSubspace(feed_list, stoich_mat)

        Cs1 = S["all_Cs"][0]
        Cs2 = S["all_Cs"][1]
        Es1 = S["all_Es"][0]
        Es2 = S["all_Es"][1]
        Es_bounds = S["bounds_Es"]
        Cs_bounds = S["bounds_Cs"]

        Cs1_ref = sp.array([[1., 0, 0, 0],
                            [0, 0, 0, 0.5],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])

        Es1_ref = sp.array([[0, 0, 0],
                            [0, 0, 0.5],
                            [1, 0, 0],
                            [1, 1, 0]])

        Cs2_ref = sp.array([[0, 2, 0, 0],
                            [0, 0, 2, 0],
                            [2, 0, 0, 0],
                            [0, 0, 0, 1]])

        Es2_ref = sp.array([[1., 0, 0],
                            [1, 2, 0],
                            [-1, 0, 0],
                            [-1, 0, 1]])

        Cs_bounds_ref = sp.array([[0., 0, 0, 0],
                                  [2, 2, 2, 1]])

        Es_bounds_ref = sp.array([[-1., 0, 0],
                                  [1, 2, 1]])

        assert (sameRows(Cs1, Cs1_ref) is True)
        assert (sameRows(Es1, Es1_ref) is True)
        assert (sameRows(Cs2, Cs2_ref) is True)
        assert (sameRows(Es2, Es2_ref) is True)
        assert (sameRows(Es_bounds, Es_bounds_ref) is True)
        assert (sameRows(Cs_bounds, Cs_bounds_ref) is True)


    def test_3D_3(self):
        # multiple feeds in a list, as column vectors

        Cf1 = sp.array([[1., 0, 0, 0]]).T
        Cf2 = sp.array([[1., 1., 0, 0]]).T

        feed_list = [Cf1, Cf2]

        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        S = stoichSubspace(feed_list, stoich_mat)

        Cs1 = S["all_Cs"][0]
        Cs2 = S["all_Cs"][1]
        Es1 = S["all_Es"][0]
        Es2 = S["all_Es"][1]
        Es_bounds = S["bounds_Es"]
        Cs_bounds = S["bounds_Cs"]

        Cs1_ref = sp.array([[1., 0, 0, 0],
                            [0, 0, 0, 0.5],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])

        Es1_ref = sp.array([[0, 0, 0],
                            [0, 0, 0.5],
                            [1, 0, 0],
                            [1, 1, 0]])

        Cs2_ref = sp.array([[0, 2, 0, 0],
                            [0, 0, 2, 0],
                            [2, 0, 0, 0],
                            [0, 0, 0, 1]])

        Es2_ref = sp.array([[1., 0, 0],
                            [1, 2, 0],
                            [-1, 0, 0],
                            [-1, 0, 1]])

        Cs_bounds_ref = sp.array([[0., 0, 0, 0],
                                  [2, 2, 2, 1]])

        Es_bounds_ref = sp.array([[-1., 0, 0],
                                  [1, 2, 1]])

        assert (sameRows(Cs1, Cs1_ref) is True)
        assert (sameRows(Es1, Es1_ref) is True)
        assert (sameRows(Cs2, Cs2_ref) is True)
        assert (sameRows(Es2, Es2_ref) is True)
        assert (sameRows(Es_bounds, Es_bounds_ref) is True)
        assert (sameRows(Cs_bounds, Cs_bounds_ref) is True)


    def test_3D_4(self):
        # multiple feeds in a 2-D array

        Cf1 = sp.array([[1., 0, 0, 0]])
        Cf2 = sp.array([[1., 1., 0, 0]])

        feeds = sp.vstack([Cf1, Cf2])

        stoich_mat = sp.array([[-1., 0, -2],
                               [1, -1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        S = stoichSubspace(feeds, stoich_mat)

        Cs1 = S["all_Cs"][0]
        Cs2 = S["all_Cs"][1]
        Es1 = S["all_Es"][0]
        Es2 = S["all_Es"][1]
        Es_bounds = S["bounds_Es"]
        Cs_bounds = S["bounds_Cs"]

        Cs1_ref = sp.array([[1., 0, 0, 0],
                            [0, 0, 0, 0.5],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]])

        Es1_ref = sp.array([[0, 0, 0],
                            [0, 0, 0.5],
                            [1, 0, 0],
                            [1, 1, 0]])

        Cs2_ref = sp.array([[0, 2, 0, 0],
                            [0, 0, 2, 0],
                            [2, 0, 0, 0],
                            [0, 0, 0, 1]])

        Es2_ref = sp.array([[1., 0, 0],
                            [1, 2, 0],
                            [-1, 0, 0],
                            [-1, 0, 1]])

        Cs_bounds_ref = sp.array([[0., 0, 0, 0],
                                  [2, 2, 2, 1]])

        Es_bounds_ref = sp.array([[-1., 0, 0],
                                  [1, 2, 1]])

        assert (sameRows(Cs1, Cs1_ref) is True)
        assert (sameRows(Es1, Es1_ref) is True)
        assert (sameRows(Cs2, Cs2_ref) is True)
        assert (sameRows(Es2, Es2_ref) is True)
        assert (sameRows(Es_bounds, Es_bounds_ref) is True)
        assert (sameRows(Cs_bounds, Cs_bounds_ref) is True)


    def test_1D_1(self):
        # A + B -> C

        # two feeds
        Cf1 = sp.array([1., 1, 0])
        Cf2 = sp.array([1, 0.5, 0.1])

        feeds = [Cf1, Cf2]

        stoich_mat = sp.array([[-1., -1, 1]]).T

        S = stoichSubspace(feeds, stoich_mat)

        Cs1 = S["all_Cs"][0]
        Cs2 = S["all_Cs"][1]
        Es1 = S["all_Es"][0]
        Es2 = S["all_Es"][1]
        Es_bounds = S["bounds_Es"]
        Cs_bounds = S["bounds_Cs"]

        Cs1_ref = sp.array([[1., 1, 0],
                            [0, 0, 1]])
        Cs2_ref = sp.array([[1., 0.5, 0.1],
                            [0.5, 0, 0.6]])

        Es1_ref = sp.array([[0., 1]]).T
        Es2_ref = sp.array([[0., 0.5]]).T

        Es_bounds_ref = sp.array([[0., 1]]).T
        Cs_bounds_ref = sp.array([[0., 0, 0],
                                  [1, 1, 1]])

        assert (sameRows(Cs1, Cs1_ref) is True)
        assert (sameRows(Es1, Es1_ref) is True)
        assert (sameRows(Cs2, Cs2_ref) is True)
        assert (sameRows(Es2, Es2_ref) is True)
        assert (sameRows(Es_bounds, Es_bounds_ref) is True)
        assert (sameRows(Cs_bounds, Cs_bounds_ref) is True)


    def test_2D_1(self):
        # test 2-D system
        # A -> B -> C

        Cf1 = sp.array([1.0, 0, 0])
        Cf2 = sp.array([1.0, 0.5, 0])

        feeds = [Cf1, Cf2]

        stoich_mat = sp.array([[-1., 0],
                               [1, -1],
                               [0, 1]])

        S = stoichSubspace(feeds, stoich_mat)

        Cs1_ref = sp.array([[1., 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

        Cs2_ref = sp.array([[1.5, 0, 0],
                            [0, 1.5, 0],
                            [0, 0, 1.5]])

        Es1_ref = sp.array([[0., 0],
                            [1, 0],
                            [1, 1]])

        Es2_ref = sp.array([[1., 0],
                            [1, 1.5],
                            [-0.5, 0]])

        Cs_bounds_ref = sp.array([[0.0, 0.0, 0.0],
                                  [1.5, 1.5, 1.5]])

        Es_bounds_ref = sp.array([[-0.5, 0.0],
                                  [1.0, 1.5]])

        Cs1 = S["all_Cs"][0]
        Cs2 = S["all_Cs"][1]
        Es1 = S["all_Es"][0]
        Es2 = S["all_Es"][1]
        Es_bounds = S["bounds_Es"]
        Cs_bounds = S["bounds_Cs"]

        assert (sameRows(Cs1, Cs1_ref) is True)
        assert (sameRows(Cs2, Cs2_ref) is True)
        assert (sameRows(Es1, Es1_ref) is True)
        assert (sameRows(Es2, Es2_ref) is True)
        assert (sameRows(Cs_bounds, Cs_bounds_ref) is True)
        assert (sameRows(Es_bounds, Es_bounds_ref) is True)


def test_steam_reforming_singleFeed_1():
    # methane steam reforming + water-gas shift
    # CH4 + H2O -> CO + 3H2
    # CO + H2O -> CO2 + H2

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
