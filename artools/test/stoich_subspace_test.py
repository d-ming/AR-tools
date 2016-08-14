import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp


def test_1():
    # single feed, as a 0-D array
    # NB: test is incomplete still

    # A -> B -> C
    # 2A -> D
    Cf = sp.array([1., 0, 0, 0])
    stoich_mat = sp.array([[-1., 0, -2],
                           [1, -1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    S = artools.stoich_subspace(Cf, stoich_mat)
    print S


def test_2():
    # multiple feeds in a list, as 0-D arrays
    # NB: test is incomplete still
    Cf1 = sp.array([1., 0, 0, 0])
    Cf2 = sp.array([1., 1., 0, 0])

    feed_list = [Cf1, Cf2]

    stoich_mat = sp.array([[-1., 0, -2],
                           [1, -1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    S = artools.stoich_subspace(feed_list, stoich_mat)
    print S


def test_3():
    # multiple feeds in a list, as row vectors
    # NB: test is incomplete still
    Cf1 = sp.array([[1., 0, 0, 0]])
    Cf2 = sp.array([[1., 1., 0, 0]])

    feed_list = [Cf1, Cf2]

    stoich_mat = sp.array([[-1., 0, -2],
                           [1, -1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    S = artools.stoich_subspace(feed_list, stoich_mat)
    print S


def test_4():
    # multiple feeds in a list, as column vectors
    # NB: test is incomplete still
    Cf1 = sp.array([[1., 0, 0, 0]]).T
    Cf2 = sp.array([[1., 1., 0, 0]]).T

    feed_list = [Cf1, Cf2]

    stoich_mat = sp.array([[-1., 0, -2],
                           [1, -1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    S = artools.stoich_subspace(feed_list, stoich_mat)
    print S


def test_5():
    # multiple feeds in a 2-D array
    # NB: test is incomplete still
    Cf1 = sp.array([[1., 0, 0, 0]])
    Cf2 = sp.array([[1., 1., 0, 0]])

    feeds = sp.vstack([Cf1, Cf2])

    stoich_mat = sp.array([[-1., 0, -2],
                           [1, -1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    S = artools.stoich_subspace(feeds, stoich_mat)
#    print S

# test 2-D system

# test for single reaction, multiple feeds

# test for single reaction, single feed

# test incompatible size feed and stoichiometric matrix

# test reversible reactions
test_1()
