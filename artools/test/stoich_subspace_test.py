import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp


def test_stoich_S_M_1():
    # 2-D system
    # A -> B -> C

    # 0-D array feed
    Cf0 = sp.array([1., 0, 0])

    stoich_mat = sp.array([[-1., 0],
                           [1, -1],
                           [0, 1]])

    Cs, Es = artools.stoich_S_M(Cf0, stoich_mat)

    Cs_ref = sp.array([[1., 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

    Es_ref = sp.array([[0., 0],
                       [1, 0],
                       [1, 1]])

    assert(artools.same_rows(Es, Es_ref))
    assert(artools.same_rows(Cs, Cs_ref))


def test_stoich_S_M_2():
    # 3-D van de Vusse system
    # A -> B -> C
    # 2A -> D

    # 0-D array feed
    Cf0 = sp.array([1., 0, 0, 0])

    stoich_mat = sp.array([[-1., 0, -2],
                           [1, -1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    Cs, Es = artools.stoich_S_M(Cf0, stoich_mat)

    Cs_ref = sp.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 0.5]])

    Es_ref = sp.array([[0, 0, 0.5],
                       [1, 1, 0],
                       [0, 0, 0],
                       [1, 0, 0]])

    assert(artools.same_rows(Es, Es_ref))
    assert(artools.same_rows(Cs, Cs_ref))


def test_stoich_S_M_3():
    # 3-D van de Vusse system
    # A -> B -> C
    # 2A -> D

    # row vector feed
    Cf0 = sp.array([[1., 0, 0, 0]])

    stoich_mat = sp.array([[-1., 0, -2],
                           [1, -1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    Cs, Es = artools.stoich_S_M(Cf0, stoich_mat)

    Cs_ref = sp.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 0.5]])

    Es_ref = sp.array([[0, 0, 0.5],
                       [1, 1, 0],
                       [0, 0, 0],
                       [1, 0, 0]])

    assert(artools.same_rows(Es, Es_ref))
    assert(artools.same_rows(Cs, Cs_ref))


def test_stoich_S_M_4():
    # 3-D van de Vusse system
    # A -> B -> C
    # 2A -> D

    # column vector feed
    Cf0 = sp.array([[1., 0, 0, 0]]).T

    stoich_mat = sp.array([[-1., 0, -2],
                           [1, -1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    Cs, Es = artools.stoich_S_M(Cf0, stoich_mat)

    Cs_ref = sp.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 0.5]])

    Es_ref = sp.array([[0, 0, 0.5],
                       [1, 1, 0],
                       [0, 0, 0],
                       [1, 0, 0]])

    assert(artools.same_rows(Es, Es_ref))
    assert(artools.same_rows(Cs, Cs_ref))


def test_stoich_S_S_1():
    # A + B -> C

    # 0-D array feed
    Cf0 = sp.array([1., 1, 0])

    # column vector stoich matrix
    stoich_mat = sp.array([[-1., -1, 1]]).T

    Cs, Es = artools.stoich_S_S(Cf0, stoich_mat)

    Cs_ref = sp.array([[1., 1, 0],
                       [0, 0, 1]])

    Es_ref = sp.array([[0., 1]]).T

    assert(artools.same_rows(Cs, Cs_ref))
    assert(artools.same_rows(Es, Es_ref))


def test_stoich_S_S_2():
    # A + B -> C

    # 0-D array feed
    Cf0 = sp.array([1., 1, 0])

    # row vector stoich matrix
    stoich_mat = sp.array([[-1., -1, 1]])

    Cs, Es = artools.stoich_S_S(Cf0, stoich_mat)

    Cs_ref = sp.array([[1., 1, 0],
                       [0, 0, 1]])

    Es_ref = sp.array([[0., 1]]).T

    assert(artools.same_rows(Cs, Cs_ref))
    assert(artools.same_rows(Es, Es_ref))


def test_stoich_S_S_3():
    # A + B -> C

    # 0-D array feed
    Cf0 = sp.array([1., 1, 0])

    # 0-D array stoich matrix
    stoich_mat = sp.array([-1., -1, 1])

    Cs, Es = artools.stoich_S_S(Cf0, stoich_mat)

    Cs_ref = sp.array([[1., 1, 0],
                       [0, 0, 1]])

    Es_ref = sp.array([[0., 1]]).T

    assert(artools.same_rows(Cs, Cs_ref))
    assert(artools.same_rows(Es, Es_ref))


def test_stoich_S_S_4():
    # A + B -> C

    # column vector feed
    Cf0 = sp.array([[1., 1, 0]]).T

    # column vector stoich matrix
    stoich_mat = sp.array([[-1., -1, 1]]).T

    Cs, Es = artools.stoich_S_S(Cf0, stoich_mat)

    Cs_ref = sp.array([[1., 1, 0],
                       [0, 0, 1]])

    Es_ref = sp.array([[0., 1]]).T

    assert(artools.same_rows(Cs, Cs_ref))
    assert(artools.same_rows(Es, Es_ref))


def test_stoich_S_S_5():
    # A + B -> C

    # row vector feed
    Cf0 = sp.array([[1., 1, 0]])

    # column vector stoich matrix
    stoich_mat = sp.array([[-1., -1, 1]]).T

    Cs, Es = artools.stoich_S_S(Cf0, stoich_mat)

    Cs_ref = sp.array([[1., 1, 0],
                       [0, 0, 1]])

    Es_ref = sp.array([[0., 1]]).T

    assert(artools.same_rows(Cs, Cs_ref))
    assert(artools.same_rows(Es, Es_ref))

#test_stoich_S_S_1()


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
#test_1()
