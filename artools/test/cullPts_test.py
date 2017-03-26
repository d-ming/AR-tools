import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp


def test_1():
    Cs = sp.array([[0.1, 0],
                   [0.2, 0],
                   [0.3, 0]], dtype=sp.float64)

    Cs_ref = sp.array([[0.1, 0],
                       [0.3, 0]], dtype=sp.float64)

    axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

    Cs_ans = artools.cullPts(Cs, min_dist=0.100, axis_lims=axis_lims)

    assert (artools.same_rows(Cs_ref, Cs_ans) is True)


def test_2():
    Cs = sp.array([[0.1, 0],
                   [0.2, 0],
                   [0.3, 0]], dtype=sp.float64)

    Cs_ref = sp.array([[0.1, 0],
                       [0.3, 0]], dtype=sp.float64)

    axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

    Cs_ans = artools.cullPts(Cs, min_dist=0.1001, axis_lims=axis_lims)

    assert (artools.same_rows(Cs_ref, Cs_ans) is True)


def test_3():
    Cs = sp.array([[0.1, 0],
                   [0.2, 0],
                   [0.3, 0]], dtype=sp.float64)

    Cs_ref = sp.array([[0.1, 0],
                       [0.3, 0]], dtype=sp.float64)

    axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

    Cs_ans = artools.cullPts(Cs, min_dist=0.999, axis_lims=axis_lims)

    assert (artools.same_rows(Cs_ref, Cs_ans) is False)


def test_4():
    Cs = sp.array([[0.1, 0],
                   [0.2, 0],
                   [0.3, 0]], dtype=sp.float64)

    Cs_ref = sp.array([[0.1, 0]], dtype=sp.float64)

    axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

    Cs_ans = artools.cullPts(Cs, min_dist=0.200, axis_lims=axis_lims)

    assert (artools.same_rows(Cs_ref, Cs_ans) is True)


def test_5():
    Cs = sp.array([[0.1, 0],
                   [0.2, 0],
                   [0.3, 0]], dtype=sp.float64)

    Cs_ref = sp.array([[0.1, 0]], dtype=sp.float64)

    axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

    Cs_ans = artools.cullPts(Cs, min_dist=0.500, axis_lims=axis_lims)

    assert (artools.same_rows(Cs_ref, Cs_ans) is True)


def test_6():
    Cs = sp.array([[0.1, 0],
                   [0.2, 0],
                   [0.3, 0]], dtype=sp.float64)

    Cs_ref = sp.array([[0.1, 0],
                       [0.2, 0],
                       [0.3, 0]], dtype=sp.float64)

    axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

    Cs_ans = artools.cullPts(Cs, min_dist=0.05, axis_lims=axis_lims)

    assert (artools.same_rows(Cs_ref, Cs_ans) is True)
