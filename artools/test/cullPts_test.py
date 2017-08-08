import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp


class TestStd:

    def test_1(self):
        Cs = sp.array([[0.1, 0],
                       [0.2, 0],
                       [0.3, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0],
                           [0.3, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.100, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)

    def test_2(self):
        Cs = sp.array([[0.1, 0],
                       [0.2, 0],
                       [0.3, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0],
                           [0.3, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.1001, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)

    def test_3(self):
        Cs = sp.array([[0.1, 0],
                       [0.2, 0],
                       [0.3, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0],
                           [0.3, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.999, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is False)

    def test_4(self):
        Cs = sp.array([[0.1, 0],
                       [0.2, 0],
                       [0.3, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.200, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)

    def test_5(self):
        Cs = sp.array([[0.1, 0],
                       [0.2, 0],
                       [0.3, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.500, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)

    def test_6(self):
        Cs = sp.array([[0.1, 0],
                       [0.2, 0],
                       [0.3, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0],
                           [0.2, 0],
                           [0.3, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.05, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)

    def test_7(self):
        Cs = sp.array([[0.1, 0],
                       [0.2, 0],
                       [0.3, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0],
                           [0.2, 0],
                           [0.3, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.0, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)


class TestDuplicate:

    def test_1(self):
        Cs = sp.array([[0.1, 0],
                       [0.1, 0],
                       [0.1, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.1, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)

    def test_2(self):
        Cs = sp.array([[0.1, 0],
                       [0.1, 0],
                       [0.1, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.0, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)

    def test_3(self):
        Cs = sp.array([[0.1, 0],
                       [0.1, 0],
                       [0.1, 0],
                       [0.2, 0]], dtype=sp.float64)

        Cs_ref = sp.array([[0.1, 0],
                           [0.2, 0]], dtype=sp.float64)

        axis_lims = sp.array([0, 1, 0, 1], dtype=sp.float64)

        Cs_ans = artools.cullPts(Cs, min_dist=0.05, axis_lims=axis_lims)

        assert (artools.same_rows(Cs_ref, Cs_ans) is True)
