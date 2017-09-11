import sys
sys.path.append('../')
from artools import hasRedundantRxns, genStoichMat

import scipy as sp
import pytest


class TestNormal:

    def test_1(self):
        A = sp.array([[-1.,  0., -1.],
                      [ 1., -1.,  0.],
                      [ 0.,  1.,  1.]])
        assert (hasRedundantRxns(A) is True)


    def test_2(self):
        A, d = genStoichMat(["N2 + 3*H2 -> 2*NH3",
                             "2*NH3 -> 3*H2 + N2"])
        assert (hasRedundantRxns(A) is True)


    def test_3(self):
        A, d = genStoichMat(["  A -> B",
                             "  B -> C",
                             "2*A -> D"])
        assert (hasRedundantRxns(A) is False)


class TestSingleRxn:

    def test_1(self):
        A = sp.array([[-1.],
                      [ 1.],
                      [ 0.]])
        assert (hasRedundantRxns(A) is False)


    def test_2(self):
        A, d = genStoichMat(["N2 + 3*H2 -> 2*NH3"])
        assert (hasRedundantRxns(A) is False)
