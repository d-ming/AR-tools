import sys
sys.path.append('../')
import artools
#from artools import calcDim, stoichSubspace

import scipy as sp


class TestNormal:

    def test_1(self):

        rxn_str = ["A -> B"]

        A = artools.genStoichMat(rxn_str)
        A_ref = sp.array([[-1.0],
                          [1.0]])

        assert (artools.sameRows(A.T, A_ref.T) is True)


    def test_2(self):

        rxn_str = ["A -> B",
                   "B -> C",
                   "2*A -> D"]

        A = artools.genStoichMat(rxn_str)
        A_ref = sp.array([[-1., 0, -2],
                          [1, -1, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        assert (artools.sameRows(A.T, A_ref.T) is True)


    def test_3(self):

        rxn_str = ['A + 2*B -> 1.5*C',
                   'A + C -> 0.5*D',
                   'C + 3.2*D -> E + 0.1*F']

        A = artools.genStoichMat(rxn_str)
        A_ref = sp.array([[-1., -1, 0],
                          [-2, 0, 0],
                          [1.5, -1, -1],
                          [0, 0.5, -3.2],
                          [0, 0, 1],
                          [0, 0, 0.1]])

        assert (artools.sameRows(A.T, A_ref.T) is True)
