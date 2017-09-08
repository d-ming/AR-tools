from __future__ import print_function
import sys
sys.path.append('../')
from artools import genStoichMat, sameRows

import scipy as sp

class TestTuple:

    def test_1(self):

        # reactions written in tuple format
        rxn_str = ("A -> B",)

        A, d = genStoichMat(rxn_str)
        A_ref = sp.array([[-1.0],
                          [1.0]])

        assert (sameRows(A, A_ref) is True)


class TestNormal:

    def test_vdv3D(self):

        rxn_str = ["A -> B",
                   "B -> C",
                   "2*A -> D"]

        A, d = genStoichMat(rxn_str)
        A_ref = sp.array([[-1., 0, -2],
                          [1, -1, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        assert (sameRows(A, A_ref) is True)


    def test_BTX(self):
        rxns = ['B + 0.5*E -> T',
                'T + 0.5*E -> X',
                '2*B -> D + H']

        A_ref = sp.array([[-1.0, 0.0, -2.0],
                          [-0.5, -0.5, 0.0],
                          [1.0, -1.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0]])

        A, d = genStoichMat(rxns)

        assert (sameRows(A, A_ref) is True)


    def test_1(self):

        rxn_str = ["A -> B"]

        A, d = genStoichMat(rxn_str)
        A_ref = sp.array([[-1.0],
                          [1.0]])

        assert (sameRows(A, A_ref) is True)


    def test_3(self):

        rxn_str = ['A + 2*B -> 1.5*C',
                   'A + C -> 0.5*D',
                   'C + 3.2*D -> E + 0.1*F']

        A, d = genStoichMat(rxn_str)
        A_ref = sp.array([[-1., -1, 0],
                          [-2, 0, 0],
                          [1.5, -1, -1],
                          [0, 0.5, -3.2],
                          [0, 0, 1],
                          [0, 0, 0.1]])

        assert (sameRows(A, A_ref) is True)


class TestRealistic:

    def test_NH3_1(self):
        rxn = ["N2 + 3*H2 -> 2*NH3"]

        A_ref = sp.array([[-1.0, -3.0, 2.0]]).T
        A, d = genStoichMat(rxn)

        assert (sameRows(A, A_ref) is True)


    def test_CH4_reform(self):
        rxns = ['CH4 + H2O -> CO + 3*H2',
                'CO + H2O -> CO2 + H2']

        stoich_mat = sp.array([[-1.0, 0.0],
                               [-1.0, -1.0],
                               [1.0, -1.0],
                               [3.0, 1.0],
                               [0.0, 1.0]])
        A, d = genStoichMat(rxns)

        assert (sameRows(stoich_mat, A) is True)


    def test_UCG_1(self):
        rxns = ['C + O2 -> CO2',
                'C + CO2 -> 2*CO',
                'CO + 0.5*O2 -> CO2',
                'C + H2O -> CO + H2',
                'CO + H2O -> CO2 + H2',
                'CO2 + H2 -> CO + H2O',
                'C + 2*H2 -> CH4',
                'CH4 + H2O -> CO + 3*H2',
                'CO + 3*H2 -> CH4 + H2O']

        A_ref = sp.array([[-1., -1, 0, -1, 0, 0, -1, 0, 0],
                          [-1, 0, -0.5, 0, 0, 0, 0, 0, 0],
                          [1, -1, 1, 0, 1, -1, 0, 0, 0],
                          [0, 2, -1, 1, -1, 1, 0, 1, -1],
                          [0, 0, 0, -1, -1, 1, 0, -1, 1],
                          [0, 0, 0, 1, 1, -1, -2, 3, -3],
                          [0, 0, 0, 0, 0, 0, 1, -1, 1]])
        A, d = genStoichMat(rxns)

        assert (sameRows(A, A_ref) is True)
