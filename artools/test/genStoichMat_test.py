from __future__ import print_function
import sys
sys.path.append('../')
from artools import genStoichMat, sameRows

import scipy as sp


def equivalentDictionaries(x, y):
    """
    Helper function to test if dictionaries x and y share the same keys and
    values OVERALL AS A SET, but not necessarily that the key-value pairs are
    the same between x and y.
    """

    for xi in x.values():
        if xi not in y.values():
            return False
    for yi in y.values():
        if yi not in x.values():
            return False

    for xi in x.keys():
        if xi not in y.keys():
            return False
    for yi in y.keys():
        if yi not in x.keys():
            return False

    return True


class TestTuple:

    def test_1(self):

        # reactions written in tuple format
        rxn_str = ("A -> B",)

        A, d = genStoichMat(rxn_str)
        A_ref = sp.array([[-1.0],
                          [1.0]])
        d_ref = {'A':0,
                 'B':1}

        assert (sameRows(A, A_ref) and equivalentDictionaries(d, d_ref) is True)


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
        d_ref = {'A':0,
                 'B':1,
                 'C':2,
                 'D':3}

        assert (sameRows(A, A_ref) and equivalentDictionaries(d, d_ref) is True)


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
        d_ref = {'B':0,
                 'E':1,
                 'T':2,
                 'X':3,
                 'D':4,
                 'H':5}

        A, d = genStoichMat(rxns)

        assert (sameRows(A, A_ref) and equivalentDictionaries(d, d_ref) is True)


    def test_1(self):

        rxn_str = ["A -> B"]

        A, d = genStoichMat(rxn_str)
        A_ref = sp.array([[-1.0],
                          [1.0]])
        d_ref = {'A':0,
                 'B':1}

        assert (sameRows(A, A_ref) and equivalentDictionaries(d, d_ref) is True)


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
        d_ref = {'A':0,
                 'B':1,
                 'C':2,
                 'D':3,
                 'E':4,
                 'F':5}

        assert (sameRows(A, A_ref) and equivalentDictionaries(d, d_ref) is True)


class TestRealistic:

    def test_NH3_1(self):
        rxn = ["N2 + 3*H2 -> 2*NH3"]

        A_ref = sp.array([[-1.0, -3.0, 2.0]]).T
        d_ref = {'N2':0,
                 'H2':1,
                 'NH3': 2}
        A, d = genStoichMat(rxn)

        assert (sameRows(A, A_ref) and equivalentDictionaries(d, d_ref) is True)


    def test_CH4_reform(self):
        rxns = ['CH4 + H2O -> CO + 3*H2',
                'CO + H2O -> CO2 + H2']

        A_ref = sp.array([[-1.0, 0.0],
                          [-1.0, -1.0],
                          [1.0, -1.0],
                          [3.0, 1.0],
                          [0.0, 1.0]])
        d_ref = {'CH4':0,
                 'H2O':1,
                 'CO':2,
                 'H2':3,
                 'CO2':4}
        A, d = genStoichMat(rxns)

        assert (sameRows(A, A_ref) and equivalentDictionaries(d, d_ref) is True)


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
        d_ref = {'C':0,
                 'O2':1,
                 'CO2':2,
                 'CO':3,
                 'H2O':4,
                 'H2':5,
                 'CH4':6}
        A, d = genStoichMat(rxns)

        assert (sameRows(A, A_ref) and equivalentDictionaries(d, d_ref) is True)
