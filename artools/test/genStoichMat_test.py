import sys
sys.path.append('../')
import artools
#from artools import calcDim, stoichSubspace

import scipy as sp


class TestRand:

    def test_0a(self):

        rxn_str = ["A -> B",
                   "B -> C",
                   "2*A -> D"]

        artools.genStoichMat(rxn_str)
        
        assert True
