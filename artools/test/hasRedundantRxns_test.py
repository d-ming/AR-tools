import scipy as sp
import pytest

import sys
sys.path.append('../')
from artools import hasRedundantRxns

import pytest


class Test1D:

    def test_1(self):
        # a single column
        stoich_mat = sp.array([[-1.0, -3.0, 2.0]]).T

        assert (hasRedundantRxns(stoich_mat) is False)
