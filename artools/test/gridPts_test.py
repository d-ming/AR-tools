import sys
sys.path.append('../')
from artools import gridPts, sameRows
import scipy as sp


class TestNormal:

    def test_1(self):
        xs = gridPts(2, [0., 1., 2., 3.5])

        xs_ref = sp.array([[ 0. ,  2. ],
                           [ 0. ,  3.5],
                           [ 1. ,  2. ],
                           [ 1. ,  3.5]])

        assert sameRows(xs, xs_ref) == True


    def test_2(self):
        xs = gridPts(2, [0., 1., 1., 2., 3., 4.])

        xs_ref = sp.array([[ 0.,  1.,  3.],
                           [ 0.,  1.,  4.],
                           [ 0.,  2.,  3.],
                           [ 0.,  2.,  4.],
                           [ 1.,  1.,  3.],
                           [ 1.,  1.,  4.],
                           [ 1.,  2.,  3.],
                           [ 1.,  2.,  4.]])

        assert sameRows(xs, xs_ref) == True
