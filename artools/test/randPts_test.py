
import sys
sys.path.append('../')
from artools import randPts
import pytest


class TestError:

    def test_float_1(self):
        # num pts can't be a float
        with pytest.raises(TypeError):
            randPts(1.5, [0., 1., 0., 1.])

    def test_odd_limits(self):
        # axis limits must have an even number of elements
        with pytest.raises(ValueError):
            randPts(10, [0., 1., 0.])
