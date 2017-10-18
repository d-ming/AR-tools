
import sys
sys.path.append('../')
from artools import isOdd


class TestNormal:

    def test_1(self):
        assert isOdd(2) == False

    def test_2(self):
        assert isOdd(201) == True


class TestFloat:

    def test_1(self):
        assert isOdd(3.0) == True

    def test_2(self):
        assert isOdd(3.01) == True
