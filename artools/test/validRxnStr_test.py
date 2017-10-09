
import sys
sys.path.append('../')
from artools import validRxnStr


class TestNormal:

    def test_1(self):
        assert (validRxnStr('A -> B') is True)


class TestMultiArrow:

    def test_1(self):
        assert (validRxnStr('A -> B -> C') is False)


    def test_2(self):
        assert (validRxnStr('A -> B -> C -> D') is False)


class TestNoArrow:

    def test_1(self):
        assert (validRxnStr('A + B') is False)
