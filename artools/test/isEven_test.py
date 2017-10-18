
import sys
sys.path.append('../')
from artools import isEven


class TestNormal:

    def test_1(self):
        assert isEven(2) == True

    def test_2(self):
        assert isEven(201) == False

        
class TestFloat:

    def test_1(self):
        assert isEven(2.0) == True

    def test_2(self):
        assert isEven(2.01) == False
