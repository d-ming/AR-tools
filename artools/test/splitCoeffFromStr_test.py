import sys
sys.path.append('../')
from artools import splitCoeffFromStr

class TestNormal:

    def test_1(self):

        coeff, comp = splitCoeffFromStr("2*A")
        assert (coeff == '2' and comp == 'A')


    def test_2(self):

        coeff, comp = splitCoeffFromStr("A")
        assert (coeff == '1' and comp == 'A')


    def test_3(self):

        coeff, comp = splitCoeffFromStr("1.5*H2O")
        assert (coeff == '1.5' and comp == 'H2O')


class TestSpace:

    def test_1(self):

        coeff, comp = splitCoeffFromStr(" 1*H2")
        assert (coeff == '1' and comp == 'H2')


    def test_2(self):

        coeff, comp = splitCoeffFromStr("  3 * A ")
        assert (coeff == '3' and comp == 'A')
