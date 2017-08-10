import sys
sys.path.append('../')
from artools import same_rows

import scipy as sp


class TestSame:

    def test_1(self):
        # identical matrices
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        assert (same_rows(A, B) is True)


    def test_2(self):
        # same rows, different order
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = sp.array([[1., 0, 0],
                      [0, 1, 3],
                      [0, 1, 2]])

        assert (same_rows(A, B) is True)


    def test_3(self):
        # 1-D rows
        A = sp.array([[1., 0, 0]])

        B = sp.array([[1., 0, 0]])

        assert (same_rows(A, B) is True)


    def test_4(self):
        # identical copies
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = A

        assert (same_rows(A, B) is True)


class TestElements:

    def test_1(self):
        # same shape, different elements
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = sp.array([[1., 0, 0],
                      [1, 0, 0],
                      [1, 0, 0]])

        assert (same_rows(A, B) is False)


    def test_2(self):
        # same shape, different elements
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = sp.array([[0., 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])

        assert (same_rows(A, B) is False)


    def test_3(self):
        # same shape, just transposed
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = A.T

        assert (same_rows(A, B) is False)


    def test_4(self):
        # 0-D numpy arrays, with different elements
        A = sp.array([1., 0, 0])

        B = sp.array([1, 2, -3])

        assert (same_rows(A, B) is False)


    def test_5(self):
        # same shape, just the negative of one another
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = -A

        assert (same_rows(A, B) is False)


class TestShape:

    def test_1(self):
        # different shapes
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = sp.array([[1., 0, 0],
                      [0, 1, 2]])

        assert (same_rows(A, B) is False)


    def test_2(self):
        # different shapes
        A = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        B = sp.array([[1., 0],
                      [0, 2],
                      [0, 3]])

        assert (same_rows(A, B) is False)


    def test_3(self):
        # different shapes
        A = sp.array([[1., 0],
                      [0, 1],
                      [0, 1]])

        B = sp.array([[1., 0, 0],
                      [0, 1, 2],
                      [0, 1, 3]])

        assert (same_rows(A, B) is False)


    def test_4(self):
        # both column vectors, different elements
        A = sp.array([[1., 0, 0, 0, 0]]).T

        B = sp.array([[1., 0, 0, 1, 0]]).T

        assert (same_rows(A, B) is False)


    def test_5(self):
        # both column vectors, same elements
        A = sp.array([[1., 2, 3, 4, 5]]).T

        B = sp.array([[1., 2, 3, 4, 5]]).T

        assert (same_rows(A, B) is True)


    def test_6(self):
        # both row vectors, different elements
        A = sp.array([[1., 0, 0, 0, 0]])

        B = sp.array([[1., 0, 0, 1, 0]])

        assert (same_rows(A, B) is False)


    def test_7(self):
        # both row vectors, same elements
        A = sp.array([[1., 0, 0, 0, 0]])

        B = sp.array([[1., 0, 0, 0, 0]])

        assert (same_rows(A, B) is True)


    def test_8(self):
        # one row vectors, one column vector
        A = sp.array([[1., 0, 0, 0, 0]])

        B = sp.array([[1., 0, 0, 0, 0]]).T

        assert (same_rows(A, B) is False)


class TestArrays:

    def test_1(self):
        # both 0-D arrays
        A = sp.array([1., 2, 3, 4, 5])
        B = sp.array([1., 2, 3, 4, 5])

        assert (same_rows(A, B) is True)


    def test_2(self):
        # A is a row vector, B is a 0-D array
        A = sp.array([[1., 2, 3, 4, 5]])
        B = sp.array([1., 2, 3, 4, 5])

        assert (same_rows(A, B) is False)


    def test_3(self):
        # A is a column vector, B is a 0-D array
        A = sp.array([[1., 2, 3, 4, 5]]).T
        B = sp.array([1., 2, 3, 4, 5])

        assert (same_rows(A, B) is False)
