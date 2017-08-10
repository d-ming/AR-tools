import sys
sys.path.append('../')
from artools import isColVector, isRowVector

import scipy as sp


class TestIsCol:

    def test_1(self):
        # standard 2-D column vector
        A = sp.array([[1., 1, 1]]).T

        assert (isColVector(A) is True)


    def test_2(self):
        # row vector
        A = sp.array([[1., 1, 1]])

        assert (isColVector(A) is False)


    def test_3(self):
        # 0-D array
        A = sp.array([1., 1, 1])

        assert (isColVector(A) is False)


    def test_4(self):
        # A list
        A = [1., 1, 1]

        assert (isColVector(A) is False)


    def test_5(self):
        # A matrix
        A = sp.array([[1., 1, 1],
                      [2, 2, 2]])

        assert (isColVector(A) is False)


class TestIsRow:

    def test_1(self):
        # standard 2-D column vector
        A = sp.array([[1., 1, 1]]).T

        assert (isRowVector(A) is False)


    def test_2(self):
        # row vector
        A = sp.array([[1., 1, 1]])

        assert (isRowVector(A) is True)


    def test_3(self):
        # 0-D array
        A = sp.array([1., 1, 1])

        assert (isRowVector(A) is False)


    def test_4(self):
        # A list
        A = [1., 1, 1]

        assert (isRowVector(A) is False)


    def test_5(self):
        # A matrix
        A = sp.array([[1., 1, 1],
                      [2, 2, 2]])

        assert (isRowVector(A) is False)
