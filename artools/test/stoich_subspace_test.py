import sys
sys.path.append('../')
import artools
artools = reload(artools)

import scipy as sp

# content of test_assert1.py
def f():
    return 3

def test_function():
    assert f() == 3
    
    
def test_1():
    Cf0 = sp.array([1., 0, 0, 0])
    
    