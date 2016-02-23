import sys
sys.path.append('artools')
import artools
artools = reload(artools)

import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import scipy.optimize


def tmp_fn(xi):
    d = sp.dot(A, xi) - b
    # ensure point actually lies within region and not just on the
    # boundary
    tmp_ks = sp.nonzero(d >= -1e-10)
    # print sum(d[tmp_ks])    #sum of errors

    # return max(d)
    return sum(d[tmp_ks])

stoich_mat = -sp.array([[-1, -1, 1.], [-2., 0, 1]]).T
Cf = sp.array([1., 1., 0.2])



A = -stoich_mat
b = Cf

v = artools.con2vert(A, b)
print v

c = scipy.linalg.lstsq(A, b)[0]
print artools.in_region(c, A, b)

fig = artools.plot_hplanes(-stoich_mat, b, lims=(-2.0, 2.0))
ax = fig.gca()
ax.hold(True)

ax.plot(c[0], c[1], "bo")
ax.plot(v[0,0], v[0,1], "ro")
ax.plot(v[1,0], v[1,1], "ro")
ax.plot(v[2,0], v[2,1], "ro")

plt.show(fig)