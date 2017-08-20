import sys
sys.path.append('../../artools')
import artools
artools = reload(artools)

import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import scipy.optimize


def con2vert(A, b):
    '''
    Compute the V-representation of a convex polytope from a set of hyperplane
    constraints. Solve the vertex enumeration problem given inequalities of the
    form A*x <= b

    Parameters:
        A

        b

    Returns:
        Vs  (L x d) array. Each row in Vs represents an extreme point
            of the convex polytope described by A*x <= b.

    Method adapted from Michael Kelder's con2vert() MATLAB function
    http://www.mathworks.com/matlabcentral/fileexchange/7894-con2vert-constraints-to-vertices
    '''

    # check if b is a column vector with ndim=2, or (L,) array with ndim=1 only
    if b.ndim == 2:
        b = b.flatten()

    # attempt to find an interior point in the feasible region
    c = scipy.linalg.lstsq(A, b)[0]

    # if c is out of the region or on the polytope boundary, try to find a new
    # c
    num_tries = 0
    while artools.outRegion(c, A, b) or sp.any(sp.dot(A, c) - b == 0.0):

        num_tries += 1
        if num_tries > 20:
            raise Exception("con2vert() failed to find an interior point"
                            "after 20 tries. Perhaps your constraints are"
                            "badly formed or the region is unbounded.")

        def tmp_fn(xi):
            # find the Chebyshev centre, xc, of the polytope (the
            # largest inscribed ball within the polytope with centre at xc.)

            d = sp.dot(A, xi) - b
            # ensure point actually lies within region and not just on the
            # boundary
            tmp_ks = sp.nonzero(d >= -1e-6)
            # print sum(d[tmp_ks])    #sum of errors

            # return max(d)
            return sum(d[tmp_ks])

        # print "c is not in the interior, need to solve for interior point!
        # %f" % (tmp_fn(c))

        # ignore output message
        c_guess = sp.rand(A.shape[1])
        solver_result = scipy.optimize.fmin(tmp_fn, c_guess, disp=False)
        c = solver_result

        plt.plot(c[0], c[1], "ks")

    # calculate D matrix?
    b_tmp = b - sp.dot(A, c)  # b_tmp is like a difference vector?
    D = A / b_tmp[:, None]

    # find indices of convex hull belonging to D?
    k = scipy.spatial.ConvexHull(D).simplices

    # Generate some kind of offset list of vertices offset from c vector
    G = sp.zeros((len(k), D.shape[1]))
    for idx in range(0, len(k)):

        # F is a matrix with rows beloning to the indices of k referencing
        # rows in matrix D??
        F = D[k[idx, :], :]

        # f is simply an nx1 column vector of ones?
        f = sp.ones((F.shape[0], 1))

        # solve the least squares problem F\f in MATLAB notation for a vector
        # that becomes a row in matrix G?
        G[idx, :] = scipy.linalg.lstsq(F, f)[0].T

    # find vertices from vi = c + Gi
    Vs = G + sp.tile(c.T, (G.shape[0], 1))
    Vs = artools.uniqueRows(Vs)[0]

    return Vs


# this combination of stoich_mat and Cf produces an initial guess that is on
# the boubdary of the feasible region and not technically in the region.
stoich_mat = -sp.array([[-1, -1, 1.], [-1., 0, 1]]).T
Cf = sp.array([1., 1., 0.2])

# this combination works fine
#stoich_mat = -sp.array([[-1, -1, 1.], [-2., 0, 1]]).T
#Cf = sp.array([1., 1., 0.2])

A = -stoich_mat
b = Cf
c = scipy.linalg.lstsq(A, b)[0]

fig = artools.plot_hplanes(-stoich_mat, b, lims=(-2.0, 2.0))
ax = fig.gca()
ax.hold(True)

v = con2vert(A, b)
print v

print artools.inRegion(c, A, b)



ax.plot(c[0], c[1], "bo")
ax.plot(v[0, 0], v[0, 1], "ro")
ax.plot(v[1, 0], v[1, 1], "ro")
ax.plot(v[2, 0], v[2, 1], "ro")

plt.show(fig)
