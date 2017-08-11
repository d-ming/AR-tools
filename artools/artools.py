# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

import scipy as sp
import scipy.spatial
import scipy.optimize
import scipy.linalg

import numpy.linalg
import numpy as np

import itertools

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------


def unique_rows(A, tol=1e-13):
    '''
    Find the unique rows of a matrix A given a tolerance

    Parameters:
        A       []

    Returns:
        tuple   []
    '''

    duplicate_ks = []
    for r1 in range(A.shape[0]):
        for r2 in range(r1 + 1, A.shape[0]):
            # check if row 1 is equal to row 2 to within tol
            if sp.all(sp.fabs(A[r1, :] - A[r2, :]) <= tol):
                # only add if row 2 has not already been added from a previous
                # pass
                if r2 not in duplicate_ks:
                    duplicate_ks.append(r2)

    # generate a list of unique indices
    unique_ks = [idx for idx in range(A.shape[0]) if idx not in duplicate_ks]

    # return matrix of unique rows and associated indices
    return (A[unique_ks, :], unique_ks)


def same_rows(A, B):
    # check if A and B are the same shape
    if A.shape != B.shape:
        return False
    else:

        if A.ndim == 2 and (A.shape[0] == 1 or A.shape[1] == 1):
            return np.allclose(A.flatten(), B.flatten())

        # now loop through each row in A and check if the same row exists in B.
        # If not, A and B are not equivalent according to their rows.
        for row_A in A:
            # does row_A exist in B?
            if not any([np.allclose(row_A, row_B) for row_B in B]):
                return False

        return True


def plot_region2d(Vs, ax=None, color="g", alpha=0.5, plot_verts=False):
    '''
    Plot a filled 2D region, similar to MATLAB's fill() function.

    Parameters:
        Vs      (L x d) A numpy array containing the region to be plotted.

        ax      Optional. A matplotlib axis object. In case an exisiting plot
                should be used and plotted over.
                Default value is None, which creates a new figure.

        color   Optional. A matplotlib compatible coolor specificaiton.
                Default value is green "g", or [0, 1, 0].

        alpha   Optional. Alpha (transparency) value for the filled region.
                Default value is 50%.

    Returns:
        fig     A Matplotlib figure object using ax.get_figure().
    '''

    # convert Vs to a scipy array (because fill can't work with marices) with
    # only unique rows
    Vs = sp.array(unique_rows(Vs)[0])

    # find indices of conv hull
    ks = scipy.spatial.ConvexHull(Vs).vertices

    if ax is None:
        fig = plt.figure()
        fig.hold(True)
        ax = fig.gca()

    ax.fill(Vs[ks, 0], Vs[ks, 1], color=color, alpha=alpha)

    if plot_verts:
        ax.plot(Vs[:, 0], Vs[:, 1], 'bo')

    return ax.get_figure()


def plot_region3d(Vs,
                  ax=None,
                  color="g",
                  alpha=0.25,
                  view=(50, 30),
                  plot_verts=False):
    '''
    Plot a filled 3D region, similar to MATLAB's trisurf() function.

    Parameters:
        Vs          (L x d) A numpy array containing the region to be plotted.

        ax          Optional. A matplotlib axis object. In case an exisiting
                    plot should be used and plotted over.
                    Default value is None, which creates a new figure.

        color       Optional. A matplotlib compatible coolor specificaiton.
                    Default value is green "g", or [0, 1, 0].

        alpha       Optional. Alpha (transparency) value for the filled region.
                    Default value is 25%.

        view (2,)   Optional. A tuple specifying the camera view:
                    (camera elevation, camera rotation)
                    Default value is (50, 30)

    Returns:
        fig         A Matplotlib figure object using ax.get_figure().
    '''

    # convert Vs to a numpy array with only unique rows.
    Vs = sp.array(unique_rows(Vs)[0])

    # find indices of conv hull
    simplices = scipy.spatial.ConvexHull(Vs).simplices

    if ax is None:
        fig = plt.figure(figsize=(6, 5))

        fig.hold(True)
        ax = fig.gca(projection='3d')

    if plot_verts:
        ax.scatter(Vs[:, 0], Vs[:, 1], Vs[:, 2], 'bo')

    xs = Vs[:, 0]
    ys = Vs[:, 1]
    zs = Vs[:, 2]
    ax.plot_trisurf(
        mtri.Triangulation(xs, ys, simplices),
        zs,
        color=color,
        alpha=alpha)

    ax.view_init(view[0], view[1])

    return ax.get_figure()


def plot_hplanes(A, b, lims=(0.0, 1.0), ax=None):
    '''
    Plot a set of hyperplane constraints given in A*x <= b format. Only for
    two-dimensional plots.

    Parameters:
        A

        b

        ax      Optional. A matplotlib axis object. In case an exisiting plot
                should be used and plotted over.
                Default value is None, which creates a new figure.

    Returns:
        fig     A Matplotlib figure object using ax.get_figure().
    '''

    # generate new figure if none supplied
    if ax is None:
        fig = plt.figure()
        fig.hold(True)
        ax = fig.gca()

    def y_fn(x, n, b):
        '''Helper function to plot y in terms of x'''
        return (b - n[0] * x) / n[1]

    def x_fn(y, n, b):
        '''Helper function to plot x in terms of y'''
        return (b - n[1] * y) / n[0]

    # limits for plotting
    xl = lims[0]
    xu = lims[1]
    yl = lims[0]
    yu = lims[1]

    # plot based on whether ny = 0 or not
    for i, ni in enumerate(A):
        bi = b[i]
        if ni[1] != 0.0:
            ax.plot([xl, xu], [y_fn(yl, ni, bi), y_fn(yu, ni, bi)], 'k-')
        else:
            ax.plot([x_fn(xl, ni, bi), x_fn(xu, ni, bi)], [yl, yu], 'k-')

    return ax.get_figure()


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
    while out_region(c, A, b) or sp.any(sp.dot(A, c) - b == 0.0):

        plt.plot(c[0], c[1], "ks")

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
    Vs = unique_rows(Vs)[0]

    return Vs


def vert2con(Vs):
    '''
    Compute the H-representation of a set of points (facet enumeration).

    Parameters:
        Vs

    Returns:
        A   (L x d) array. Each row in A represents hyperplane normal.
        b   (L x 1) array. Each element in b represents the hyperpalne
            constant bi

    Method adapted from Michael Kelder's vert2con() MATLAB function
    http://www.mathworks.com/matlabcentral/fileexchange/7895-vert2con-vertices-to-constraints
    '''

    hull = scipy.spatial.ConvexHull(Vs)
    K = hull.simplices
    c = sp.mean(Vs[hull.vertices, :], 0)  # c is a (1xd) vector

    # perform affine transformation (subtract c from every row in Vs)
    V = Vs - c

    A = sp.NaN * sp.empty((K.shape[0], Vs.shape[1]))

    rc = 0
    for i in range(K.shape[0]):
        ks = K[i, :]
        F = V[ks, :]

        if numpy.linalg.matrix_rank(F) == F.shape[0]:
            f = sp.ones(F.shape[0])
            A[rc, :] = scipy.linalg.solve(F, f)
            rc += 1

    A = A[0:rc, :]
    # b = ones(size(A)[1], 1)
    b = sp.dot(A, c) + 1.0

    # remove duplicate entries in A and b?
    # X = [A b]
    # X = unique(round(X,7),1)
    # A = X[:,1:end-1]
    # b = X[:,end]

    return (A, b)


def in_region(xi, A, b, tol=1e-12):
    '''
    Determine whether point xi lies within the region or on the region boundary
    defined by the system of inequalities A*xi <= b

    Parameters:
        A

        b

        tol     Optional. A tolerance for how close a point need to be to the
                region before it is considererd 'in' the region.
                Default value is 1e-12.

    Returns:
        bool    True/False value indicating if xi is in the region relative to
                the tolerance specified.
    '''

    # check if b is a column vector with ndim=2, or (L,) array with ndim=1 only
    if b.ndim == 2:
        b = b.flatten()

    if sp.all(sp.dot(A, xi) - b <= tol):
        return True
    else:
        return False


def out_region(xi, A, b, tol=1e-12):
    '''
    Determine whether point xi lies strictly outside of the region (NOT on the
    region boundary) defined by the system of inequalities A*xi <= b

    Parameters:
        A

        b

        tol     Optional. Float. A tolerance for how close a point need to be
                to the region before it is considererd 'in' the region.
                Default value is based on what is specified in inregion().

    Returns:
        bool    True/False value indicating if xi is in the region relative to
                the tolerance specified.
    '''

    # check if b is a column vector with ndim=2, or (L,) array with ndim=1 only
    if b.ndim == 2:
        b = b.flatten()

    if in_region(xi, A, b, tol=tol):
        return False
    else:
        return True


def pts_in_region(Xs, A, b, tol=1e-12):
    '''
    Similar to inregion(), but works on an array of points and returns the
    points and indices.

    Parameters:

    Returns:

    '''

    # check if b is a column vector with ndim=2, or (L,) array with ndim=1 only
    if b.ndim == 2:
        b = b.flatten()

    ks = []
    for idx, xi in enumerate(Xs):
        if in_region(xi, A, b, tol=tol):
            ks.append(idx)

    Cs = Xs[ks, :]

    return Cs, ks


def pts_out_region(Xs, A, b, tol=1e-12):
    '''
    Similar to outregion(), but works on an array of points and returns the
    points and indices.

    Parameters:
        Xs

        A

        b

        tol     Optional. Float. Tolerance for checking if a point is contained
                in a region.
                Default value is 1e-12.

    Returns:
        Cs

        ks
    '''

    # check if b is a column vector with ndim=2, or (L,) array with ndim=1 only
    if b.ndim == 2:
        b = b.flatten()

    ks = []
    for idx, xi in enumerate(Xs):
        if out_region(xi, A, b, tol=tol):
            ks.append(idx)

    Cs = Xs[ks, :]

    return Cs, ks


def allcomb(*X):
    '''
    Cartesian product of a list of vectors.

    Parameters:
        *X      A variable argument list of vectors

    Returns:
        Xs      A numpy array containing the combinations of the Cartesian
                product.
    '''

    combs = itertools.product(*X)
    Xs = sp.array(list(combs))
    return Xs


def rand_pts(Npts, axis_lims):
    '''
    Generate a list of random points within a user-specified range.

    Parameters:
        Npts        Number of points to generate.

        axis_lims   An array of axis min-max pairs.
                    e.g. [xmin, xmax, ymin, ymax, zmin, zmax, etc.] where
                    d = len(axis_lims)/2

    Returns:
        Ys          (Npts x d) numpy array of random points.
    '''

    dim = len(axis_lims) / 2

    Xs = sp.rand(int(Npts), dim)
    # axis_lims = sp.array([-0.5, 1.25, 0, 1.5])

    # convert axis lims list into a Lx2 array that can be used with matrix
    # multiplication to scale the random points
    AX = sp.reshape(axis_lims, (-1, 2))
    D = sp.diag(AX[:, 1] - AX[:, 0])

    Ys = sp.dot(Xs, D) + AX[:, 0]

    return Ys


def calc_pfr_trajectory(Cf, rate_fn, t_end, NUM_PTS=250, linspace_ts=False):
    '''
    Convenience function that integrate the PFR trajecotry from the feed point
    specified Cf, using scipy.integrate.odeint().
    Time is based on a logscaling

    Parameters:
        Cf          (d x 1) numpy array. Feed concentration to the PFR.

        rate_fn     Python function. Rate function in (C,t) format that returns
                    an array equal to the length of Cf.

        t_end       Float indicating the residence time of the PFR.

        NUM_PTS     Optional. Number of PFR points.
                    Default value is 250 points.

    Returns:
        pfr_cs      (NUM_PTS x d) numpy array representing the PFR trajectory
                    points.

        pfr_ts      (NUM_PTS x 1) numpy array of PFR residence times
                    corresponding to pfr_cs.
    '''

    # TODO: optional accuracy for integration

    # since logspace can't give log10(0), append 0.0 to the beginning of pfr_ts
    # and decrese NUM_PTS by 1
    if linspace_ts:
        pfr_ts = sp.linspace(0, t_end, NUM_PTS)
    else:
        pfr_ts = sp.append(0.0, sp.logspace(-3, sp.log10(t_end), NUM_PTS - 1))

    pfr_cs = scipy.integrate.odeint(rate_fn, Cf, pfr_ts)

    return pfr_cs, pfr_ts


def calc_cstr_locus(Cf, rate_fn, NUM_PTS, axis_lims, tol=1e-6, N=2e4):
    '''
    Brute-force CSTR locus solver using geometric CSTR colinearity condition
    between r(C) and (C - Cf).

    Parameters:
        Cf          []

        rate_fn     []

        NUM_PTS     []

        axis_lims   []

        tol         Optional.
                    Default value is 1e-6.

        N           Optional.
                    Default value is 2e4.

    Returns:
        cstr_cs     A list of cstr effluent concentrations.

        cstr_ts     CSTR residence times corresponding to cstr_cs.
    '''

    Cs = Cf
    ts = [0.0]

    N = int(N)  # block length

    while Cs.shape[0] < NUM_PTS:

        # update display
        print "%.2f%% complete..." % (float(Cs.shape[0]) / float(NUM_PTS) *
                                      100.0)

        # generate random points within the axis limits in blocks of N points
        Xs = rand_pts(N, axis_lims)

        # loop through each point and determine if it is a CSTR point
        ks = []
        for i, ci in enumerate(Xs):
            # calculate rate vector ri and mixing vector vi
            ri = rate_fn(ci, 1)
            vi = ci - Cf

            # normalise ri and vi
            vn = vi / scipy.linalg.norm(vi)
            rn = ri / scipy.linalg.norm(ri)

            # calculate colinearity between rn and vn
            if sp.fabs(sp.fabs(sp.dot(vn, rn) - 1.0)) <= tol:
                ks.append(i)

                # calc corresponding cstr residence time (based on 1st element)
                tau = vi[0] / ri[0]
                ts.append(tau)

        # append colinear points to current list of CSTR points
        Cs = sp.vstack([Cs, Xs[ks, :]])

    # chop to desired number of points
    Cs = Cs[0:NUM_PTS, :]
    ts = sp.array(ts[0:NUM_PTS])

    return Cs, ts


def calc_cstr_locus_fast(Cf, rate_fn, t_end, num_pts):
    '''
    Quick (potentially inexact) CSTR solver using a standard non-linear solver
    (Newton). The initial guess is based on the previous solution.
    Note: this method will not find multiple solutions and may behave poorly
    with systems with multiple solutions. Use only if you know that the system
    is 'simple' (no multiple solutions) and you need a quick answer

    Parameters:
        Cf

        rate_fn

        t_end

        num_pts

    Returns:
        cstr_cs

        cstr_ts
    '''

    cstr_ts = sp.hstack([0., sp.logspace(-3, sp.log10(t_end), num_pts - 1)])
    cstr_cs = []

    # loop through each cstr residence time and solve for the corresponding
    # cstr effluent concentration
    C_guess = Cf
    for ti in cstr_ts:

        # define CSTR function
        def cstr_fn(C):
            return Cf + ti * rate_fn(C, 1) - C

        # solve
        ci = scipy.optimize.newton_krylov(cstr_fn, C_guess)

        cstr_cs.append(ci)

        # update guess
        C_guess = ci

    # convert to numpy array
    cstr_cs = sp.array(cstr_cs)

    return cstr_cs, cstr_ts


def convhull_pts(Xs):
    '''
    A wrapper for SciPy's ConvexHull() function that returns the convex hull
    points directly and neatens up the syntax slightly. Use when you just need
    the convex hull points and not the indices to the vertices or facets.

    Parameters:
        Xs  (L x d) array where L is the number of point and d is the number of
            components (the dimension of the points). We compute conv(Xs).

    Returns:
        Vs  (k x d) array where k is the number of points belonging to the
            convex hull of Xs, conv(Xs), and d is the number of components (the
            dimension of the points).
    '''

    K = scipy.spatial.ConvexHull(Xs).vertices
    Vs = Xs[K, :]

    return Vs


def isColVector(A):
    """
    Checks if input A is a 2-D numpy array, orientated as a column vector
    """

    if isinstance(A, sp.ndarray) and A.ndim == 2:
        row_num, col_num = A.shape
        if col_num == 1 and row_num > 1:
            return True

    return False


def isRowVector(A):
    """
    Checks if input A is a 2-D numpy array, orientated as a row vector
    """

    if isinstance(A, sp.ndarray) and A.ndim == 2:
        row_num, col_num = A.shape
        if col_num > 1 and row_num == 1:
            return True

    return False


def stoich_S_1D(Cf0, stoich_mat):
    """
    A helper function for stoichSubspace().
    Single feed, single reaction version.
    """

    # check for positive concentrations
    if sp.any(Cf0 < 0):
        raise Exception("Feed concentrations must be positive")

    # flatten Cf0 and stoich_mat to 1-D arrays for consistency
    if Cf0.ndim == 2:
        Cf0 = Cf0.flatten()
    if stoich_mat.ndim == 2:
        stoich_mat = stoich_mat.flatten()

    # calculate the limiting requirements
    limiting = Cf0/stoich_mat

    # only choose negative coefficients as these indicate reactants
    k = limiting < 0.0

    # calc maximum extent based on limiting reactant and calc C
    # we take max() because of the negative convention of the limiting
    # requirements
    e_max = sp.fabs(max(limiting[k]))

    # calculate the corresponding point in concentration space
    C = Cf0 + stoich_mat*e_max

    # form Cs and Es and return
    Cs = sp.vstack([Cf0, C])
    Es = sp.array([[0.0, e_max]]).T

    return (Cs, Es)


def stoich_S_nD(Cf0, stoich_mat):
    """
    A helper function for stoichSubspace().
    Single feed, multiple reactions version.
    """

    # check for positive concentrations
    if sp.any(Cf0 < 0):
        raise Exception("Feed concentrations must be positive")

    # flatten Cf0 to 1-D array for consistency
    if Cf0.ndim == 2:
        Cf0 = Cf0.flatten()

    # extent associated with each feed vector
    Es = con2vert(-stoich_mat, Cf0)

    # calculate the corresponding points in concentration space
    Cs = (Cf0[:, None] + sp.dot(stoich_mat, Es.T)).T

    return (Cs, Es)


def getExtrema(Xs, axis=0):
    """
    Collect the max and min values according to a user-specified axis direction
    of Xs.
    """

    Xs = sp.vstack(Xs)
    Xs_mins = sp.amin(Xs, axis)
    Xs_maxs = sp.amax(Xs, axis)
    Xs_bounds = sp.vstack([Xs_mins, Xs_maxs])

    return Xs_bounds


def stoichSubspace(Cf0s, stoich_mat):
    """
    Compute the extreme points of the stoichiometric subspace, S, from multiple
    feed points and a stoichoimetric coefficient matrix.

    Parameters:
        stoich_mat      (n x d) array. Each row in stoich_mat corresponds to a
                        component and each column corresponds to a reaction.

        Cf0s            (M x n) matrix. Each row in Cf0s corresponds to an
                        individual feed and each column corresponds to a
                        component.

    Returns:
        S_attributes    dictionary containing the vertices of the
                        stoichiometric subspace in extent and concentration
                        space for individual feeds.

        keys:
            all_Es      vertices of the individual stoichiometric subspaces in
                        extent space.

            all_Cs      vertices of the individual stoichiometric subspaces in
                        concentration space.

            bounds_Cs   bounds of the stoichiometric subspace in concentration
                        space.

            bounds_Es   bounds of the stoichiometric subspace in extent space.
    """

    # if user Cf0s is not in a list, then check to see if it is a matrix of
    # feeds (with multiple rows), otherwise, put it in a list
    if not isinstance(Cf0s, list):
        # is Cf0s a matrix of feed(s), or just a single row/column vector?
        if Cf0s.ndim == 1 or (isColVector(Cf0s) or isRowVector(Cf0s)):
            Cf0s = [Cf0s]

    # always treat stoich_mat as a matrix for consistency. Convert 'single rxn'
    # row into a column vector
    if stoich_mat.ndim == 1:
        stoich_mat = stoich_mat.reshape((len(stoich_mat), 1))

    # loop through each feed and calculate stoich subspace
    all_Es = []
    all_Cs = []
    for Cf0 in Cf0s:
        # convert Cf0 to (L,) for consistency
        if Cf0.ndim == 2:
            Cf0 = Cf0.flatten()

        # check num components is consistent between Cf0 and stoich_mat
        if len(Cf0) != stoich_mat.shape[0]:
            raise Exception("The number of components in the feed does not \
                             match the number of rows in the stoichiometric \
                             matrix.")

        # compute S based on a single or multiple reactions
        if isColVector(stoich_mat):
            Cs, Es = stoich_S_1D(Cf0, stoich_mat)
        else:
            Cs, Es = stoich_S_nD(Cf0, stoich_mat)

        # append vertices for S in extent and concentration space
        all_Es.append(Es)
        all_Cs.append(Cs)

    # get max and min bounds for Cs and Es
    Cs_bounds = getExtrema(all_Cs)
    Es_bounds = getExtrema(all_Es)

    # if there was only one feed, return the data unpacked (so that it's not in
    # a one-element) list
    if len(all_Cs) == 1:
        all_Cs = all_Cs[0]
    if len(all_Es) == 1:
        all_Es = all_Es[0]

    # create a dictionary containing all the attributes of the stoich subspace
    S = {
        'all_Es': all_Es,
        'all_Cs': all_Cs,
        'bounds_Es': Es_bounds,
        'bounds_Cs': Cs_bounds
    }

    return S


def nullspace(A, tol=1e-15):
    '''
    Compute the nullspace of A using singular value decomposition (SVD). Factor
    A into three matrices U,S,V such that A = U*S*(V.T), where V.T is the
    transpose of V. If A has size (m x n), then U is (m x m), S is (m x n) and
    V.T is (n x n).

    If A is (m x n) and has rank = r, then the dimension of the nullspace
    matrix is (n x (n-r))

    Note:
        Unlike MATLAB's svd() function, Scipy returns V.T automatically and not
        V. Also, the S variable returned by scipy.linalg.svd() is an array and
        not a (m x n) matrix as in MATLAB.

    Parameters:
        A       (m x n) matrix. A MUST have ndim==2 since a 1d numpy array is
                ambiguous -- is it a mx1 column vector or a 1xm row vector?

        tol     Optional. Tolerance to determine singular values.
                Default value is 1e-15.

    Returns:
        N   (n x n-r) matrix. Columns in N correspond to a basis of the
            nullspace of A, null(A).
    '''

    U, s, V = scipy.linalg.svd(A)

    # scipy's svd() function works different to MATLAB's. The s returned is an
    # array and not a matrix.
    # convert s to an array that has the same number of columns as V (if A is
    # mxn, then V is nxn and len(S) = n)
    S = sp.zeros(V.shape[1])

    # fill S with values in s (the singular values that are meant to be on the
    # diagoanl of the S matrix like in MATLAB)
    for i, si in enumerate(s):
        S[i] = si

    # find smallest singualr values
    ks = sp.nonzero(S <= tol)[0]

    # extract columns in V. Note that V here is V.T by MATLAB's standards.
    N = V[:, ks]

    return N


def rank(A):
    '''
    Wrapper to numpy.linalg.matrix_rank(). Calculates the rank of matrix A.
    Useful for critical CSTR and DSR calculations.

    Parameters:
        A   (m x n) numpy array.

    Returns:
        r   The rank of matrix A.
    '''

    return numpy.linalg.matrix_rank(A)


def cullPts(Xs, min_dist, axis_lims=None):
    '''
    Thin out a set of points Xs by removing all neighboring points in Xs that
    lie within an open radius of an elipse, given by the elipse equation:
        r^2 = (x1/s1)^2 + (x2/s2)^2 + ... + (xn/sn)^2
    This function is useful for when we wish to spread out a set of points in
    space where all points are at least min_dist apart. For example, plotting
    a locus of CSTR points generated by a Monte Carlo method where the original
    points are not evenly spaced, but the markers on a plot need to be evenly
    spaced for display purposes.

    Parameters:
        Xs          A (N x d) numpy array of points that we wish to space out.

        min_dist    Positive float. Minimum distance. If points are less than
                    min_dist apart, remove from list.

        axis_lims   Optional. S is an array of floats used to adjust the shape
                    of the elipse, which is based on the axis limits. By
                    example, if xlim = [0, 1], ylim = [0, 0.1] and
                    zlim = [0.1, 0.45], then
                    S[0] = 1-0 = 1;
                    S[1] = 0.1-0 = 0.1 and
                    S[2] = 0.45-0.1 = 0.35
                    Default value is None, in which case all S[i]'s are set to
                    one.

    Returns:
        Vs          Numpy array where are points are spaced at least min-dist
                    apart.
    '''

    # generate S array that holds the scaling values that distorts the shape
    # of the elipse
    if axis_lims is None:
        S = sp.ones((Xs.shape[0], 1))
    else:
        S = []

        for i in range(0, len(axis_lims), 2):
            S.append(axis_lims[i + 1] - axis_lims[i])

        S = sp.array(S)

    # now remove points. Loop through each point and check distance to all
    # other points.

    # TODO: ensure that convex hull points are not removed.

    i = 0
    while i < Xs.shape[0]:
        xi = Xs[i, :]

        # check distance of all other points from xi and remove points that
        # are closer than tol.
        ks = []
        for j, xj in enumerate(Xs):

            if i != j:
                # calc distance from xi to xj
                dx = xi - xj
                r = sp.sqrt(sp.sum((dx / S)**2))

                if r <= min_dist:
                    ks.append(j)

        if len(ks) > 0:
            # remove points and reset counter so that we don't miss any
            # previous points
            Xs = sp.delete(Xs, ks, 0)
            i = 0
        else:
            i += 1

    Vs = Xs
    return Vs


def calcDim(Xs):
    """
    Compute the dimension of a set of point Xs
    """

    # check for a single row or column vector
    if isRowVector(Xs) or isColVector(Xs) or Xs.ndim==1:
        return 0

    # convert N points to N-1 vectors
    Vs = Xs - Xs[0, :]

    return rank(Vs)
