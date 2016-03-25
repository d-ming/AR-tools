# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

import scipy as sp
import scipy.spatial
import scipy.optimize
import scipy.linalg

import numpy.linalg

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


def plot_hplanes(A, b, ax=None):
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

    # plot based on whether ny = 0 or not
    for i, ni in enumerate(A):
        bi = b[i]
        if ni[1] != 0:
            ax.plot([0., 1.0], [y_fn(0., ni, bi), y_fn(1.0, ni, bi)], 'k-')
        else:
            ax.plot([x_fn(0.0, ni, bi), x_fn(1.0, ni, bi)], [0., 1.0], 'k-')

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

    if sp.any(sp.dot(A, c) - b > 0.0):

        def tmp_fn(xi):
            d = sp.dot(A, xi) - b
            # ensure point actually lies within region and not just on the
            # boundary
            tmp_ks = sp.nonzero(d >= -1e-10)
            # print sum(d[tmp_ks])    #sum of errors

            # return max(d)
            return sum(d[tmp_ks])

        # print "c is not in the interior, need to solve for interior point!
        # %f" % (tmp_fn(c))

        # ignore output message
        solver_result = scipy.optimize.fmin(tmp_fn, c, disp=False)
        c = solver_result

        # TODO: check if c is now an interior point...

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
    Determine whether point xi lies within the region defined by the system
    of inequalities A*xi <= b

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
    Determine whether point xi lies outside of the region defined by the system
    of inequalities A*xi <= b

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


def stoich_subspace(Cf0s, stoich_mat):
    """ 
    Compute the bounds of the stoichiometric subspace, S, from multiple feed points and a stoichoimetric coefficient matrix.

    Parameters:
    
        stoich_mat    (n x d) array. Each row in stoich_mat corresponds to a component and each column corresponds to a reaction.
        
        Cf0s          (M x n) matrix. Each row in Cf0s corresponds to an individual feed and each column corresponds to a component.


    Returns:
    
        S_attributes   dictionary that contains the vertices stoichiometric subspace in extent and concentration space for individual feeds                        as well as overall stoichiometric subspace for multiple feeds.                         
        
        keys:
        
            all_Es      vertices of the individual stoichiometric subspaces in extent space.

            all_Cs      vertices of the individual stoichiometric subspaces in concentration space.

            all_Es_mat  list of vertices of the overall stoichiometric subspace in extent space.

            all_Cs_mat  list of vertices of the overall stoichiometric subspace in concentration space.

            hull_Es     extreme vertices of the overall stoichiometric subspace in the extent space.              

            hull_Cs     extreme vertices of the overall stoichiometric subspace in concentration space.

            bounds      bounds of the stoichiometric subspace in concentration space.

    """

    # create an empty list of bounds/ axis_lims
    min_lims = []
    max_lims = []

    # to store stoichSubspace_attributes
    S_attributes = {}

    # to store vertices for each feed and stoich_mat in extent and concentration space
    all_Es = []
    all_Cs = []

    # if user input is not a list, then convert into a list
    if not isinstance(
            Cf0s, list) and not Cf0s.shape[0] > 1 and not Cf0s.shape[1] > 1:
        # put it in a list
        Cf0s = [Cf0s]

    for Cf0 in Cf0s:
        # loop through each feed point, Cf0, and check if it is a column vector 
        # with ndim=2, or a (L,) array with ndim=1 only
        if Cf0.ndim == 2:
            Cf0 = Cf0.flatten()  # converts into (L,)

            # raise an error if the no. of components is inconsistent between the feed and stoichiometric matrix
        if len(Cf0) != stoich_mat.shape[0]:
            raise Exception(
                "The number of components in the feed does not match the number of rows in the stoichiometric matrix.")

            # always treat stoich_mat as a matrix for consistency, convert if not
        if stoich_mat.ndim == 1:
            # converts a 'single rxn' row into column vector
            stoich_mat = stoich_mat.reshape((len(stoich_mat), 1))

        # check if  a single reaction or multiple reactions are occuring
        if stoich_mat.shape[1] == 1 or stoich_mat.ndim == 1:
            # if stoich_mat is (L,) array this'stoich_mat.shape[1]' raises an error 'tuple out of range'

            # converts into (L,)
            stoich_mat = stoich_mat.flatten()

            # calculate the limiting requirements
            limiting = Cf0 / stoich_mat

            # only choose negative coefficients as these indicate reactants
            k = limiting < 0.0

            # calc maximum extent based on limiting reactant and calc C
            # we take max() because of the negative convention of the limiting requirements
            e_max = sp.fabs(max(limiting[k]))

            # calc the corresponding point in concentration space
            C = Cf0 + stoich_mat * e_max

            # form Cs and Es and return
            Cs = sp.vstack([Cf0, C])
            Es = sp.array([[0., e_max]]).T

        else:
            # extent associated with each feed vector
            Es = con2vert(-stoich_mat, Cf0)

            # calc the corresponding point in concentration space
            Cs = (Cf0[:, None] + sp.dot(stoich_mat, Es.T)).T

        # vertices for each feed and stoich_mat in extent and concentration space
        all_Es.append(Es)
        all_Cs.append(Cs)

        # stack vertices in one list and find the overall stoichiometric subspace(convex hull)
        all_Es_mat = sp.vstack(all_Es)
        all_Cs_mat = sp.vstack(all_Cs)

    # compute the convexhull of the overall stoichiometric subspace 
    # if n > d + 1, then hull_Cs is returned as the full list of vertices
    if len(Cf0) > rank(stoich_mat) + 1:
        # convexHull vertices are returned as the whole stack of points
        hull_Es = all_Es_mat
        hull_Cs = all_Cs_mat
    else:
        # convexHull vertices for the overall stoichiometric subspace in extent space
        hull_all = scipy.spatial.ConvexHull(all_Es_mat)
        ks = hull_all.vertices
        hull_Es = all_Es_mat[ks, :]

        # convexHull vertices for the overall stoichiometric subspace in concentration space
        hull_all = scipy.spatial.ConvexHull(all_Cs_mat)
        ks = hull_all.vertices
        hull_Cs = all_Cs_mat[ks, :]

    # no. of components
    N = stoich_mat.shape[0]

    # create a matrix of indices
    components = sp.linspace(0, N - 1, num=N)

    for i in components:
        # loop through each component and find the (min, max) => bounds of the axis
        minMatrix = min(hull_Cs[:, i])
        maxMatrix = max(hull_Cs[:, i])

        # append limits into preallocated lists (min_lims, max_lims)
        min_lims.append(minMatrix)
        max_lims.append(maxMatrix)

        # stack them into an ndarray and flatten() into a row vector
        bounds = sp.vstack((min_lims, max_lims)).T
        bounds = bounds.flatten()  # alternating min, max values

    # create a dictionary containing all the 'attributes' of the 'stoich_subspace'
    S_attributes = {
        'all_Es': all_Es,
        'all_Cs': all_Cs,
        'all_Es_mat': all_Es_mat,
        'all_Cs_mat': all_Cs_mat,
        'hull_Es': hull_Es,
        'hull_Cs': hull_Cs,
        'bounds': bounds
    }

    return S_attributes


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
        not a (m x n) matrix like in MATLAB.

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
    Wrapper to numpy.linalg.matrik_rank(). Calculates the rank of matrix A.
    Useful for critical CSTR and DSR calculations.

    Parameters:
        A   (m x n) numpy array.

    Returns:
        r   The rank of matrix A.
    '''

    return numpy.linalg.matrix_rank(A)


def thin_out_pts(Xs, min_dist, axis_lims=None):
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
