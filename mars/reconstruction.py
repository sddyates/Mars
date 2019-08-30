
from numba import jit
import numpy as np


@jit
def minmod(V, gz, dxi):
    """
    Synopsis
    --------
    Obtain the left and right hand states from linear extrapolation
    and apply the minmod limiter.

    Args
    ----
    V: array-like
    Array of primative variables to perform the reconstruction
    on.

    Attributes
    ----------
    None

    TODO
    ----
    Fine the way to calculate the coefficients for such that
    a non-regular grid can be used.
    """

    nvar = V.shape[0]
    imax = V.shape[1] - 1

    #for i in range(1, imax-1):
    #    wp[d][i] = dx[i]/(xgc[i+1] - xgc[i])
    #    wm[d][i] = dx[i]/(xgc[i] - xgc[i-1])

    #    cp[d][i] = (xgc[i+1] - xgc[i])/(xr[i] - xgc[i])
    #    cm[d][i] = (xgc[i] - xgc[i-1])/(xgc[i] - xr[i-1])

    #    dp[d][i] = (xr[i] - xgc[i])/dx[i]
    #    dm[d][i] = (xgc[i] - xr[i-1])/dx[i]

    a = V[:, 1:imax-1] - V[:, :imax-2]
    b = V[:, 2:imax] - V[:, 1:imax-1]

    m = np.zeros([nvar, imax-1], dtype=np.float64)
    for var in range(nvar):
        for i in range(1, imax-1):
            gradient = a[var, i-1] if abs(a[var, i-1]) < abs(b[var, i-1]) \
                else b[var, i-1]
            m[var, i-1] = gradient if a[var, i-1]*b[var, i-1] > 0.0 else 0.0

    L = V[:, 1:imax-1] + m[:, :imax-2]*0.5
    R = V[:, 2:imax] - m[:, 1:imax-1]*0.5

    return L, R


@jit
def flat(y, g, dxi):
    """
    Synopsis
    --------
    Obtain the left and right hand states from and
    apply the minmod limiter.

    Args
    ----
    V: array-like
    Array of primative variables to perform the reconstruction
    on.

    Attributes
    ----------
    None

    TODO
    ----
    Fine the way to calculate the coefficients for such that
    a non-regular grid can be used.
    """
    return y[:, :-g.gz], y[:, g.gz:]
