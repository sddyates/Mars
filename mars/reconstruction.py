import numba as nb
from numba import jit
import numpy as np


@nb.jit(cache=True)
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

    m = np.zeros((nvar, imax-1), dtype=np.float64)

    for var in range(nvar):
        for i in range(1, imax):

            a = V[var, i] - V[var, i-1]
            b = V[var, i+1] - V[var, i]

            if np.absolute(a) < np.absolute(b):
                gradient = a
            else:
                gradient = b

            if a*b > 0.0:
                m[var, i-1] = gradient
            else:
                m[var, i-1] = 0.0

    L = V[:, 1:imax-1] + m[:, :imax-2]*0.5
    R = V[:, 2:imax] - m[:, 1:imax-1]*0.5

    return L, R


@nb.jit(cache=True)
def flat(V, gz, dxi):
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
    return V[:, :-gz], V[:, gz:]
