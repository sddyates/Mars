
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def minmod(np.ndarray[DTYPE_t, ndim=2] V):
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

    cdef int i, var
    cdef int nvar = V.shape[0]

    cdef double gradient
    cdef double a, b

    cdef np.ndarray[DTYPE_t, ndim=2] m = np.empty((nvar, V.shape[1] - 2), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] L = np.empty((nvar, m.shape[1] - 1), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.empty((nvar, m.shape[1] - 1), dtype=DTYPE)

    for var in range(nvar):
        for i in range(m.shape[1]):

            a = (V[var, i+1] - V[var, i])
            b = (V[var, i+2] - V[var, i+1])

            # print(var, i, a, b)

            if np.absolute(a) < np.absolute(b):
                gradient = a
            else:
                gradient = b

            if a*b > 0.0:
                m[var, i] = gradient
            else:
                m[var, i] = 0.0

        for i in range(L.shape[1]):
            L[var, i] = V[var, i+1] + m[var, i]*0.5
            R[var, i] = V[var, i+2] - m[var, i+1]*0.5

    return L, R


def flat(np.ndarray[DTYPE_t, ndim=2] V):
    """
    Synopsis
    --------
    Obtain the left and right hand states of the cell faces.
    This method assumes the states are equal to the volume
    average at the center of the cell.

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
    None
    """

    cdef int i, var
    cdef np.ndarray[DTYPE_t, ndim=2] L = np.empty((V.shape[0], V.shape[1] - 1), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.empty((V.shape[0], V.shape[1] - 1), dtype=DTYPE)

    for var in range(V.shape[0]):
        for i in range(V.shape[1]):
            L = V[var, i]
            R = V[var, i+1]

    return L,  R
