import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def minmod(np.ndarray[DTYPE_t, ndim=2] y, int gz, double dxi):

    cdef int i, var
    cdef int nvar = y.shape[0]
    cdef int ymax = y.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] m = np.zeros([nvar, ymax-2], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] L = np.zeros([nvar, ymax-3], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.zeros([nvar, ymax-3], dtype=DTYPE)

    for i in range(1, ymax-1):
        for var in range(nvar):

            a = (y[var, i] - y[var, i-1])/dxi
            b = (y[var, i+1] - y[var, i])/dxi

            gradient = a if abs(a) < abs(b) else b
            m[var, i-1] = gradient if a*b > 0.0 else 0.0

    for i in range(1, ymax-2):
        for var in range(nvar):
            L[var, i-1] = y[var, i] + m[var, i-1]/2.0*dxi
            R[var, i-1] = y[var, i+1] - m[var, i]/2.0*dxi

    return L, R


def flat(y, g):
    return y[:, :-g.gz], y[:, g.gz:]


