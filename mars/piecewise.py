
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def minmod(np.ndarray[DTYPE_t, ndim=2] V, int gz, double dxi):

    cdef int i, var
    cdef int nvar = V.shape[0]
    cdef int imax = V.shape[1]-1

    cdef double gradient
    cdef double a, b
    cdef double dxi2 = 0.5*dxi
    cdef double dxi1 = 1.0/dxi

    cdef np.ndarray[DTYPE_t, ndim=2] m = np.zeros([nvar, imax-1], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] L = np.zeros([nvar, imax-2], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.zeros([nvar, imax-2], dtype=DTYPE)

    for var in range(nvar):
        for i in range(1, imax-1):

            a = (V[var, i] - V[var, i-1])*dxi1
            b = (V[var, i+1] - V[var, i])*dxi1

            gradient = a if abs(a) < abs(b) else b
            m[var, i-1] = gradient if a*b > 0.0 else 0.0

    for i in range(1, imax-1):
        for var in range(nvar):
            L[var, i-1] = V[var, i] + m[var, i-1]*dxi2
            R[var, i-1] = V[var, i+1] - m[var, i]*dxi2

    return L, R


def flat(y, g):
    return y[:, :-g.gz], y[:, g.gz:]


