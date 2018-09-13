
from numba import jit
import numpy as np

@jit
def minmod(V, gz, dxi):

    nvar = V.shape[0]
    imax = V.shape[1]-1

    a = (V[:, 1:imax-1] - V[:, :imax-2])/dxi
    b = (V[:, 2:imax] - V[:, 1:imax-1])/dxi

    m = np.zeros([nvar, imax-1], dtype=np.float64)
    for i in range(1, imax-1):
        for var in range(nvar):
            gradient = a[var, i-1] if abs(a[var, i-1]) < abs(b[var, i-1]) else b[var, i-1]
            m[var, i-1] = gradient if a[var, i-1]*b[var, i-1] > 0.0 else 0.0

    L = V[:, 1:imax-1] + m[:, :imax-2]*0.5*dxi
    R = V[:, 2:imax] - m[:, 1:imax-1]*0.5*dxi

    return L, R

@jit
def flat(y, g, dxi):
    return y[:, :-g.gz], y[:, g.gz:]