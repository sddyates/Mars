
import numpy as np

def minmod(V, gz, dxi):

    nvar = V.shape[0]
    imax = V.shape[1]-1

    dxi2 = 0.5*dxi
    dxi1 = 1.0/dxi

    m = np.zeros([nvar, imax-1], dtype=np.float64)
    L = np.zeros([nvar, imax-2], dtype=np.float64)
    R = np.zeros([nvar, imax-2], dtype=np.float64)

    for i in range(1, imax-1):

        a = (V[:, i] - V[:, i-1])*dxi1
        b = (V[:, i+1] - V[:, i])*dxi1

        for var in range(nvar):
            gradient = a[var] if abs(a[var]) < abs(b[var]) else b[var]
            m[var, i-1] = gradient if a[var]*b[var] > 0.0 else 0.0

    for i in range(1, imax-1):

        L[:, i-1] = V[:, i] + m[:, i-1]*dxi2
        R[:, i-1] = V[:, i+1] - m[:, i]*dxi2

    return L, R


def flat(y, g, dxi):
    return y[:, :-g.gz], y[:, g.gz:]


