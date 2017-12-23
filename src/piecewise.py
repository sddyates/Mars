import numpy as np
from globe import *

def reconstruction(y, g, axis, recon_type):

    def flat(y, g):
        return y[:, :-g.gz], y[:, g.gz:]


    def linear(y, g, dxi):
        return minmod(y, g, dxi)


    def minmod(y, g, dxi):

        m = np.zeros(shape=y[:, 1:-1].shape)

        for i in range(1, y.shape[1]-1):
            for var in range(g.nvar):

                a = (y[var, i] - y[var, i-1])/dxi
                b = (y[var, i+1] - y[var, i])/dxi

                gradient = a if abs(a) < abs(b) else b
                m[var, i-1] = gradient if a*b > 0.0 else 0.0

            L = y[:, g.gz - 1:-g.gz] + m[:, :-1]/2.0*dxi
            R = y[:, g.gz:-g.gz + 1] - m[:, 1:]/2.0*dxi

        return L, R


    if axis == 'i':
        dxi = g.dx1
    if axis == 'j':
        dxi = g.dx2

    if recon_type == 'flat':
        L, R = flat(y, g)
    elif recon_type == 'linear':
        L, R = linear(y, g, dxi)  
    else:
        print('Error: Invalid reconstructor.')
        sys.exit()

    if np.isnan(np.sum(L)) or np.isnan(np.sum(R)):
        print("Error, nan in array, function: reconstruction")
        sys.exit()

    return L, R





