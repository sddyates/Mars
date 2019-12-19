
import numba as nb
from numba import jit
import numpy as np


@nb.jit(cache=True)
def minmod(V):
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

    m = np.empty((V.shape[0], V.shape[1] - 2), dtype=np.float64)
    L = np.empty((V.shape[0], m.shape[1] - 1), dtype=np.float64)
    R = np.empty((V.shape[0], m.shape[1] - 1), dtype=np.float64)

    for var in range(V.shape[0]):
        for i in range(m.shape[1]):

            a = (V[var, i+1] - V[var, i])
            b = (V[var, i+2] - V[var, i+1])

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


@nb.jit(cache=True)
def flat(V):
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
    return V[:, :-1], V[:, 1:]


# spec = [
#     ('_m', nb.float64[:]),
#     ('_L', nb.float64[:]),
#     ('_R', nb.float64[:])
# ]
# @nb.jitclass(spec)
# class Reconstruction(flat, minmod):
#     """
#     Synopsis
#     --------
#     Obtain the left and right hand states
#     from either 1st or 2nd order extrapolation
#     and apply the minmod limiter (2nd order only).
#
#     Args
#     ----
#     V: array-like
#     Array of primative variables to perform the reconstruction
#     on.
#
#     Attributes
#     ----------
#     None
#
#     TODO
#     ----
#     Fine the way to calculate the coefficients for such that
#     a non-regular grid can be used.
#     """
#
#     def __init__(self, V):
#         self._m = np.empty((V.shape[0], V.shape[1] - 2), dtype=np.float64)
#         self._L = np.empty((V.shape[0], m.shape[1] - 1), dtype=np.float64)
#         self._R = np.empty((V.shape[0], m.shape[1] - 1), dtype=np.float64)
#
#         if order_is_1st:
#             self.apply = flat(V)
#         elif order_is_2nd:
#             self.apply = minmod(V)
#
#     def apply(self, V):
#         if order_is_1st:
#             return flat(V)
#         elif order_is_2nd:
#             return minmod(V)
#
#     def _1st_order(self, V):
#         return V[:, :-1], V[:, 1:]
#
#     def _2nd_order(self, V):
#
#         for var in range(V.shape[0]):
#             for i in range(m.shape[1]):
#
#                 a = (V[var, i+1] - V[var, i])
#                 b = (V[var, i+2] - V[var, i+1])
#
#                 if np.absolute(a) < np.absolute(b):
#                     gradient = a
#                 else:
#                     gradient = b
#
#                 if a*b > 0.0:
#                     self._m[var, i] = gradient
#                 else:
#                     self._m[var, i] = 0.0
#
#             for i in range(L.shape[1]):
#                 self._L[var, i] = V[var, i+1] + self._m[var, i]*0.5
#                 self._R[var, i] = V[var, i+2] - self._m[var, i+1]*0.5
#
#         return self._L, self._R
