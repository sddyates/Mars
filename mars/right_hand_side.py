
import sys
import numpy as np
import numba as nb
from numba import prange

from settings import *
from tools import flux_tensor, cons_to_prims, prims_to_cons


def flux_difference(U, g, a, t):
    """
    Synopsis
    --------
    Construct the fluxes through the cell
    faces normal to the direction of "axis".

    Args
    ----
    U: numpy array-like
    state vector containing all
    conservative variables.

    g: object-like
    object containing all variables related to
    the grid, e.g. cell width.

    a: object-like
    object containing specified algorithms for use
    in the seprate stages of a time step.

    vx(n,t,b): int-like
    indexes for for the normal, tangential and
    bi-tangential velocity components.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    V = np.empty(shape=U.shape, dtype=np.float64)
    cons_to_prims(U, V, a.gamma_1)

    if g.rank == 0:
        t.start_reconstruction()
    VL, VR = a.reconstruction(V)
    if g.rank == 0:
        t.stop_reconstruction()

    UL = np.empty(shape=VL.shape, dtype=np.float64)
    UR = np.empty(shape=VR.shape, dtype=np.float64)
    prims_to_cons(VL, UL, a.igamma_1)
    prims_to_cons(VR, UR, a.igamma_1)

    FL = np.empty(shape=VL.shape, dtype=np.float64)
    FR = np.empty(shape=VR.shape, dtype=np.float64)
    flux_tensor(UL, VL, FL, g.vxntb[0], g.vxntb[1], g.vxntb[2])
    flux_tensor(UR, VR, FR, g.vxntb[0], g.vxntb[1], g.vxntb[2])

    if g.rank == 0:
        t.start_riemann()
    dflux, g.speed_max = a.riemann_solver(
        FL, FR, UL, UR, VL, VR,
        g.speed_max, a.gamma, g.dt/g.dx[g.vxntb[0]-2],
        g.vxntb[0], g.vxntb[1], g.vxntb[2]
    )
    if g.rank == 0:
        t.stop_riemann()

    return dflux


def RHSOperator(U, g, a, t):
    """
    Synopsis
    --------
    Determine the right hand side operator.

    Args
    ----
    U: numpy array-like
    state vector containing all
    conservative variables.

    g: object-like
    object containing all variables related to
    the grid, e.g. cell width.

    a: object-like
    object containing specified algorithms for use
    in the seprate stages of a time step.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    if g.rank == 0:
        t.start_space_loop()

    rhs = np.zeros(shape=U.shape, dtype=np.float64)

    #  i: g.beg[2]:g.end[2]
    #  j: g.beg[1]:g.end[1]
    #  k: g.beg[0]:g.end[0]

    if g.ndims == 2:

        g.vxntb = [2, 3, 4]
        rhs[:, g.beg[0]:g.end[0]] = flux_difference(U, g, a, t)

    if g.ndims == 3:

        for j in range(g.beg[0], g.end[0]):
            g.vxntb = [2, 3, 4]
            rhs[:, j, g.beg[1]:g.end[1]] = flux_difference(U[:, j, :], g, a, t)

        for i in range(g.beg[1], g.end[1]):
            g.vxntb = [3, 2, 4]
            rhs[:, g.beg[0]:g.end[0], i] += flux_difference(U[:, :, i], g, a, t)

    if g.ndims == 4:

        for k in range(g.beg[0], g.end[0]):
            for j in range(g.beg[1], g.end[1]):
                g.vxntb = [2, 3, 4]
                rhs[:, k, j, g.beg[2]:g.end[2]] = flux_difference(U[:, k, j, :], g, a, t)

        for k in range(g.beg[0], g.end[0]):
            for i in range(g.beg[2], g.end[2]):
                g.vxntb = [3, 2, 4]
                rhs[:, k, g.beg[1]:g.end[1], i] += flux_difference(U[:, k, :, i], g, a, t)

        for j in range(g.beg[1], g.end[1]):
            for i in range(g.beg[2], g.end[2]):
                g.vxntb = [4, 2, 3]
                rhs[:, g.beg[0]:g.end[0], j, i] += flux_difference(U[:, :, j, i], g, a, t)

    if g.rank == 0:
        t.stop_space_loop()

    return rhs
