
import sys
import numpy as np
import numba as nb
from numba import prange

from settings import *
from tools import flux_tensor, cons_to_prims, prims_to_cons


def flux_difference(U, grid, algorithm, timer):
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
    cons_to_prims(U, V, algorithm.gamma_1)

    timer.start_reconstruction()
    VL, VR = algorithm.reconstruction(V)
    timer.stop_reconstruction()

    UL = np.empty(shape=VL.shape, dtype=np.float64)
    UR = np.empty(shape=VR.shape, dtype=np.float64)
    prims_to_cons(VL, UL, algorithm.igamma_1)
    prims_to_cons(VR, UR, algorithm.igamma_1)

    FL = np.empty(shape=VL.shape, dtype=np.float64)
    FR = np.empty(shape=VR.shape, dtype=np.float64)
    flux_tensor(UL, VL, FL, grid.vxntb[0], grid.vxntb[1], grid.vxntb[2])
    flux_tensor(UR, VR, FR, grid.vxntb[0], grid.vxntb[1], grid.vxntb[2])

    timer.start_riemann()
    dflux, grid.speed_max = algorithm.riemann_solver(
        FL, FR, UL, UR, VL, VR,
        grid.speed_max, a.gamma, grid.dt/grid.dxi[grid.vxntb[0]-2],
        grid.vxntb[0], grid.vxntb[1], grid.vxntb[2]
    )
    timer.stop_riemann()

    return dflux


def RHSOperator(U, grid, algorithm, timer):
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
    the grid, e.grid. cell width.

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

    timer.start_space_loop()

    rhs = np.zeros(shape=U.shape, dtype=np.float64)

    if algorithm.is_1D:

        grid.vxntb = [2, 3, 4]
        rhs[:, grid.ibeg:grid.iend] = flux_difference(
            U, grid, algorithm, timer)

    if algorithm.is_2D:

        for j in range(grid.jbeg, grid.jend):
            grid.vxntb = [2, 3, 4]
            rhs[:, j, grid.ibeg:grid.iend] = flux_difference(
                U[:, j, :], grid, algorithm, timer)

        for i in range(grid.ibeg, grid.iend):
            grid.vxntb = [3, 2, 4]
            rhs[:, grid.jbeg:grid.jend, i] += flux_difference(
                U[:, :, i], grid, algorithm, timer)

    if algorithm.is_3D:

        for k in range(grid.jbeg, grid.jend):
            for j in range(grid.jbeg, grid.jend):
                grid.vxntb = [2, 3, 4]
                rhs[:, k, j, grid.ibeg:grid.iend] = flux_difference(
                    U[:, k, j, :], grid, algorithm, timer)

        for k in range(grid.kbeg, grid.kend):
            for i in range(grid.ibeg, grid.iend):
                grid.vxntb = [3, 2, 4]
                rhs[:, k, grid.jbeg:grid.jend, i] += flux_difference(
                    U[:, k, :, i], grid, algorithm, timer)

        for j in range(grid.jbeg, grid.jend):
            for i in range(grid.ibeg, grid.iend):
                grid.vxntb = [4, 2, 3]
                rhs[:, grid.kbeg:grid.kend, j, i] += flux_difference(
                    U[:, :, j, i], grid, algorithm, timer)

    timer.stop_space_loop()

    return rhs
