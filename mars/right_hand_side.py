
import sys
import numpy as np
import numba as nb
from numba import prange

from multiprocessing import Pool
from multiprocessing import Process, Value, Array

from settings import *
from tools import flux_tensor, cons_to_prims, prims_to_cons
from reconstruction import flat, minmod
from riemann_solvers import tvdlf, hll, hllc


@nb.jit(cache=True)
def flux_difference(U, vxntb, dxi, dt, gamma, gamma_1, igamma_1, speed_max):
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
    cons_to_prims(U, V, gamma_1)

    VL, VR = minmod(V)

    UL = np.empty(shape=VL.shape, dtype=np.float64)
    UR = np.empty(shape=VR.shape, dtype=np.float64)
    prims_to_cons(VL, UL, igamma_1)
    prims_to_cons(VR, UR, igamma_1)

    FL = np.empty(shape=VL.shape, dtype=np.float64)
    FR = np.empty(shape=VR.shape, dtype=np.float64)
    flux_tensor(UL, VL, FL, vxntb[0], vxntb[1], vxntb[2])
    flux_tensor(UR, VR, FR, vxntb[0], vxntb[1], vxntb[2])

    dflux, speed_max = hll(
        FL, FR, UL, UR, VL, VR,
        speed_max, gamma, dt/dxi[vxntb[0]-2],
        vxntb[0], vxntb[1], vxntb[2]
    )

    return dflux, np.float64(speed_max)


#@nb.jit(cache=True, nopython=False, parallel=True)
def RHSOperator(U, ibeg, iend, jbeg, jend, kbeg, kend, gamma, dt, dxi, speed_max):
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

    gamma_1 = gamma - 1.0
    igamma_1 = 1.0/gamma_1

    #  t.start_space_loop()

    rhs = np.zeros(shape=U.shape, dtype=np.float64)

    # if U.shape[0] == 3:
    #
    #     vxntb = np.array([2, 3, 4], dtype=np.int32)
    #     df = flux_difference(
    #         U, vxntb, dxi, dt, gamma, gamma_1, igamma_1, speed_max)
    #     rhs[:, ibeg:iend] = df[0]
    #     speed_max = df[1]

    if U.shape[0] == 4:

        vxntb = np.array([2, 3, 4], dtype=np.int32)
        for j in prange(jbeg, jend):
            df = flux_difference(
                U[:, j, :], vxntb, dxi, dt, gamma, gamma_1, igamma_1, speed_max)
            print(type(df[0]), type(df[1]), type(U[:, j, :]), type(speed_max))
            rhs[:, j, ibeg:iend] = df[0]
            speed_max = df[1]

        vxntb = np.array([3, 2, 4], dtype=np.int32)
        for i in prange(ibeg, iend):
            df = flux_difference(
                U[:, :, i], vxntb, dxi, dt, gamma, gamma_1, igamma_1, speed_max)
            rhs[:, jbeg:jend, i] += df[0]
            speed_max = df[1]

    # if U.shape[0] == 5:
    #
    #     vxntb = np.array([2, 3, 4], dtype=np.int32)
    #     for k in range(jbeg, jend):
    #         for j in range(jbeg, jend):
    #             df = flux_difference(
    #                 U[:, k, j, :], vxntb, dxi, dt, gamma, gamma_1, igamma_1, speed_max)
    #             rhs[:, k, j, ibeg:iend] = df[0]
    #             speed_max = df[1]
    #
    #     vxntb = np.array([3, 2, 4], dtype=np.int32)
    #     for k in range(kbeg, kend):
    #         for i in range(ibeg, iend):
    #             df = flux_difference(
    #                 U[:, k, :, i], vxntb, dxi, dt, gamma, gamma_1, igamma_1, speed_max)
    #             rhs[:, k, jbeg:jend, i] += df[0]
    #             speed_max = df[1]
    #
    #     vxntb = np.array([4, 2, 3], dtype=np.int32)
    #     for j in range(jbeg, jend):
    #         for i in range(ibeg, iend):
    #             df = flux_difference(U[:, :, j, i], vxntb, dxi, dt, gamma, gamma_1, igamma_1, speed_max)
    #             rhs[:, kbeg:kend, j, i] += df[0]
    #             speed_max = df[1]

    #  t.stop_space_loop()

    return rhs, speed_max
