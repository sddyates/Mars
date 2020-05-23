
import sys
import numpy as np
import numba as nb
from numba import prange

from settings import *
from tools import flux_tensor, cons_to_prims, prims_to_cons
from stencils import OneD_first_order_stencil

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

    # dflux, g.speed_max = OneD_first_order_stencil(
    #     U, a.gamma, a.gamma_1, a.igamma_1,
    #     g.speed_max, g.dt/g.dxi[g.vxntb[0]-2],
    #     g.vxntb[0], g.vxntb[1], g.vxntb[2]
    # )

    V = np.empty(shape=U.shape, dtype=np.float64)
    cons_to_prims(U, V, a.gamma_1)

    t.start_reconstruction()
    VL, VR = a.reconstruction(V)
    t.stop_reconstruction()

    UL = np.empty(shape=VL.shape, dtype=np.float64)
    UR = np.empty(shape=VR.shape, dtype=np.float64)
    prims_to_cons(VL, UL, a.igamma_1)
    prims_to_cons(VR, UR, a.igamma_1)

    FL = np.empty(shape=VL.shape, dtype=np.float64)
    FR = np.empty(shape=VR.shape, dtype=np.float64)
    flux_tensor(UL, VL, FL, g.vxntb[0], g.vxntb[1], g.vxntb[2])
    flux_tensor(UR, VR, FR, g.vxntb[0], g.vxntb[1], g.vxntb[2])

    t.start_riemann()
    dflux, g.speed_max = a.riemann_solver(
        FL, FR, UL, UR, VL, VR,
        g.speed_max, a.gamma, g.dt/g.dxi[g.vxntb[0]-2],
        g.vxntb[0], g.vxntb[1], g.vxntb[2]
    )
    t.stop_riemann()

    return dflux


#@nb.jit(cache=True)
def RHSOperator(U, g, a, t):
#def RHSOperator(U, speed_max, dt, dxi, gamma, ibeg, iend, jbeg, jend):
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

    #t.start_space_loop()
    rhs = np.zeros(shape=U.shape, dtype=np.float64)

    # gamma_1 = gamma - 1.0
    # igamma_1 = 1.0/gamma_1

    if U.shape[0] == 3:

        g.vxntb = [2, 3, 4]
        rhs[:, g.ibeg:g.iend] = flux_difference(U, g, a, t)

    #speed_max = g.speed_max

    # if U.shape[0] == 3:
    #
    #     nvar = U.shape[0]
    #     imax = U.shape[1]
    #
    #     vxntb = [2, 3, 4]
    #     dtdx = dt/dxi[vxntb[0]-2]
    #
    #     if np.isnan(U[rho]).any():
    #         print("rho is nan before:", U[rho])
    #         sys.exit()
    #
    #     dflux = OneD_first_order_stencil(
    #         U, gamma, gamma_1, igamma_1,
    #         speed_max, dtdx,
    #         2, 3, 4
    #     )
    #
    #     #print("dflux shape:", dflux[0].shape)
    #
    #     for var in range(nvar):
    #         for i in range(ibeg+2, iend-2):
    #             #print("i=", i)
    #             rhs[var, i] = dflux[0][var, i]
    #
    #     speed_max = dflux[1]

    #sys.exit()

    # if U.shape[0] == 4:
    #
    #     for j in range(g.jbeg, g.jend):
    #         g.vxntb = [2, 3, 4]
    #         rhs[:, j, g.ibeg:g.iend] = flux_difference(U[:, j, :], g, a, t)
    #
    #     for i in range(g.ibeg, g.iend):
    #         g.vxntb = [3, 2, 4]
    #         rhs[:, g.jbeg:g.jend, i] += flux_difference(U[:, :, i], g, a, t)

    # if nvar == 4:
    #
    #     nvar = U.shape[0]
    #     jmax = U.shape[1]
    #     imax = U.shape[2]
    #
    #     vxntb = [2, 3, 4]
    #     dtdx = dt/dxi[vxntb[0]-2]
    #
    #     buffer = np.empty((nvar, imax), dtype=np.float64)
    #     for j in range(jbeg, jend):
    #
    #         for var in range(nvar):
    #             for i in range(imax):
    #                 buffer[var, i] = U[var, j, i]
    #
    #         #print("buffer=",buffer, buffer.shape, U[var, j, :].shape)
    #
    #         dflux = OneD_first_order_stencil(
    #             buffer, gamma, gamma_1, igamma_1,
    #             speed_max, dtdx,
    #             2, 3, 4
    #         )
    #
    #         for var in range(nvar):
    #             for i in range(ibeg+2, iend-2):
    #                 rhs[var, j, i] = dflux[0][var, i]
    #
    #         speed_max = dflux[1]
    #
    #     #print("rhs[rho]=", rhs[rho])
    #
    #     # j-direction
    #     vxntb = [3, 2, 4]
    #     dtdx = dt/dxi[vxntb[0]-2]
    #
    #     buffer = np.empty((nvar, jmax), dtype=np.float64)
    #     for i in range(ibeg, iend):
    #
    #         for var in range(nvar):
    #             for j in range(jmax):
    #                 buffer[var, j] = U[var, j, i]
    #
    #         dflux = OneD_first_order_stencil(
    #             buffer, gamma, gamma_1, igamma_1,
    #             speed_max, dtdx,
    #             3, 2, 4
    #         )
    #
    #         for var in range(nvar):
    #             for j in range(jbeg+2, jend-2):
    #                 rhs[var, j, i] += dflux[0][var, j]
    #
    #         speed_max = dflux[1]

    # if U.shape[0] == 5:
    #
    #     for k in range(g.kbeg, g.kend):
    #         for j in range(g.jbeg, g.jend):
    #             g.vxntb = [2, 3, 4]
    #             rhs[:, k, j, g.ibeg:g.iend] = flux_difference(U[:, k, j, :], g, a, t)
    #
    #     for k in range(g.kbeg, g.kend):
    #         for i in range(g.ibeg, g.iend):
    #             g.vxntb = [3, 2, 4]
    #             rhs[:, k, g.jbeg:g.jend, i] += flux_difference(U[:, k, :, i], g, a, t)
    #
    #     for j in range(g.jbeg, g.jend):
    #         for i in range(g.ibeg, g.iend):
    #             g.vxntb = [4, 2, 3]
    #             rhs[:, g.kbeg:g.kend, j, i] += flux_difference(U[:, :, j, i], g, a, t)
    # t.stop_space_loop()

    #print(speed_max)

    return rhs#, speed_max
