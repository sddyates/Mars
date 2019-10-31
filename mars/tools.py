
import numpy as np
import numba as nb
import sys

from settings import *

@nb.jit(cache=True)
def flux_tensor(U, V, F, vxn, vxt, vxb):
    """
    Synopsis
    --------
    construct the flux tensor from the
    conservative and primative vaiables.

    Args
    ----
    U: numpy array-like
    State vector containing all
    conservative variables.

    a: object-like
    object containing specified algorithms for use
    in the seprate stages of a time step.

    vx(n,t,b): int-like
    Indexes for for the normal, tangential and
    bi-tangential velocity components.

    Returns
    -------
    F: numpy array-like
    Array of numerical fluxes.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    F[rho] = U[vxn]
    F[eng] = V[vxn]*(U[eng] + V[prs])
    F[vxn] = U[vxn]*V[vxn]
    if U.shape[0] > 3:
        F[vxt] = U[vxn]*V[vxt]
    if U.shape[0] > 4:
        F[vxb] = U[vxn]*V[vxb]

    return


@nb.jit(cache=True)
def mhd_flux_tensor(U, V, F, vxn, vxt, vxb):
    """
    Synopsis
    --------
    construct the MHD flux tensor from the
    conservative and primative vaiables.

    Args
    ----
    U: numpy array-like
    State vector containing all
    conservative variables.

    a: object-like
    object containing specified algorithms for use
    in the seprate stages of a time step.

    vx(n,t,b): int-like
    Indexes for for the normal, tangential and
    bi-tangential velocity components.

    Returns
    -------
    F: numpy array-like
    Array of numerical fluxes.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    F[rho] = U[vxn]
    F[vxn] = U[vxn]*V[vxn] - V[bxn]*V[bx1]
    F[vxt] = U[vxn]*V[vxt] - V[bxn]*V[bx2]
    F[vxb] = U[vxn]*V[vxb] - V[bxn]*V[bx3]

    F[bxn] = 0.0
    F[bxt] = V[bx2]*V[vxn] - V[bx1]*V[vxb]
    F[bxb] = V[bx3]*V[vxn] - V[bx1]*V[vxb]

    B2 = V[bx1]*V[bx1] + V[bx2]*V[bx2] + V[bx3]*V[bx3]
    vb = V[bx1]*V[vx1] + V[bx2]*V[vx2] + V[bx3]*V[vx3]

    pT = v[prs] + 0.5*B2
    F[eng] = V[vxn]*(U[eng] + pT) - V[bxn]*vb

    return


@nb.jit(cache=True)
def cons_to_prims(U, V, gamma_1):

    V[rho] = U[rho]

    irho = 1.0/V[rho]

    m2 = U[mvx1]**2
    V[vx1] = U[mvx1]*irho
    if U.shape[0] > 3:
        m2 += U[mvx2]**2
        V[vx2] = U[mvx2]*irho
    if U.shape[0] > 4:
        m2 += U[mvx3]**2
        V[vx3] = U[mvx3]*irho

    kinE = 0.5*m2*irho
    V[prs] = gamma_1*(U[eng] - kinE)

    return


@nb.jit(cache=True)
def prims_to_cons(V, U, igamma_1):

    U[rho] = V[rho]

    v2 = V[vx1]**2
    U[mvx1] = V[vx1]*V[rho]
    if U.shape[0] > 3:
        v2 += V[vx2]**2
        U[mvx2] = V[vx2]*V[rho]
    if U.shape[0] > 4:
        v2 += V[vx3]**2
        U[mvx3] = V[vx3]*V[rho]

    U[eng] = 0.5*V[rho]*v2 + V[prs]*igamma_1

    return


def time_step(t, g, a, p):

    dt_new = a.cfl*g.min_dxi/g.speed_max

    dt = min(dt_new, p['max dt increase']*dt_new)

    if (t + dt) > p['max time']:
        dt = p['max time'] - t

    if dt < a.small_dt:
        print("dt to small, exiting.")
        sys.exit()

    return dt
