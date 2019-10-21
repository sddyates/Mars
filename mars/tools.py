
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


    F[rho] = U[rho]*V[vxn]
    F[vxn] = U[rho]*V[vxn]*V[vxn]
    F[eng] = V[vxn]*(U[eng] + V[prs])

    if U.shape[0] > 3:
        F[vxt] = U[rho]*V[vxn]*V[vxt]

    if U.shape[0] > 4:
        F[vxt] = U[rho]*V[vxn]*V[vxt]
        F[vxb] = U[rho]*V[vxn]*V[vxb]

    return


def JAx1(V, a):
    cs2 = a.gamma*V[prs]/V[rho]
    return np.array([
        [V[vx1], V[rho],     0.0,    0.0,    0.0       ],
        [0.0,    V[vx1],     0.0,    0.0,    1.0/V[rho]],
        [0.0,    0.0,        V[vx1], 0.0,    0.0       ],
        [0.0,    0.0,        0.0,    V[vx1], 0.0       ],
        [0.0,    V[rho]*cs2, 0.0,    0.0,    V[vx1]    ]
    ])


def JAx2(V, a):
    cs2 = a.gamma*V[prs]/V[rho]
    return np.array([
        [V[vx2], V[rho], 0.0,        0.0,    0.0       ],
        [0.0,    V[vx2], 0.0,        0.0,    0.0       ],
        [0.0,    0.0,    V[vx2],     0.0,    1.0/V[rho]],
        [0.0,    0.0,    0.0,        V[vx2], 0.0       ],
        [0.0,    0.0,    V[rho]*cs2, 0.0,    V[vx2]    ]
    ])


def JAx3(V, a):
    cs2 = a.gamma*V[prs]/V[rho]
    return np.array([
        [V[vx3], V[rho], 0.0,    0.0,        0.0       ],
        [0.0,    V[vx3], 0.0,    0.0,        0.0       ],
        [0.0,    0.0,    V[vx3], 0.0,        0.0       ],
        [0.0,    0.0,    0.0,    V[vx3],     1.0/V[rho]],
        [0.0,    0.0,    0.0,    V[rho]*cs2, V[vx3]    ]
    ])

@nb.jit(cache=True)
def cons_to_prims(U, V, gamma_1):

    m2 = np.sum(U[mvx1:]**2, axis=0)
    kinE = 0.5*m2/U[rho]

    V[rho] = U[rho]
    V[vx1:] = U[mvx1:]/U[rho]
    V[prs] = gamma_1*(U[eng] - kinE)

    return

@nb.jit(cache=True)
def prims_to_cons(V, U, gamma_1):

    U[rho] = V[rho]
    U[mvx1:] = V[rho]*V[vx1:]
    v2 = np.sum(V[vx1:]**2, axis=0)
    U[eng] = 0.5*V[rho]*v2 + V[prs]/gamma_1

    return


def time_step(g, a):

    #cs = np.sqrt(a.gamma*V[prs]/V[rho])

    max_velocity = g.speed_max#np.amax(abs(V[vx1:]))
    #max_speed = np.amax(abs(V[vx1:]) + cs)

    dt = a.cfl*g.min_dxi/g.speed_max
    mach_number = 0.0#np.amax(abs(V[vx1:])/cs)

    if dt < small_dt:
        print("dt to small, exiting.")
        sys.exit()

    return dt, max_velocity, mach_number
