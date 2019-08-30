
import numpy as np
import sys
from settings import *


def flux_tensor(U, a, vxn, vxt, vxb):
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

    F = np.zeros(shape=U.shape)

    V = cons_to_prims(U, a)

    if a.is_1D:
        F[rho] = U[rho]*V[vxn]
        F[vxn] = U[rho]*V[vxn]*V[vxn]
        F[eng] = V[vxn]*(U[eng] + V[prs])

    elif a.is_2D:
        F[rho] = U[rho]*V[vxn]
        F[vxn] = U[rho]*V[vxn]*V[vxn]
        F[vxt] = U[rho]*V[vxn]*V[vxt]
        F[eng] = V[vxn]*(U[eng] + V[prs])

    elif a.is_3D:
        F[rho] = U[rho]*V[vxn]
        F[vxn] = U[rho]*V[vxn]*V[vxn]
        F[vxt] = U[rho]*V[vxn]*V[vxt]
        F[vxb] = U[rho]*V[vxn]*V[vxb]
        F[eng] = V[vxn]*(U[eng] + V[prs])

    return F


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


def cons_to_prims(U, a):

    V = np.zeros(shape=U.shape)

    m2 = np.sum(U[mvx1:]**2, axis=0)
    kinE = 0.5*m2/U[rho]

    U[eng, U[eng, :] < 0.0] = small_pressure/a.gamma_1 \
        + kinE[U[eng, :] < 0.0]

    V[rho] = U[rho]
    V[vx1:] = U[mvx1:]/U[rho]
    V[prs] = a.gamma_1*(U[eng] - kinE)

    V[prs, V[prs, :] < 0.0] = small_pressure

    if np.isnan(V).any():
        print("Error, nan in cons_to_prims")
        sys.exit()

    return V


def prims_to_cons(V, a):

    U = np.zeros(shape=V.shape)

    U[rho] = V[rho]
    U[mvx1:] = V[rho]*V[vx1:]
    v2 = np.sum(V[vx1:]**2, axis=0)
    U[eng] = 0.5*V[rho]*v2 + V[prs]/a.gamma_1

    if np.isnan(U).any():
        print("Error, nan in prims_to_cons")
        sys.exit()

    return U


def time_step(V, g, a):

    cs = np.sqrt(a.gamma*V[prs]/V[rho])

    max_velocity = np.amax(abs(V[vx1:]))
    max_speed = np.amax(abs(V[vx1:]) + cs)
    dt = a.cfl*g.min_dxi/max_speed
    mach_number = np.amax(abs(V[vx1:])/cs)

    if np.isnan(dt):
        print("Error, nan in time_step, cs =", cs)
        sys.exit()

    if dt < small_dt:
        print("dt to small, exiting.")
        sys.exit()

    return dt, max_velocity, mach_number
