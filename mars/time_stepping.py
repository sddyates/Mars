
from right_hand_side import RHSOperator
from tools import prims_to_cons, \
    cons_to_prims, JAx1, JAx2, JAx3


def Euler(V, dt, g, a, p):
    """
    Synopsis
    --------
    Evolve the simulation domain though dt
    using the Euler method.

    Args
    ----
    V: numpy array-like
    State vector containing the hole solution
    and all variables

    dt: double-like
    Time step, in simulation units.

    g: object-like
    Object containing all variables related to
    the grid, e.g. cell width.

    a: object-like
    object containing specified algorithms for use
    in the seprate stages of a time step.

    p: dic-like
    Dictionary of user defined ps, e.g.
    maximum simulation time.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """
    U = prims_to_cons(V, a)
    U_new = U + dt*RHSOperator(U, g, a)
    g.boundary(U_new, p)
    V = cons_to_prims(U_new, a)

    return V


def RungaKutta2(V, dt, g, a, p):
    """
    Synopsis
    --------
    Evolve the simulation domain though time dt
    using the 2nd order RK method.

    Args
    ----
    V: numpy array-like
    State vector containing the hole solution
    and all variables

    dt: double-like
    Time step, in simulation units.

    g: object-like
    Object containing all variables related to
    the grid, e.g. cell width.

    a: object-like
    object containing specified algorithms for use
    in the seprate stages of a time step.

    p: dic-like
    Dictionary of user defined ps, e.g.
    maximum simulation time.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    U = prims_to_cons(V, a)
    K1 = dt*RHSOperator(U, g, a)
    g.boundary(K1, p)

    # My need to recalculate the time step here.

    K2 = dt*RHSOperator(U+K1, g, a)
    U_new = U + 0.5*(K1 + K2)
    g.boundary(U_new, p)
    V = cons_to_prims(U_new, a)

    return V


def muscl_hanock(V, dt, g, a, p):
    """
    Synopsis
    --------
    Evolve the simulation domain though time dt
    using the 2nd order MUSCL-hanck method.

    Args
    ----
    V: numpy array-like
    State vector containing the hole solution
    and all variables

    dt: double-like
    Time step, in simulation units.

    g: object-like
    Object containing all variables related to
    the grid, e.g. cell width.

    a: object-like
    object containing specified algorithms for use
    in the seprate stages of a time step.

    p: dic-like
    Dictionary of user defined ps, e.g.
    maximum simulation time.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    V_xp =

    V_Hatx = V_x
        - 0.5*dt/g.dx1*JAx1(V, a)*(V_xp - Vxm)
        - 0.5*dt/g.dx2*JAx2(V, a)*(V_yp - Vym)
        - 0.5*dt/g.dx3*JAx3(V, a)*(V_zp - Vzm)

    U = prims_to_cons(V, a)

    F_xp = Flux_MH(U_xp[:, i, j, k], U_xm[:, i+1, j, k])
    F_xm = Flux_MH(U_xm[:, i-1, j, k], U_xp[:, i, j, k])

    F_yp = Flux_MH(U_yp[:, i, j, k], U_ym[:, i, j+1, k])
    F_ym = Flux_MH(U_ym[:, i, j-1, k], U_yp[:, i, j, k])

    F_zp = Flux_MH(U_zp[:, i, j, k], U_zm[:, i, j, k+1])
    F_zm = Flux_MH(U_zm[:, i, j, k-1], U_zp[:, i, j, k])

    U_new = U
        - dt/g.dx*(F_xp - F_xm)
        - dt/g.dy*(F_yp - F_ym)
        - dt/g.dz*(F_zp - F_zm)

    return V
