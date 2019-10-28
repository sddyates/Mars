
from right_hand_side import RHSOperator
from tools import prims_to_cons, cons_to_prims


def Euler(U, dt, g, a, t, p):
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

    U_new = U + dt*RHSOperator(U, g, a)
    g.boundary(U_new, p)
    del U

    return U_new


def RungaKutta2(U, dt, g, a, t, p):
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

    K1 = RHSOperator(U, g, a, t, dt)
    g.boundary(K1, p)

    # My need to recalculate the time step here.

    U_new = U + 0.5*(K1 + RHSOperator(U+K1, g, a, t, dt))
    g.boundary(U_new, p)
    del K1, U

    return U_new
