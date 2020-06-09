
from right_hand_side import RHSOperator
from tools import prims_to_cons, cons_to_prims


def Euler(U, g, a, t, p):
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

    U_new = U + RHSOperator(U, g, a, t)

    # User source term call.

    g.boundary(U_new)
    del U

    # Analysis function call.

    return U_new


def RungaKutta2(U_n, g, a, t, p):
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

    L_n = RHSOperator(U_n, g, a, t)
    #L_n, g.speed_max = RHSOperator(U_n, g.speed_max, g.dt, g.dxi, a.gamma, g.ibeg, g.iend, 0, 0)#, g.jbeg, g.jend)

    U_star = U_n + L_n
    # User source term call.
    g.boundary(U_star)
    # My need to recalculate the time step here.

    L_star = RHSOperator(U_star, g, a, t)
    #L_star, g.speed_max = RHSOperator(U_star, g.speed_max, g.dt, g.dxi, a.gamma, g.ibeg, g.iend, 0, 0)#, g.jbeg, g.jend)

    U_np1 = U_n + 0.5*(L_n + L_star)
    # User source term call.
    g.boundary(U_np1)

    del L_n, U_star, L_star, U_n

    # Analysis function call.

    return U_np1


    # K1 = RHSOperator(U, g, a, t)
    #
    # # User source term call.
    #
    # g.boundary(K1, p)
    #
    # # My need to recalculate the time step here.
    #
    # U_new = U + 0.5*(K1 + RHSOperator(U+K1, g, a, t))
    #
    # # User source term call.
    #
    # g.boundary(U_new, p)
    # del K1, U
    #
    # # Analysis function call.
    #
    # return U_new


def RungaKutta3(U_n, g, a, t, p):
    """
    Synopsis
    --------
    Evolve the simulation domain though time dt
    using the 3nd order RK method.

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

    L_n = RHSOperator(U_n, g, a, t)

    U_star = U_n + L_n
    # User source term call.
    g.boundary(U_star)
    # My need to recalculate the time step here.

    L_star = RHSOperator(U_star, g, a, t)

    U_star_star = 0.25*(3.0*U_n + U_star + L_n)
    # User source term call.
    g.boundary(U_star_star)
    # My need to recalculate the time step here.

    L_star_star = RHSOperator(U_star_star, g, a, t)

    U_np1 = 1.0/3.0*(U_n + 2.0*U_star_star + 2.0*L_star_star)
    # User source term call.
    g.boundary(U_np1)

    del L_n, U_star, U_star_star, L_star, L_star_star, U_n

    # Analysis function call.

    return U_np1



    # K1 = RHSOperator(U, g, a, t)
    #
    # # User source term call.
    #
    # g.boundary(K1, p)
    #
    # # My need to recalculate the time step here.
    #
    # K2 = U + 0.5*(K1 + RHSOperator(U+K1, g, a, t))
    #
    # # User source term call.
    #
    # g.boundary(U_new, p)
    # del K1, U
    #
    # # Analysis function call.
    #
    # return U_new
