
from right_hand_side import RHSOperator
from tools import prims_to_cons, cons_to_prims


def Euler(U, grid, algorithm, timer, parameter):
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

    U_new = U + RHSOperator(U, grid, algorithm, timer)

    # User source term call.

    grid.boundary(U_new, parameter)
    del U

    # Analysis function call.

    return U_new


def RungaKutta2(U_n, grid, algorithm, timer, parameter):
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

    L_n = RHSOperator(U_n, grid, algorithm, timer)

    U_star = U_n + L_n

    # User source term call.
    U_star += algorithm.source_terms(U_n, grid)

    grid.boundary(U_star, parameter)
    # My need to recalculate the time step here.

    L_star = RHSOperator(U_star, grid, algorithm, timer)

    U_np1 = U_n + 0.5*(L_n + L_star)

    # User source term call.
    U_np1 += algorithm.source_terms(U_n, grid)

    grid.boundary(U_np1, parameter)

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


def RungaKutta3(U_n, grid, algorithm, timer, parameter):
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

    L_n = RHSOperator(U_n, grid, algorithm, timer)

    U_star = U_n + L_n
    # User source term call.
    grid.boundary(U_star, parameter)
    # My need to recalculate the time step here.

    L_star = RHSOperator(U_star, grid, algorithm, timer)

    U_star_star = 0.25*(3.0*U_n + U_star + L_n)
    # User source term call.
    grid.boundary(U_star_star, parameter)
    # My need to recalculate the time step here.

    L_star_star = RHSOperator(U_star_star, grid, algorithm, timer)

    U_np1 = 1.0/3.0*(U_n + 2.0*U_star_star + 2.0*L_star_star)
    # User source term call.
    grid.boundary(U_np1, parameter)

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
