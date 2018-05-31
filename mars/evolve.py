import sys
import numpy as np
from cython_lib.solvers import hll, hllc
from cython_lib.piecewise import flat, minmod
from tools import cons_to_prims, prims_to_cons, eigenvalues, time_step

def flux_tensor(U, p, axis):
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

    p: dic-like
    Dictionary of user defined ps, e.g. 
    maximum simulation time.        

    axis: chr-like
    Character specifying which axis the loop 
    is over.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    F = np.zeros(shape=U.shape)

    V = cons_to_prims(U, p)

    if p['Dimensions'] == '1D':
        F[rho] = V[rho]*V[vx1]
        F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
        F[eng] = V[vx1]*(U[eng] + V[prs])

    if p['Dimensions'] and axis == 'i':
        F[rho] = V[rho]*V[vx1]
        F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
        F[mvx2] = V[rho]*V[vx1]*V[vx2]
        F[eng] = V[vx1]*(U[eng] + V[prs])
    elif p['Dimensions'] == '2D' and axis == 'j':
        F[rho] = V[rho]*V[vx2]
        F[mvx1] = V[rho]*V[vx1]*V[vx2]
        F[mvx2] = V[rho]*V[vx2]**2 + V[prs]
        F[eng] = V[vx2]*(U[eng] + V[prs])

    if p['Dimensions'] == '3D' and axis == 'i':
        F[rho] = V[rho]*V[vx1]
        F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
        F[mvx2] = V[rho]*V[vx1]*V[vx2]
        F[mvx3] = V[rho]*V[vx1]*V[vx3]
        F[eng] = V[vx1]*(U[eng] + V[prs])
    elif p['Dimensions'] == '3D' and axis == 'j':
        F[rho] = V[rho]*V[vx2]
        F[mvx1] = V[rho]*V[vx1]*V[vx2]
        F[mvx2] = V[rho]*V[vx2]**2 + V[prs]
        F[mvx3] = V[rho]*V[vx2]*V[vx3]
        F[eng] = V[vx2]*(U[eng] + V[prs])
    elif p['Dimensions'] == '3D' and axis == 'k':
        F[rho] = V[rho]*V[vx3]
        F[mvx1] = V[rho]*V[vx1]*V[vx3]
        F[mvx2] = V[rho]*V[vx2]*V[vx3]
        F[mvx3] = V[rho]*V[vx3]**2 + V[prs]
        F[eng] = V[vx3]*(U[eng] + V[prs])

    return F


def riennman(g, p, axis):
    """
    Synopsis
    --------
    Execute the riemann solver specified by 
    user via the p (parameters) dic. 

    Args
    ----
    g: object-like
    object containing all variables related to 
    the grid, e.g. cell width.

    p: dic-like
    dictionary of user defined ps, e.g. 
    maximum simulation time.        

    axis: chr-like
    Character specifying which axis the loop 
    is over.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    if p['riemann'] == 'tvdlf':
        tvdlf(g)
    elif p['riemann'] == 'hll':
        hll(g.flux.T, g.SL, g.SR, g.FL.T, g.FR.T, g.UL.T, g.UR.T)
    elif p['riemann'] == 'hllc':
        hllc(g.flux.T, g.SL, g.SR, g.FL.T, g.FR.T, g.UL.T, 
             g.UR.T, g.VL.T, g.VR.T, p, axis)
    else:
        print('Error: invalid riennman solver.')
        sys.exit()

    if np.isnan(np.sum(s.flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return


def reconstruction(y, g, p, axis):

    if axis == 'i':
        dxi = g.dx1
    if axis == 'j':
        dxi = g.dx2
    if axis == 'k':
        dxi = g.dx3

    if p['reconstruction'] == 'flat':
        L, R = flat(y, g)
    elif p['reconstruction'] == 'linear':
        L, R = minmod(y, g.gz, dxi)  
    else:
        print('Error: Invalid reconstructor.')
        sys.exit()

    if np.isnan(np.sum(L)) or np.isnan(np.sum(R)):
        print("Error, nan in array, function: reconstruction")
        sys.exit()

    return L, R


def face_flux(U, s, g, p, axis):
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

    s: object-like
    object containing all the fluxes needed to 
    evolve the solution.

    g: object-like
    object containing all variables related to 
    the grid, e.g. cell width.

    p: dic-like
    dictionary of user defined ps, e.g. 
    maximum simulation time.        

    axis: chr-like
    Character specifying which axis the loop 
    is over.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    g.build(axis)

    g.UL, g.UR = reconstruction(U, g, p, axis)

    g.VL = cons_to_prims(g.UL, p)
    g.VR = cons_to_prims(g.UR, p)

    g.FL = flux_tensor(g.UL, p, axis)
    g.FR = flux_tensor(g.UR, p, axis)

    g.SL, g.SR = eigenvalues(g.UL, g.UR, p, axis)

    riennman(g, p, axis)

    if np.isnan(np.sum(g.flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return g.flux


def RHSOperator(U, s, g, p):
    """
    Synopsis
    --------
    Evolve the simulation domain though one 
    time step.

    Args
    ----
    U: numpy array-like
    state vector containing all 
    conservative variables.

    s: object-like
    object containing all the fluxes needed to 
    evolve the solution.

    g: object-like
    object containing all variables related to 
    the grid, e.g. cell width.

    p: dic-like
    dictionary of user defined ps, e.g. 
    maximum simulation time.        

    axis: chr-like
    Character specifying which axis the loop 
    is over.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    if p['Dimensions'] == '1D':

        dflux_x1 = np.zeros(shape=U.shape)

        face_fluxes = face_flux(U, s, g, p, 'i')

        Fneg = face_fluxes[:, :-1]
        Fpos = face_fluxes[:, 1:]

        dflux_x1[:, g.ibeg:g.iend] = -(Fpos - Fneg)
        dflux = dflux_x1/g.dx1

    if p['Dimensions'] == '2D':

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)

        for j in range(g.jbeg, g.jend):

            face_flux_x1 = face_flux(U[:, j, :], s, g, p, 'i')

            Fneg = face_flux_x1[:, :-1]
            Fpos = face_flux_x1[:, 1:]

            dflux_x1[:, j, g.ibeg:g.iend] = -(Fpos - Fneg)/g.dx1

        for i in range(g.ibeg, g.iend):

            face_flux_x2 = face_flux(U[:, :, i], s, g, p, 'j')

            Fneg = face_flux_x2[:, :-1]
            Fpos = face_flux_x2[:, 1:]

            dflux_x2[:, g.jbeg:g.jend, i] = -(Fpos - Fneg)/g.dx2

        dflux = dflux_x1 + dflux_x2

    if p['Dimensions'] == '3D':

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)
        dflux_x3 = np.zeros(shape=U.shape)

        for k in range(g.kbeg, g.kend):
            for j in range(g.jbeg, g.jend):

                face_flux_x1 = face_flux(U[:, k, j, :], s, g, p, 'i')

                Fneg = face_flux_x1[:, :-1]
                Fpos = face_flux_x1[:, 1:]

                dflux_x1[:, k, j, g.ibeg:g.iend] = -(Fpos - Fneg)/g.dx1

        for k in range(g.kbeg, g.kend):
            for i in range(g.ibeg, g.iend):

                face_flux_x2 = face_flux(U[:, k, :, i], s, g, p, 'j')

                Fneg = face_flux_x2[:, :-1]
                Fpos = face_flux_x2[:, 1:]

                dflux_x2[:, k, g.jbeg:g.jend, i] = -(Fpos - Fneg)/g.dx2

        for j in range(g.jbeg, g.jend): 
            for i in range(g.ibeg, g.iend):

                face_flux_x3 = face_flux(U[:, :, j, i], s, g, p, 'k')

                Fneg = face_flux_x3[:, :-1]
                Fpos = face_flux_x3[:, 1:]

                dflux_x3[:, g.kbeg:g.kend, j, i] = -(Fpos - Fneg)/g.dx3

        dflux = dflux_x1 + dflux_x2 + dflux_x3

    if np.isnan(np.sum(dflux)):
        print("Error, nan in array, function: flux")
        sys.exit()

    return dflux


def incriment(V, dt, s, g, p):
    """
    Synopsis
    --------
    Evolve the simulation domain though one 
    time step.

    Args
    ----
    V: numpy array-like
    State vector containing the hole solution 
    and all variables

    dt: double-like
    Time step, in simulation units.

    s: object-like
    object containing all the fluxes needed to 
    evolve the solution.

    g: object-like
    Object containing all variables related to 
    the grid, e.g. cell width.

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

    U = prims_to_cons(V, p)

    if p['time stepping'] == 'Euler':
        U_new = U + dt*RHSOperator(U, s, g, p)
        g.boundary(U_new, p)
        
    elif p['time stepping'] == 'RK2':
        K1 = dt*RHSOperator(U, s, g, p)
        g.boundary(K1, p)
        K2 = dt*RHSOperator(U+K1, s, g, p)
        U_new = U + 0.5*(K1 + K2) 
        g.boundary(U_new, p)
       
    else:
        print('Error: Invalid integrator.')
        sys.exit()

    V = cons_to_prims(U_new, p)

    if np.isnan(np.sum(V)):
        print("Error, nan in array, function: incriment")
        sys.exit()

    return V


