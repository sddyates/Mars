
import sys
import numpy as np
from settings import *
from cython_lib.solvers import hll, hllc, tvdlf
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
        F[rho] = U[mvx1]
        F[mvx1] = U[mvx1]*V[vx1]
        F[eng] = V[vx1]*(U[eng] + V[prs])
    elif p['Dimensions'] == '2D' and axis == 'i':
        F[rho] = U[mvx1]
        F[mvx1] = U[mvx1]*V[vx1]
        F[mvx2] = U[mvx2]*V[vx1]
        F[eng] = V[vx1]*(U[eng] + V[prs])
    elif p['Dimensions'] == '2D' and axis == 'j':
        F[rho] = U[mvx2]
        F[mvx1] = U[mvx1]*V[vx2]
        F[mvx2] = U[mvx2]*V[vx2]
        F[eng] = V[vx2]*(U[eng] + V[prs])
    elif p['Dimensions'] == '3D' and axis == 'i':
        F[rho] = U[mvx1]
        F[mvx1] = U[mvx1]*V[vx1]
        F[mvx2] = U[mvx2]*V[vx1]
        F[mvx3] = U[mvx3]*V[vx1]
        F[eng] = V[vx1]*(U[eng] + V[prs])
    elif p['Dimensions'] == '3D' and axis == 'j':
        F[rho] = U[mvx2]
        F[mvx1] = U[mvx1]*V[vx2]
        F[mvx2] = U[mvx2]*V[vx2]
        F[mvx3] = U[mvx3]*V[vx2]
        F[eng] = V[vx2]*(U[eng] + V[prs])
    elif p['Dimensions'] == '3D' and axis == 'k':
        F[rho] = U[mvx3]
        F[mvx1] = U[mvx1]*V[vx3]
        F[mvx2] = U[mvx2]*V[vx3]
        F[mvx3] = U[mvx3]*V[vx3]
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
        tvdlf(g, p, axis)
    elif p['riemann'] == 'hll':
        #Geometry.riemann[p['riemann']](g.flux.T, g.pres.T, g.SL, g.SR, g.FL.T, g.FR.T, g.UL.T, g.UR.T, 
        #    g.VL.T, g.VR.T)
        hll(g.flux.T, g.pres.T, g.SL, g.SR, g.FL.T, g.FR.T, g.UL.T, g.UR.T, 
            g.VL.T, g.VR.T, p, axis)
    elif p['riemann'] == 'hllc':
        hllc(g.flux.T, g.pres.T, g.SL, g.SR, g.FL.T, g.FR.T, g.UL.T, g.UR.T, 
            g.VL.T, g.VR.T, p, axis)
    else:
        print('Error: invalid riennman solver.')
        sys.exit()

    if np.isnan(np.sum(g.flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return


def reconstruction(V, g, p, axis):

    if axis == 'i':
        dxi = g.dx1
    if axis == 'j':
        dxi = g.dx2
    if axis == 'k':
        dxi = g.dx3

    if p['reconstruction'] == 'flat':
        L, R = flat(V, g)
    elif p['reconstruction'] == 'linear':
        L, R = minmod(V, g.gz, dxi)  
    else:
        print('Error: Invalid reconstructor.')
        sys.exit()

    if np.isnan(np.sum(L)) or np.isnan(np.sum(R)):
        print("Error, nan in array, function: reconstruction")
        sys.exit()

    return L, R


def face_flux(U, g, p, axis):
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

    g.build_fluxes(axis)

    V = cons_to_prims(U, p)

    g.VL, g.VR = reconstruction(V, g, p, axis)

    g.UL = prims_to_cons(g.VL, p)
    g.UR = prims_to_cons(g.VR, p)

    g.FL = flux_tensor(g.UL, p, axis)
    g.FR = flux_tensor(g.UR, p, axis)

    g.SL, g.SR = eigenvalues(g.UL, g.UR, p, axis)

    riennman(g, p, axis)

    if np.isnan(np.sum(g.flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return g.flux


def RHSOperator(U, g, p):
    """
    Synopsis
    --------
    Determine the right hand side operator.

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

        face_fluxes = face_flux(U, g, p, 'i')

        dflux_x1[:, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
        dflux_x1[mvx1, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1

    if p['Dimensions'] == '2D':

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)

        for j in range(g.jbeg, g.jend):

            face_flux_x1 = face_flux(U[:, j, :], g, p, 'i')

            Fneg = face_flux_x1[:, :-1]
            Fpos = face_flux_x1[:, 1:]

            dflux_x1[:, j, g.ibeg:g.iend] = -(Fpos - Fneg)
            dflux_x1[mvx1, j, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        for i in range(g.ibeg, g.iend):

            face_flux_x2 = face_flux(U[:, :, i], g, p, 'j')

            Fneg = face_flux_x2[:, :-1]
            Fpos = face_flux_x2[:, 1:]

            dflux_x2[:, g.jbeg:g.jend, i] = -(Fpos - Fneg)
            dflux_x2[mvx2, g.jbeg:g.jend, i] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1 + dflux_x2/g.dx2

    if p['Dimensions'] == '3D':

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)
        dflux_x3 = np.zeros(shape=U.shape)

        for k in range(g.kbeg, g.kend):
            for j in range(g.jbeg, g.jend):

                face_flux_x1 = face_flux(U[:, k, j, :], g, p, 'i')

                Fneg = face_flux_x1[:, :-1]
                Fpos = face_flux_x1[:, 1:]

                dflux_x1[:, k, j, g.ibeg:g.iend] = -(Fpos - Fneg)
                dflux_x1[mvx1, k, j, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        for k in range(g.kbeg, g.kend):
            for i in range(g.ibeg, g.iend):

                face_flux_x2 = face_flux(U[:, k, :, i], g, p, 'j')

                Fneg = face_flux_x2[:, :-1]
                Fpos = face_flux_x2[:, 1:]

                dflux_x2[:, k, g.jbeg:g.jend, i] = -(Fpos - Fneg)
                dflux_x1[mvx2, k, g.jbeg:g.jend, i] -= g.pres[1:] - g.pres[:-1]

        for j in range(g.jbeg, g.jend): 
            for i in range(g.ibeg, g.iend):

                face_flux_x3 = face_flux(U[:, :, j, i], g, p, 'k')

                Fneg = face_flux_x3[:, :-1]
                Fpos = face_flux_x3[:, 1:]

                dflux_x3[:, g.kbeg:g.kend, j, i] = -(Fpos - Fneg)
                dflux_x1[mvx3, g.kbeg:g.kend, j, i] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1 + dflux_x2/g.dx2 + dflux_x3/g.dx3

    if np.isnan(np.sum(dflux)):
        print("Error, nan in array, function: flux")
        sys.exit()

    return dflux


def incriment(V, dt, g, p):
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
        U_new = U + dt*RHSOperator(U, g, p)
        g.boundary(U_new, p)
        
    elif p['time stepping'] == 'RK2':
        K1 = dt*RHSOperator(U, g, p)
        g.boundary(K1, p)
        # My need to recalculate the time step here.
        K2 = dt*RHSOperator(U+K1, g, p)
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


