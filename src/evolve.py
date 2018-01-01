import sys
import numpy as np
import matplotlib.pyplot as plt

from globe import *
from output import mesh_plot, line_plot
from piecewise import reconstruction
from solver import riennman
from tools import *


def flux_tensor(U, p, axis):

    F = np.zeros(shape=U.shape)

    V = cons_to_prims(U, p)

    if p['Dimensions'] == '1D':
        F[rho] = V[rho]*V[vx1]
        F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
        F[eng] = V[vx1]*(U[eng] + V[prs])

    if axis == 'i' and p['Dimensions'] == '2D':
        F[rho] = V[rho]*V[vx1]
        F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
        F[mvx2] = V[rho]*V[vx1]*V[vx2]
        F[eng] = V[vx1]*(U[eng] + V[prs])
    elif axis == 'j' and p['Dimensions'] == '2D':
        F[rho] = V[rho]*V[vx2]
        F[mvx1] = V[rho]*V[vx1]*V[vx2]
        F[mvx2] = V[rho]*V[vx2]**2 + V[prs]
        F[eng] = V[vx2]*(U[eng] + V[prs])

    return F


def face_flux(U, s, g, p, axis):

    s.build(g, axis)

    s.UL, s.UR = reconstruction(U, g, p, axis)

    s.VL = cons_to_prims(s.UL, p)
    s.VR = cons_to_prims(s.UR, p)

    s.FL = flux_tensor(s.UL, p, axis)
    s.FR = flux_tensor(s.UR, p, axis)

    s.SL, s.SR = eigenvalues(s.UL, s.UR, p, axis)

    riennman(s, g, p, axis)

    if np.isnan(np.sum(s.flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return s.flux


def RHSOperator(U, s, g, p):

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

    if np.isnan(np.sum(dflux)):
        print("Error, nan in array, function: flux")
        sys.exit()

    return dflux


def incriment(V, dt, flux, g, p):
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
        U_new = U + dt*RHSOperator(U, flux, g, p)
        
    elif p['time stepping'] == 'RK2':
        K1 = dt*RHSOperator(U, flux, g, p)
        g.boundary(K1, p)
        K2 = dt*RHSOperator(U+K1, flux, g, p)
        U_new = U + 0.5*(K1 + K2) 
        
    else:
        print('Error: Invalid integrator.')
        sys.exit()

    V = cons_to_prims(U_new, p)

    if np.isnan(np.sum(V)):
        print("Error, nan in array, function: incriment")
        sys.exit()

    return V


