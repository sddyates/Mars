import sys
import numpy as np
import matplotlib.pyplot as plt

from globe import *
from output import mesh_plot, line_plot
from piecewise import reconstruction
from solver import riennman
from tools import *

class Evolve:

    def __init__(self):
        None

    def incriment(self, V, dt, g, p):
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

        U = prims_to_cons(V, p['gamma'], p['Dimensions'])

        if p['time stepping'] == 'Euler':
            U_new = U + dt*self._RHSOperator(U, g, p)
        
        elif p['time stepping'] == 'RK2':

            U_star = U + 0.5*dt*self._RHSOperator(U, g, p)
            g.boundary(U_star, p)
            U_new = 0.5*(U + U_star + dt*self._RHSOperator(U_star, g, p))
        
        else:
            print('Error: Invalid integrator.')
            sys.exit()

        V = cons_to_prims(U_new, p['gamma'], p['Dimensions'])

        if np.isnan(np.sum(V)):
            print("Error, nan in array, function: incriment")
            sys.exit()

        return V


    def _RHSOperator(self, U, g, p):

        dflux_x1 = np.zeros(shape=U.shape)

        if p['Dimensions'] == '1D':
            face_fluxes = self._riemann(U, g, p, 'i')
            Fneg = face_fluxes[:, :-1]
            Fpos = face_fluxes[:, 1:]

            dflux_x1[:, g.ibeg:g.iend] = -(Fpos - Fneg)
            #dflux_x1[mvx1, g.ibeg:g.iend] -= (press[1:] - press[:-1])
            dflux = dflux_x1/g.dx1

        if p['Dimensions'] == '2D':
            dflux_x2 = np.zeros(shape=U.shape)
            # Transvers loop over columns in x1.
            for j in range(g.jbeg, g.jend):
                face_flux_x1 = self._riemann(U[:, j, :], g, p, 'i')
                Fneg = face_flux_x1[:, :-1]
                Fpos = face_flux_x1[:, 1:]
                dflux_x1[:, j, g.ibeg:g.iend] = -(Fpos - Fneg)/g.dx1

            # Transvers loop over columns in x2.
            for i in range(g.ibeg, g.iend):
                face_flux_x2 = self._riemann(U[:, :, i], g, p, 'j')
                Fneg = face_flux_x2[:, :-1]
                Fpos = face_flux_x2[:, 1:]
                dflux_x2[:, g.jbeg:g.jend, i] = -(Fpos - Fneg)/g.dx2

            dflux = dflux_x1 + dflux_x2

        if np.isnan(np.sum(dflux)):
            print("Error, nan in array, function: flux")
            sys.exit()

        return dflux


    def _riemann(self, U, g, p, axis):

        #Initialise arrays to hold flux vales.
        if axis == 'i':
            flux = np.zeros(shape=g.shape_flux_x1)
            FR = np.zeros(shape=g.shape_flux_x1)
            FL = np.zeros(shape=g.shape_flux_x1)
        if axis == 'j':
            flux = np.zeros(shape=g.shape_flux_x2)
            FR = np.zeros(shape=g.shape_flux_x2)
            FL = np.zeros(shape=g.shape_flux_x2)

        # Obtain the states on the left (L) and right (R) faces of 
        # the cell for the conservative variables.

        UL, UR = reconstruction(U, g, axis, p['reconstruction'])
        #UL, UR = self._reconstruction(U, g, p, axis)

        VL = cons_to_prims(UL, p['gamma'], p['Dimensions'])
        VR = cons_to_prims(UR, p['gamma'], p['Dimensions'])

        # Construct the flux tensor 
        # either side of interface.
        FL = self._flux_tensor(UL, axis, p['gamma'], p['Dimensions'])
        FR = self._flux_tensor(UR, axis, p['gamma'], p['Dimensions'])

        SL, SR = eigenvalues(UL, UR, p['gamma'], axis, p['Dimensions'])

        riennman(flux, FL, FR, SL, SR, UL, UR, 
                 VL, VR, g, axis, 
                 p['gamma'], p['Dimensions'], p['riemann'])

        if np.isnan(np.sum(flux)):
            print("Error, nan in array, function: riemann")
            sys.exit()

        return flux


    def _flux_tensor(self, U, axis, gamma, dim):

        F = np.zeros(shape=U.shape)

        V = cons_to_prims(U, gamma, dim)

        if dim == '1D':
            F[rho] = V[rho]*V[vx1]
            F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
            F[eng] = V[vx1]*(U[eng] + V[prs])

        if axis == 'i' and dim == '2D':
            F[rho] = V[rho]*V[vx1]
            F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
            F[mvx2] = V[rho]*V[vx1]*V[vx2]
            F[eng] = V[vx1]*(U[eng] + V[prs])
        elif axis == 'j' and dim == '2D':
            F[rho] = V[rho]*V[vx2]
            F[mvx1] = V[rho]*V[vx1]*V[vx2]
            F[mvx2] = V[rho]*V[vx2]**2 + V[prs]
            F[eng] = V[vx2]*(U[eng] + V[prs])

        return F

